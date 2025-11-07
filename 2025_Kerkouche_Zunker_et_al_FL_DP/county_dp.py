import os
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from collections import defaultdict, OrderedDict
import random
import copy
from tqdm.auto import tqdm
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
import numpy as np
from datetime import datetime
from community_dp import (
    RegressionDataset,
    get_update,
    reduce_weights_divide,
    update_global_model,
    clip,
    get_noise_multiplier,
)


try:
    from IPython.display import display
except ImportError:
    display = print

# --- Global Parameters ---
EPOCHS = 30
LEARNING_RATE = 0.001
Nbr_selected_Counties = 100
FED_rounds = 75
sensitivity = 0.5
DELTA = 10**-5
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
day_f = 7
day_before = 10
p_train = 0.9
scale_to_relative = False
daily_data = True  # Use daily data instead of accumulated
SEED = 1832
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if scale_to_relative:
    sensitivity = sensitivity / 1000

year = 2020

start_date = '2022-03-01'
end_date = '2022-04-01'

if year == 2020:
    start_date = '2020-11-01'
    end_date = '2020-12-01'

# --- Data Loading and Preprocessing ---


# --- County-Specific Data Loading and Preprocessing ---
DATA_BASE_PATH = os.path.join(os.getcwd(), "casedata")
COUNTY_CASES_FILE = os.path.join(DATA_BASE_PATH, "cases_all_county_ma7.json")
COUNTY_POP_FILE = os.path.join(DATA_BASE_PATH, "county_population.json")
path_json = COUNTY_CASES_FILE


def load_county_data(path):
    """Loads county data from JSON."""
    df = pd.read_json(path)
    df = df.sort_values(by=['ID_County', 'Date'])
    # Ensure Date is datetime object if not already
    df['Date'] = pd.to_datetime(df['Date'])
    # delete all dates which are not in the range [start_date, end_date]
    # also the data is accumulated, we want daily
    if daily_data:
        start_date_dt = pd.to_datetime(start_date)
        start_date_dt_m1 = start_date_dt - pd.Timedelta(days=1)
        df = df[(df['Date'] >= start_date_dt_m1) & (
            df['Date'] <= pd.to_datetime(end_date))].reset_index(drop=True)
        # Remove accumulated data, keep only daily changes
        df['Confirmed'] = df.groupby('ID_County')['Confirmed'].diff()
        # delete m1 date
        df = df[(df['Date'] >= start_date) & (
            df['Date'] <= end_date)].reset_index(drop=True)
    else:
        df = df[(df['Date'] >= start_date) & (
            df['Date'] <= end_date)].reset_index(drop=True)
    return df.to_numpy()

# Adapted preprocessing function from County notebook, aligned with community_dp structure


def preprocess_county_dataset(a, day_f, day_before):
    """
    Preprocesses county data to predict 'day_f' days ahead using 'day_before' historical days.
    """
    X = []
    Y = []

    # Assuming 'a' has columns: [Date (datetime), ID_County, Confirmed] after loading
    dates = a[:, 0]
    county_ids = a[:, 1]
    counts = a[:, 2]
    n_samples = len(a)

    for k in range(day_before, n_samples - day_f):
        current_county_id = county_ids[k]

        # --- Target Check ---
        target_idx = k + day_f
        # Check if target date is exactly day_f days after current date (k)
        # and if the county ID is the same
        if (dates[target_idx] - dates[k]).days != day_f or \
           county_ids[target_idx] != current_county_id:
            continue

        # --- Generate Historical Features ---
        historical_cases = []
        valid_history = True

        temp_history = []
        for i in range(day_before):
            hist_idx = k - i  # Index going back in time
            prev_hist_idx = hist_idx - 1

            # Check date continuity (backwards) and location consistency
            if prev_hist_idx < 0 or \
                (dates[hist_idx] - dates[prev_hist_idx]).days != 1 or \
                county_ids[hist_idx] != current_county_id or \
                    county_ids[prev_hist_idx] != current_county_id:  # Check previous point's county too
                valid_history = False
                break
            temp_history.append(float(counts[hist_idx]))

        if not valid_history:
            continue  # Skip if history is broken

        # Reverse to get chronological order
        historical_cases = list(reversed(temp_history))

        # --- Create Feature Vector and Label ---
        # Feature format: [County_ID (for splitting), Month, Hist_1, ..., Hist_day_before]
        tab_keep = [
            # ID_County
            int(current_county_id),
            int(dates[k].month),         # Month at time k
            *historical_cases
        ]
        X.append(tab_keep)
        Y.append(float(counts[target_idx]))  # Target value

    print(f"Preprocessing done. Generated {len(X)} samples.")
    if not X:
        # Return empty arrays if no samples generated
        return np.array([]), np.array([])
    return np.array(X), np.array(Y)


def scale_by_population(X_processed, Y_processed, path_pop_data):
    """
    Scales case counts by population size.
    """
    print("Scaling data relative to population...")
    pop_data = pd.read_json(path_pop_data)
    pop_data = pop_data.iloc[:, [0, 1]]

    X_scaled = copy.deepcopy(X_processed)
    Y_scaled = copy.deepcopy(Y_processed)

    for i in range(len(X_scaled)):
        county_id = int(X_scaled[i, 0])
        population = pop_data[pop_data['ID_County']
                              == county_id]['Population'].values[0]
        if population is not None and population > 0:
            X_scaled[i, 2:] /= population
            Y_scaled[i] /= population
        elif population == 0:
            print(
                f"Warning: Population for ID {county_id} not found. Scaling skipped.")

    return X_scaled, Y_scaled


def unscale_data(y_pred, y_true, x_test, test_county_ids, path_pop_data):
    """ Scales predictions, true values, and inputs back using population data. """
    pop_data = pd.read_json(path_pop_data)
    pop_data = pop_data.set_index('ID_County')['Population']

    y_pred_unscaled = np.zeros_like(y_pred)
    y_true_unscaled = np.zeros_like(y_true)
    x_test_unscaled = np.zeros_like(x_test, dtype=float)

    for i in range(len(test_county_ids)):
        county_id = int(test_county_ids[i])
        population = pop_data.get(county_id)
        if population is not None and population > 0:
            y_pred_unscaled[i] = y_pred[i] * population
            y_true_unscaled[i] = y_true[i] * population
            x_test_unscaled[i] = x_test[i] * population
        else:
            print(
                f"Warning: Population for ID {county_id} not found. Using scaled values.")
            y_pred_unscaled[i] = y_pred[i]
            y_true_unscaled[i] = y_true[i]
            x_test_unscaled[i] = x_test[i]

    return y_pred_unscaled, y_true_unscaled, x_test_unscaled


# --- County-Specific Data Splitting ---


def split_county_data(X_processed, Y_processed, p_train=0.9):
    """Splits data based on County ID."""
    Dic_county = defaultdict(list)
    # Group indices by County ID (first column of X_processed)
    for i in range(len(X_processed)):
        county_id = str(int(X_processed[i, 0]))
        Dic_county[county_id].append(i)

    keys_county = list(Dic_county.keys())

    Dic_indices_train_county = {}
    list_indices_test_county = np.array([], dtype=int)

    for key in keys_county:
        indices = np.array(Dic_county.get(key))
        if len(indices) > 10:
            # Shuffle indices within the county before splitting
            np.random.shuffle(indices)
            split_idx = int(len(indices) * p_train)
            train_indices = indices[:split_idx]
            test_indices = indices[split_idx:]

            # Store as list
            Dic_indices_train_county[key] = train_indices.tolist()
            list_indices_test_county = np.concatenate(
                (list_indices_test_county, test_indices))
        else:
            print(
                f"Skipping County {key}: Only {len(indices)} samples (<= 10).")
            pass  # Skip counties with too few samples

    list_indices_test_county = list_indices_test_county.astype(int)
    # county ids for test indices
    test_county_ids = X_processed[list_indices_test_county, 0]

    # Remove County ID and Month columns (indices 0 and 1) from features after splitting
    if X_processed.shape[0] > 0:
        X_final = np.delete(X_processed, [0, 1], 1)
    else:
        X_final = np.array([])

    print(
        f"Splitting done. Training counties: {len(Dic_indices_train_county)}, Test samples: {len(list_indices_test_county)}")
    return X_final, Y_processed, Dic_indices_train_county, list_indices_test_county, test_county_ids


# --- Experiment Function ---
def run_county_experiment(epsilon_value, dataset, list_indices_test, test_county_ids_for_unscaling, run_idx, Dic_indices_train_county, keys_train_county, NUM_FEATURES, device):
    print(
        f"\n--- Running County Experiment for Epsilon = {epsilon_value if epsilon_value != float('inf') else 'Non-DP'} (Run {run_idx+1}) ---")

    # --- DP Setup ---
    if epsilon_value == float('inf'):
        DP_flag = 0
        noise_multiplier = 0.0
        print("DP Flag: OFF")
    else:
        DP_flag = 1
        num_train_counties = len(keys_train_county)
        # Use Nbr_selected_Counties global variable
        current_sampling_prob = Nbr_selected_Counties / \
            num_train_counties if num_train_counties > 0 else 0

        if current_sampling_prob <= 0 or current_sampling_prob > 1:
            print(
                f"Warning: Invalid sampling probability ({current_sampling_prob:.4f}). Check Nbr_selected_Counties ({Nbr_selected_Counties}) and number of training counties ({num_train_counties}).")
            noise_multiplier = float('inf')
        else:
            try:
                # Use imported RDP accountant function
                noise_multiplier = get_noise_multiplier(
                    target_epsilon=epsilon_value,
                    target_delta=DELTA,
                    sample_rate=current_sampling_prob,
                    steps=FED_rounds
                )
                print(
                    f"Calculated Noise Multiplier: {noise_multiplier:.4f} for Epsilon: {epsilon_value}, Sampling Prob: {current_sampling_prob:.4f}")
            except NameError:
                print(
                    "Error: get_noise_multiplier function not found.")
                noise_multiplier = float('inf')
            except Exception as e:
                print(f"Error calculating noise multiplier: {e}")
                noise_multiplier = float('inf')

        print(
            f"DP Flag: ON, Epsilon: {epsilon_value}, Noise Multiplier: {noise_multiplier:.4f}")

    # --- Model Definition (Define inside function to reset weights each time) ---
    model = nn.Sequential(
        nn.Linear(NUM_FEATURES, 128), nn.ReLU(),
        nn.Linear(128, 64), nn.ReLU(),
        nn.Linear(64, 32), nn.ReLU(),
        nn.Linear(32, 1)
    ).to(device)

    ref_model = copy.deepcopy(model)

    # --- County-specific datasets (NO SCALING) ---
    County_datasets = {}
    for county_key in keys_train_county:
        indices = Dic_indices_train_county.get(county_key, [])
        if indices:
            # Create a subset for this county directly from the main dataset
            County_datasets[county_key] = Subset(dataset, indices)

    # --- Test Dataset ---
    test_dataset = Subset(dataset, list_indices_test)

    # --- Training Loop ---
    if not keys_train_county:
        print("Skipping training: No valid counties found.")
    else:
        num_train_counties = len(keys_train_county)
        for e in tqdm(range(1, FED_rounds + 1), desc=f"Federated rounds (County, Eps={epsilon_value if epsilon_value != float('inf') else 'Non-DP'})", leave=False):
            # Select Counties
            selected_Counties = []  # Initialize empty
            if DP_flag == 1 and current_sampling_prob > 0 and noise_multiplier != float('inf'):
                # Ensure p sums to 1
                prob_select = min(max(current_sampling_prob, 0.0), 1.0)
                selected_county_indices = np.random.choice(
                    a=[False, True], size=num_train_counties, p=[1 - prob_select, prob_select]
                )
                selected_Counties = np.array(keys_train_county)[
                    selected_county_indices]
            elif not DP_flag and num_train_counties > 0:
                num_to_select = min(
                    int(Nbr_selected_Counties), num_train_counties)
                if num_to_select > 0:
                    selected_Counties = np.random.choice(
                        np.array(keys_train_county), num_to_select, replace=False)

            if len(selected_Counties) == 0:
                continue

            local_updates = OrderedDict()

            for i, county_key in enumerate(selected_Counties):
                # Start local model from global
                model_local = copy.deepcopy(ref_model)
                criterion = nn.MSELoss()
                optimizer = optim.Adam(
                    model_local.parameters(), lr=LEARNING_RATE)

                # Use the locally scaled dataset for this County if available
                if county_key in County_datasets:
                    Sub_train = County_datasets[county_key]
                    # Ensure batch size is at least 1
                    BATCH_SIZE = max(1, len(Sub_train))
                    train_loader = DataLoader(
                        dataset=Sub_train, batch_size=BATCH_SIZE, shuffle=True)
                else:
                    # Fallback to unscaled data if County wasn't scaled
                    indices_county_train = Dic_indices_train_county.get(
                        county_key)
                    if not indices_county_train:
                        continue
                    Sub_train = Subset(dataset, indices_county_train)
                    # Ensure batch size is at least 1
                    BATCH_SIZE = max(1, len(indices_county_train))
                    train_loader = DataLoader(
                        dataset=Sub_train, batch_size=BATCH_SIZE, shuffle=True)

                model_local.train()
                for _ in range(0, EPOCHS):
                    for X_train_batch, y_train_batch in train_loader:
                        X_train_batch, y_train_batch = X_train_batch.to(
                            device), y_train_batch.to(device)
                        optimizer.zero_grad()
                        y_train_pred = model_local(X_train_batch)
                        train_loss = criterion(
                            y_train_pred, y_train_batch.unsqueeze(1))
                        train_loss.backward()
                        optimizer.step()

                # Calculate update
                update = get_update(ref_model, model_local)

                # Clip update if DP is enabled
                if DP_flag == 1 and noise_multiplier != float('inf'):
                    update = clip(update, sensitivity)

                # Aggregate clipped (but not yet noisy) updates
                if not local_updates:
                    local_updates = copy.deepcopy(update)
                else:
                    for key in update:
                        local_updates[key] += update[key]

            # Average aggregated updates
            if local_updates:
                averaged_updates = reduce_weights_divide(
                    local_updates, Nbr_selected_Counties)

                # Add noise after averaging if DP is enabled and noise multiplier is valid
                if DP_flag == 1 and noise_multiplier > 0 and noise_multiplier != float('inf'):
                    noise_std = (sensitivity * noise_multiplier) / \
                        Nbr_selected_Counties
                    final_update = copy.deepcopy(averaged_updates)
                    for key in final_update:
                        noise = torch.normal(
                            mean=0.0, std=noise_std, size=final_update[key].size()).to(device)
                        final_update[key] += noise
                else:
                    final_update = averaged_updates  # No noise if DP off or multiplier invalid

                # Update global model
                global_weights = update_global_model(
                    ref_model.state_dict(), final_update)
                ref_model.load_state_dict(global_weights)

    # --- Evaluation ---
    # Use the scaled test dataset
    Sub_test = test_dataset
    # Use a reasonable batch size
    test_loader = DataLoader(dataset=Sub_test, batch_size=64)

    y_pred_list = []
    y_true_list = []
    x_test_list = []
    final_model = ref_model
    final_model.eval()
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            x_test_list.extend(X_batch.cpu().numpy())
            X_batch = X_batch.to(device)
            y_test_pred = final_model(X_batch)  # Shape: [B, 1]
            y_pred_list.extend(y_test_pred.cpu().numpy().squeeze())
            y_true_list.extend(y_batch.cpu().numpy())

    # Convert lists to 1D numpy arrays
    y_pred_np = np.array(y_pred_list)
    y_true_np = np.array(y_true_list)
    x_test_np = np.array(x_test_list)

    if scale_to_relative:
        y_pred_unscaled_np, y_true_unscaled_np, x_test_unscaled_np = unscale_data(
            y_pred_np, y_true_np, x_test_np, test_county_ids_for_unscaling, COUNTY_POP_FILE
        )
    else:
        y_pred_unscaled_np = y_pred_np
        y_true_unscaled_np = y_true_np
        x_test_unscaled_np = x_test_np

    # --- Calculate Metrics ---
    # single mape per test sample
    mape_individuals = []
    mask = y_true_unscaled_np != 0
    if np.any(mask):
        mape_individuals = np.abs(
            (y_true_unscaled_np[mask] - y_pred_unscaled_np[mask]) / y_true_unscaled_np[mask]) * 100

    mse, mae, mape, mdape, r_square = np.nan, np.nan, np.nan, np.nan, np.nan
    if y_true_unscaled_np.size > 0 and y_pred_unscaled_np.size == y_true_unscaled_np.size:
        try:
            mse = mean_squared_error(y_true_unscaled_np, y_pred_unscaled_np)
            mae = mean_absolute_error(y_true_unscaled_np, y_pred_unscaled_np)
            mask = y_true_unscaled_np != 0
            if np.any(mask):
                mape = np.mean(np.abs(
                    (y_true_unscaled_np[mask] - y_pred_unscaled_np[mask]) / y_true_unscaled_np[mask])) * 100
            elif np.allclose(y_true_unscaled_np, y_pred_unscaled_np):
                mape = 0.0
            else:
                mape = np.inf

            # Calculate MdAPE (Median Absolute Percentage Error)
            mask_mdape = y_true_unscaled_np != 0
            if np.any(mask_mdape):
                percentage_errors = np.abs(
                    (y_true_unscaled_np[mask_mdape] - y_pred_unscaled_np[mask_mdape]) / y_true_unscaled_np[mask_mdape]) * 100
                mdape = np.median(percentage_errors)
            # All true are zero and all pred are zero
            elif np.allclose(y_true_unscaled_np, y_pred_unscaled_np):
                mdape = 0.0
            # All true are zero, but predictions are not (implies infinite error)
            else:
                mdape = np.inf

            r_square = r2_score(y_true_unscaled_np, y_pred_unscaled_np)
        except Exception as e:
            print(f"Error calculating metrics: {e}")
    else:
        print("Warning: Cannot calculate metrics due to empty or mismatched arrays.")

    return {
        'Epsilon': epsilon_value if epsilon_value != float('inf') else 'Non-DP',
        'MSE': mse,
        'MAE': mae,
        'MAPE (%)': mape,
        'MdAPE (%)': mdape,
        'R2': r_square,
        'MAPE_Individuals': mape_individuals.tolist() if isinstance(mape_individuals, np.ndarray) else mape_individuals,
        'y_pred': y_pred_unscaled_np.tolist(),
        'y_true': y_true_unscaled_np.tolist(),
        'inputs': x_test_unscaled_np.tolist(),
        'test_ids': test_county_ids_for_unscaling.tolist()
    }


# --- Main Execution ---
if __name__ == "__main__":
    num_runs = 15
    epsilon_values = [0.3, 0.5, 1.0, 2.0, float('inf')]
    all_run_results_dfs = []
    all_raw_results = []  # To store detailed results including predictions

    # Load County Data
    county_data_raw = load_county_data(path_json)

    if county_data_raw is not None and county_data_raw.shape[0] > 0:
        # Preprocess County Data
        X_processed, Y_processed = preprocess_county_dataset(
            county_data_raw, day_f, day_before)

        if scale_to_relative:
            X_processed, Y_processed = scale_by_population(
                X_processed, Y_processed, COUNTY_POP_FILE
            )

        if X_processed.shape[0] > 0:
            # Split County Data
            X_final, Y_final, Dic_indices_train_county, list_indices_test_county, test_county_ids_for_unscaling = split_county_data(
                X_processed, Y_processed, p_train=p_train
            )

            if X_final.shape[0] > 0 and list_indices_test_county.size > 0:
                county_dataset = RegressionDataset(torch.from_numpy(
                    X_final).float(), torch.from_numpy(Y_final).float())

                NUM_FEATURES = X_final.shape[1]
                training_keys_county = list(Dic_indices_train_county.keys())
                Total_Counties = len(training_keys_county)

                if Total_Counties == 0:
                    print(
                        "Error: No counties available for training after filtering. Exiting.")
                else:
                    print(
                        f"Ready for training. Features: {NUM_FEATURES}, Training Counties: {Total_Counties}, Test Samples: {len(list_indices_test_county)}")

                    for run_idx in range(num_runs):
                        print(
                            f"\n===== Starting County Run {run_idx + 1}/{num_runs} =====")

                        run_results = []  # Results for this specific run

                        for eps in epsilon_values:
                            results = run_county_experiment(
                                epsilon_value=eps,
                                dataset=county_dataset,
                                list_indices_test=list_indices_test_county,
                                test_county_ids_for_unscaling=test_county_ids_for_unscaling,
                                run_idx=run_idx,
                                Dic_indices_train_county=Dic_indices_train_county,
                                keys_train_county=training_keys_county,
                                NUM_FEATURES=NUM_FEATURES,
                                device=device,
                            )
                            # Store full results with predictions
                            results_with_run = results.copy()
                            results_with_run['run'] = run_idx + 1
                            all_raw_results.append(results_with_run)

                            # Prepare results for DataFrame (without list-like objects)
                            metrics_for_df = {k: v for k,
                                              v in results.items() if not isinstance(v, list)}
                            run_results.append(metrics_for_df)

                        if run_results:
                            run_df = pd.DataFrame(run_results)
                            if 'Epsilon' in run_df.columns:
                                run_df['Epsilon'] = run_df['Epsilon'].astype(
                                    str)
                                run_df.set_index('Epsilon', inplace=True)
                                all_run_results_dfs.append(run_df)
                            else:
                                print(
                                    f"Epsilon col missing in results for run {run_idx+1}.")
                        else:
                            print(f"No results generated for run {run_idx+1}.")

                    # --- Aggregate and Display Final Results ---
                    if all_run_results_dfs:
                        combined_df = pd.concat(all_run_results_dfs)
                        numeric_cols = combined_df.select_dtypes(
                            include=np.number).columns
                        aggregated_mean = combined_df.groupby(
                            level='Epsilon')[numeric_cols].mean()
                        aggregated_std = combined_df.groupby(level='Epsilon')[
                            numeric_cols].std()

                        try:
                            eps_order_str = [
                                str(e) for e in epsilon_values if e != float('inf')] + ['Non-DP']
                            aggregated_mean = aggregated_mean.reindex(
                                eps_order_str)
                            aggregated_std = aggregated_std.reindex(
                                eps_order_str)
                        except Exception as e:
                            print(f"Warning: Could not reorder index - {e}")

                        display_df = pd.DataFrame(index=aggregated_mean.index)
                        for col in aggregated_mean.columns:
                            mean_str = aggregated_mean[col].map(
                                '{:.4f}'.format)
                            std_str = aggregated_std[col].map('{:.4f}'.format)
                            display_df[col] = mean_str + ' ± ' + \
                                std_str.fillna('NaN')  # Handle NaN std

                        print(
                            "\n--- Aggregated County Evaluation Metrics (Mean ± Std Dev over {} runs) ---".format(num_runs))
                        display(display_df)
                        print("\n--- LaTeX Format ---")
                        # Ensure float_format handles potential NaNs if needed, though map should produce strings
                        print(display_df.to_latex(
                            index=True, escape=False, na_rep='NaN'))

                        # --- Save Raw Results to CSV ---
                        if all_raw_results:
                            # Convert the list of dictionaries to a DataFrame
                            raw_results_df = pd.DataFrame(all_raw_results)

                            # Define the filename
                            csv_filename = f"year-{year}_county_predictions_scaled-{scale_to_relative}_runs-{num_runs}_rounds-{FED_rounds}.csv"

                            # Save the DataFrame to a CSV file
                            raw_results_df.to_csv(csv_filename, index=False)
                            print(f"\nRaw results saved to {csv_filename}")

                    else:
                        print("\nNo results to aggregate.")
            else:
                print(
                    "Error: Data splitting resulted in no training or test data. Cannot proceed.")
        else:
            print("Error: Preprocessing failed to generate samples. Cannot proceed.")
    else:
        print("Error: Failed to load county data. Cannot proceed.")
