import os
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from collections import defaultdict
from RDP_moment_accountant import get_noise_multiplier
from collections import OrderedDict
import random
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import copy
from tqdm.auto import tqdm
import torch
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib
matplotlib.use('Agg')

try:
    from IPython.display import display
except ImportError:
    display = print

# --- Global Parameters ---
EPOCHS = 30
LEARNING_RATE = 0.001
Nbr_selected_LHA = 500
FED_rounds = 200
sensitivity = 1.0
DELTA = 10**-5
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
day_f = 7
day_before = 10
p_train = 0.9
scale_to_relative = False
fill_gaps = True
mavg = 7
SEED = 1838
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if scale_to_relative:
    sensitivity = sensitivity / 1000


# --- Data Loading and Preprocessing ---

extended_add = ""
if fill_gaps:
    extended_add = "_extended"
data_year = 2022  # 2020  #
DATA_BASE_PATH = os.path.join(os.getcwd(), "casedata")
COMMUNITY_CASES = os.path.join(
    DATA_BASE_PATH, f"cases_agg_{data_year}{extended_add}_ma{mavg}.csv")
POP_FILE = os.path.join(DATA_BASE_PATH, "12411-02-03-5.xlsx")
path_csv = COMMUNITY_CASES
path_pop_data = POP_FILE

df = pd.read_csv(path_csv)
df = df.groupby(['Date', 'ID_County', 'ID_Community'],
                as_index=False)['Count'].sum()
df = df.sort_values(by=['ID_County', 'ID_Community'])
a = df.to_numpy()


def preprocess_dataset(a, day_f, day_before):
    X = []
    Y = []
    dates = np.array([datetime.strptime(str(date), "%Y-%m-%d")
                     for date in a[:, 0]])

    # Iterate up to the point where we have enough history and a future target
    for k in range(day_before, len(a) - day_f):

        # --- Generate Historical Features ---
        historical_cases = []
        # Check historical continuity if not filling gaps
        continuous_history = True
        temp_history = []
        for i in range(day_before):
            current_idx = k - i
            prev_idx = current_idx - 1  # k-i-1

            # Check date continuity (backwards) and location consistency
            if (dates[current_idx] - dates[prev_idx]).days != 1 or \
                    a[current_idx, 1] != a[k, 1] or \
                    a[current_idx, 2] != a[k, 2]:
                continuous_history = False
                break
            temp_history.append(float(a[current_idx, 3]))

        if not continuous_history:
            continue  # Skip if history is broken

        # Reverse to get chronological order
        historical_cases = list(reversed(temp_history))

        target_idx = k + day_f
        # Check if target date is exactly day_f days after current date (k)
        # and if the location is the same
        if (dates[target_idx] - dates[k]).days != day_f or \
           a[target_idx, 1] != a[k, 1] or \
           a[target_idx, 2] != a[k, 2]:
            continue

        # --- Create Feature Vector and Label ---
        tab_keep = [
            # ID_County
            int(a[k, 1]),
            # ID_Community
            int(a[k, 2]),
            # Month at time k
            int(dates[k].month),
            # Historical case counts
            *historical_cases
        ]
        X.append(tab_keep)
        Y.append(float(a[target_idx, 3]))

    return np.array(X), np.array(Y)


X, Y = preprocess_dataset(a, day_f, day_before)

# --- Data Splitting and Preparation ---
Dic = defaultdict(list)
for i in range(0, len(X)):
    Dic[str(int(X[i, 0])) + "_" + str(int(X[i, 1]))].append(i)

keys = list(Dic.keys())
Dic_indices_train = {}
list_indices_test = np.array([], dtype=int)
list_ids_test = np.array([], dtype=int)

for key in keys:
    indices = Dic.get(key)
    if len(indices) > 10:
        train = indices[0:int(len(indices)*p_train)]
        test = np.setdiff1d(indices, train)
        Dic_indices_train[key] = copy.deepcopy(train)
        list_indices_test = np.concatenate(
            (list_indices_test, copy.deepcopy(test)))
        list_ids_test = np.concatenate(
            (list_ids_test, np.array([key for _ in test])))
    else:
        pass

list_indices_test = list_indices_test.astype(int)

if scale_to_relative:
    pop_data = pd.read_excel(
        path_pop_data, sheet_name='12411-02-03-5', header=5)
    pop_data = pop_data.iloc[3:13922, [1, 20]]
    pop_data.columns = ['ID', 'Population']
    # transform to dictionary with IDs as key and population as value
    pop_dict = dict(zip(pop_data.iloc[:, 0], pop_data.iloc[:, 1]))

    # iiterate over the dataset and scale the data
    for i in range(len(X)):
        # transform county and community to a string with 3 digits
        county = str(int(X[i, 0]))
        comm = str(int(X[i, 1])).zfill(3)
        if comm == '000':
            comm = ""
        # Hamburg
        if county == '2000':
            county = '2'
        # Berlin
        elif county == '11000':
            county = '11'
        # create the key for the population dictionary
        key = f"{county}{comm}"
        if int(key) not in pop_dict:
            print(
                f"Warning: Key {key} not found in population data. Population will be set to 0.")
        population_key = pop_dict.get(int(key))
        if population_key == 0:
            print(
                f"Warning: Population for key {key} is 0. Scaling will be skipped for this entry.")
            continue
        X[i, 3:] = X[i, 3:] / population_key
        Y[i] = Y[i] / population_key


def unscale_data(y_pred, y_true, x_test):
    """ Scales predictions, true values, and inputs back using population data. """
    path_pop_data = "/localdata1/code_fl/casedata/12411-02-03-5.xlsx"
    pop_data = pd.read_excel(
        path_pop_data, sheet_name='12411-02-03-5', header=5)
    pop_data = pop_data.iloc[3:13922, [1, 20]]
    pop_data.columns = ['ID', 'Population']
    # transform to dictionary with IDs as key and population as value
    pop_dict = dict(zip(pop_data.iloc[:, 0], pop_data.iloc[:, 1]))

    y_pred_unscaled = np.zeros_like(y_pred)
    y_true_unscaled = np.zeros_like(y_true)
    x_test_unscaled = np.zeros_like(x_test, dtype=float)

    for i in range(len(list_indices_test)):
        id = list_ids_test[i]

        # transform id to correct format
        county_id = int(id.split('_')[0])
        community_id = int(id.split('_')[1])
        if community_id == '000' or community_id == 0:
            community_id = ""
        # Hamburg
        if county_id == 2000:
            county_id = 2
        # Berlin
        elif county_id == 11000:
            county_id = 11
        # create the key for the population dictionary
        if community_id != "":
            community_id = str(community_id).zfill(3)
        key = f"{county_id}{community_id}"
        population = pop_dict.get(int(key))
        if population is not None and population > 0:
            y_pred_unscaled[i] = y_pred[i] * population
            y_true_unscaled[i] = y_true[i] * population
            x_test_unscaled[i] = x_test[i] * population
        else:
            print(
                f"Warning: Population for ID {key} not found. Using scaled values.")
            y_pred_unscaled[i] = y_pred[i]
            y_true_unscaled[i] = y_true[i]
            x_test_unscaled[i] = x_test[i]

    return y_pred_unscaled, y_true_unscaled, x_test_unscaled


X = np.delete(X, [0, 1, 2], 1)  # Remove County ID, Community ID and month


class RegressionDataset(Dataset):
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]

    def __len__(self):
        return len(self.X_data)


dataset = RegressionDataset(torch.from_numpy(
    X).float(), torch.from_numpy(Y).float())


# Use keys from the filtered training dict
NUM_FEATURES = X.shape[1]
Total_LHAs = len(list(Dic_indices_train.keys()))
sampling_probability = Nbr_selected_LHA / Total_LHAs if Total_LHAs > 0 else 0

# --- Helper Functions ---


def get_update(global_model, updated_model):
    w_g = copy.deepcopy(global_model.state_dict())
    w_u = copy.deepcopy(updated_model.state_dict())
    update = copy.deepcopy(global_model.state_dict())
    for key in list(w_g.keys()):
        update[key] = torch.subtract(copy.deepcopy(
            w_u).get(key), copy.deepcopy(w_g).get(key))
    return update


def reduce_weights_divide(w, N):
    w_avg = copy.deepcopy(w)
    for key in w_avg.keys():
        if N > 0:
            w_avg[key] = torch.div(w_avg[key], N)
    return w_avg


def update_global_model(w, u):
    w_tmp = copy.deepcopy(w)
    u_tmp = copy.deepcopy(u)
    for key in w_tmp.keys():
        if w_tmp[key].dtype != u_tmp[key].dtype:
            w_tmp[key] = w_tmp[key] + u_tmp[key].to(dtype=w_tmp[key].dtype)
        else:
            w_tmp[key] += u_tmp[key]
    return w_tmp


def flatten_from_weights(weights):
    if not isinstance(weights, dict):
        raise TypeError("Input must be a dictionary of weights.")
    return torch.cat([torch.flatten(weights.get(key).detach()) for key in weights])


def norm_2(model_weights):
    flat = flatten_from_weights(model_weights)
    norm = torch.norm(flat, 2)
    return norm


def clip(update_weights, desired_norm):
    current_norm = norm_2(update_weights)
    if desired_norm <= 0:
        print("Warning: Desired norm for clipping is non-positive. Clipping skipped.")
        return update_weights
    scale = torch.maximum(torch.tensor(1.0).to(device),
                          current_norm / desired_norm)
    clipped_update = copy.deepcopy(update_weights)
    for key in clipped_update:
        clipped_update[key] = torch.divide(clipped_update[key], scale)
    return clipped_update


def run_experiment(epsilon_value, dataset, list_indices_test, Dic_indices_train, keys_train, NUM_FEATURES, device):
    print(
        f"\n--- Running Experiment for Epsilon = {epsilon_value if epsilon_value != float('inf') else 'Non-DP'} ---")

    # --- DP Setup ---
    if epsilon_value == float('inf'):
        DP_flag = 0
        noise_multiplier = 0.0
        print("DP Flag: OFF")
    else:
        DP_flag = 1
        # Ensure sampling_probability is calculated based on available training keys
        num_train_lhas = len(keys_train)
        current_sampling_prob = Nbr_selected_LHA / \
            num_train_lhas if num_train_lhas > 0 else 0

        if current_sampling_prob <= 0 or current_sampling_prob > 1:
            print(
                f"Warning: Invalid sampling probability ({current_sampling_prob:.4f}). Check Nbr_selected_LHA and number of training LHAs.")
            noise_multiplier = float('inf')
        else:
            try:
                noise_multiplier = get_noise_multiplier(
                    target_epsilon=epsilon_value,
                    target_delta=DELTA,
                    sample_rate=current_sampling_prob,
                    steps=FED_rounds
                )
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

    LHA_datasets = {}
    for lha_key in keys_train:
        indices = Dic_indices_train.get(lha_key, [])
        if indices:
            LHA_datasets[lha_key] = Subset(dataset, indices)

    test_dataset = Subset(dataset, list_indices_test)

    update_norms = []
    # --- Training Loop ---
    if not keys_train:  # Skip training if no LHAs
        print("Skipping training: No valid LHAs found.")
    else:
        num_train_lhas = len(keys_train)  # Recalculate here for clarity
        for e in tqdm(range(1, FED_rounds + 1), desc=f"Federated rounds (Epsilon={epsilon_value if epsilon_value != float('inf') else 'Non-DP'})", leave=False):
            # Select LHAs
            selected_LHA = []  # Initialize empty
            if DP_flag == 1 and current_sampling_prob > 0 and noise_multiplier != float('inf'):
                prob_select = min(max(current_sampling_prob, 0.0), 1.0)
                selected_LHA_indices = np.random.choice(
                    a=[False, True], size=num_train_lhas, p=[1 - prob_select, prob_select]
                )
                selected_LHA = np.array(keys_train)[selected_LHA_indices]
            elif not DP_flag and num_train_lhas > 0:
                num_to_select = min(int(Nbr_selected_LHA), num_train_lhas)
                if num_to_select > 0:
                    selected_LHA = np.random.choice(
                        np.array(keys_train), num_to_select, replace=False)

            if len(selected_LHA) == 0:
                continue  # Skip round if no LHAs selected

            local_updates = OrderedDict()

            for i, LHA in enumerate(selected_LHA):
                # Start local model from global
                model_local = copy.deepcopy(ref_model)
                criterion = nn.MSELoss()
                optimizer = optim.Adam(
                    model_local.parameters(), lr=LEARNING_RATE)

                # Use the locally scaled dataset for this LHA if available
                if LHA in LHA_datasets:
                    Sub_train = LHA_datasets[LHA]
                    BATCH_SIZE = max(1, len(Sub_train))
                    train_loader = DataLoader(
                        dataset=Sub_train, batch_size=BATCH_SIZE, shuffle=True)
                else:
                    # Fallback to unscaled data if LHA wasn't scaled
                    indices_LHA_data_train = Dic_indices_train.get(LHA)
                    if not indices_LHA_data_train:
                        continue
                    Sub_train = Subset(dataset, indices_LHA_data_train)
                    BATCH_SIZE = max(1, len(indices_LHA_data_train))
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
                update_norms.append(
                    norm_2(update).item())

                update = clip(update, sensitivity)

                # Aggregate clipped (but not yet noisy) updates (simple sum)
                if not local_updates:
                    local_updates = copy.deepcopy(update)
                else:
                    for key in update:
                        local_updates[key] += update[key]

            # Average aggregated updates (use predefined batch size Nbr_selected_LHA)
            if np.max(update_norms) > sensitivity:
                print(
                    f"Warning: Maximum update norm {np.max(update_norms):.4f} exceeds sensitivity {sensitivity}. Clipping may not be effective.")
            if local_updates:
                averaged_updates = reduce_weights_divide(
                    local_updates, Nbr_selected_LHA)

                # Add noise after averaging if DP is enabled and noise multiplier is valid
                if DP_flag == 1 and noise_multiplier > 0 and noise_multiplier != float('inf'):
                    # Calculate correct noise standard deviation for the *averaged* update.
                    noise_std = (sensitivity * noise_multiplier) / \
                        Nbr_selected_LHA
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
    BATCH_SIZE = max(1, len(Sub_test))
    test_loader = DataLoader(
        dataset=Sub_test, batch_size=BATCH_SIZE, shuffle=False)

    y_pred_list = []
    y_true_list = []
    x_test_list = []
    final_model = ref_model
    final_model.eval()
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            x_test_list.extend(X_batch.cpu().numpy())
            X_batch = X_batch.to(device)
            y_test_pred = final_model(X_batch)
            y_pred_list.extend(y_test_pred.cpu().numpy().squeeze())
            y_true_list.extend(y_batch.cpu().numpy())

    # Convert lists to 1D numpy arrays
    y_pred_np = np.array(y_pred_list)
    y_true_np = np.array(y_true_list)
    x_test_np = np.array(x_test_list)

    # Calculate metrics
    # scale predictions back to unscaled values if required
    if scale_to_relative:
        y_pred_np, y_true_np, x_test_np = unscale_data(
            y_pred_np, y_true_np, x_test_np
        )

    # single mape per test sample
    mape_individuals = []
    mask = y_true_np != 0
    if np.any(mask):
        mape_individuals = np.abs(
            (y_true_np[mask] - y_pred_np[mask]) / y_true_np[mask]) * 100

    mse, mae, mape, mdape, r_square = np.nan, np.nan, np.nan, np.nan, np.nan
    if y_true_np.size > 0 and y_pred_np.size == y_true_np.size:
        try:
            mse = mean_squared_error(y_true_np, y_pred_np)
        except Exception as e:
            print(f"MSE calculation error: {e}")
        try:
            mae = mean_absolute_error(y_true_np, y_pred_np)
        except Exception as e:
            print(f"MAE calculation error: {e}")
        try:
            # Avoid division by zero in MAPE
            mask = y_true_np != 0
            if np.any(mask):
                mape = np.mean(np.abs(
                    (y_true_np[mask] - y_pred_np[mask]) / y_true_np[mask])) * 100
            elif np.allclose(y_true_np, y_pred_np):
                mape = 0.0
            else:
                mape = np.inf
        except Exception as e:
            print(f"MAPE calculation error: {e}")

        # Calculate MdAPE (Median Absolute Percentage Error)
        try:
            mask_mdape = y_true_np != 0
            if np.any(mask_mdape):
                percentage_errors = np.abs(
                    (y_true_np[mask_mdape] - y_pred_np[mask_mdape]) / y_true_np[mask_mdape]) * 100
                mdape = np.median(percentage_errors)
            # All true are zero and all pred are zero
            elif np.allclose(y_true_np, y_pred_np):
                mdape = 0.0
            # All true are zero, but predictions are not (implies infinite error)
            else:
                mdape = np.inf
        except Exception as e:
            print(f"MdAPE calculation error: {e}")
            mdape = np.nan  # Ensure mdape is NaN on error

        try:
            r_square = r2_score(y_true_np, y_pred_np)
        except Exception as e:
            print(f"R2 calculation error: {e}")
    else:
        print("Warning: Mismatch in true/pred shapes or empty arrays during evaluation.")

    return {
        'Epsilon': epsilon_value if epsilon_value != float('inf') else 'Non-DP',
        'MSE': mse,
        'MAE': mae,
        'MAPE (%)': mape,
        'MdAPE (%)': mdape,
        'R2': r_square,
        'MAPE_Individuals': mape_individuals.tolist() if isinstance(mape_individuals, np.ndarray) else mape_individuals,
        'y_pred': y_pred_np.tolist(),
        'y_true': y_true_np.tolist(),
        'inputs': x_test_np.tolist(),
        'test_ids': list_ids_test.tolist()
    }


# --- Main Execution ---
if __name__ == "__main__":
    num_runs = 15  # Number of times to repeat the full experiment
    epsilon_values = [0.3, 0.5, 1.0, 2.0, float('inf')]
    all_run_results_dfs = []
    all_raw_results = []  # To store detailed results including predictions

    # Use the keys from the dictionary that contains actual training data indices
    training_keys = list(Dic_indices_train.keys())
    if not training_keys:
        print("Error: No locations available for training after filtering.")
    else:
        print(f"Found {len(training_keys)} training LHAs.")

        for run_idx in range(num_runs):
            print(f"\n===== Starting Run {run_idx + 1}/{num_runs} =====")

            run_results_current_run = []

            print(f"Starting epsilon experiments for run {run_idx+1}.")
            for eps in epsilon_values:
                results = run_experiment(
                    epsilon_value=eps,
                    dataset=dataset,
                    list_indices_test=list_indices_test,
                    Dic_indices_train=Dic_indices_train,
                    keys_train=training_keys,
                    NUM_FEATURES=NUM_FEATURES,
                    device=device
                )
                # Store full results with predictions
                results_with_run = results.copy()
                results_with_run['run'] = run_idx + 1
                all_raw_results.append(results_with_run)

                # Prepare results for DataFrame (without list-like objects)
                metrics_for_df = {k: v for k,
                                  v in results.items() if not isinstance(v, list)}
                run_results_current_run.append(metrics_for_df)

            if run_results_current_run:
                # Changed from run_results
                run_df = pd.DataFrame(run_results_current_run)
                if 'Epsilon' in run_df.columns:
                    # Convert Epsilon to string to handle float('inf') as index
                    run_df['Epsilon'] = run_df['Epsilon'].astype(str)
                    run_df.set_index('Epsilon', inplace=True)
                    all_run_results_dfs.append(run_df)
                else:
                    print(
                        f"Epsilon col missing in results for run {run_idx+1}.")
            else:
                print(f"No results generated for run {run_idx+1}.")

        # --- Aggregate and Display Final Results ---
        if all_run_results_dfs:
            # Concatenate all run DataFrames
            combined_df = pd.concat(all_run_results_dfs)

            # Calculate mean and std dev, grouping by Epsilon index
            numeric_cols = combined_df.select_dtypes(include=np.number).columns
            aggregated_mean = combined_df.groupby(
                level='Epsilon')[numeric_cols].mean()
            aggregated_std = combined_df.groupby(level='Epsilon')[
                numeric_cols].std()

            # Use 'Non-DP' in the index
            eps_order_str = [
                str(e) for e in epsilon_values if e != float('inf')] + ['Non-DP']
            aggregated_mean = aggregated_mean.reindex(eps_order_str)
            aggregated_std = aggregated_std.reindex(eps_order_str)

            # Combine mean and std into a format "mean ± std"
            display_df = pd.DataFrame(index=aggregated_mean.index)
            for col in aggregated_mean.columns:
                # Handle potential NaN in std (e.g., if num_runs=1)
                mean_str = aggregated_mean[col].map('{:.4f}'.format)
                std_str = aggregated_std[col].map('{:.4f}'.format)
                display_df[col] = mean_str + ' ± ' + std_str.fillna('NaN')

            print(
                "\n--- Aggregated Evaluation Metrics (Mean ± Std Dev over {} runs) ---".format(num_runs))
            display(display_df)

            # print in latex format
            print("\n--- LaTeX Format ---")
            print(display_df.to_latex(index=True,
                  float_format="%.4f", escape=False))

        # --- Save Raw Results to CSV ---
        if all_raw_results:
            # Convert the list of dictionaries to a DataFrame
            raw_results_df = pd.DataFrame(all_raw_results)

            # Define the filename
            csv_filename = f"year-{data_year}_predictions_scaled-{scale_to_relative}_runs-{num_runs}_rounds-{FED_rounds}_mavg-{mavg}.csv"

            # Save the DataFrame to a CSV file
            raw_results_df.to_csv(csv_filename, index=False)
