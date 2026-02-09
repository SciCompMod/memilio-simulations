import os
import pickle
import time
import pandas as pd
import numpy as np
import tensorflow as tf
from keras.layers import Dense
from keras.models import Model
from spektral.layers import ARMAConv
import spektral

from memilio.surrogatemodel.GNN.data_generation_nodeswithvariance import (
    run_secir_groups_simulation, get_graph)
from memilio.surrogatemodel.GNN.GNN_utils import transform_mobility_directory
from memilio.simulation import set_log_level, LogLevel

# Reduce log verbosity
set_log_level(LogLevel.Critical)

# GNN architecture (compatible with Schmidt et al.)


class Net(Model):
    def __init__(self, channels=512, n_labels=1440):
        # n_labels: Days * 8 Compartments * 6 Age Groups
        super().__init__()
        self.conv1 = ARMAConv(channels, activation='relu')
        self.conv2 = ARMAConv(channels, activation='relu')
        self.conv3 = ARMAConv(channels, activation='relu')
        self.conv4 = ARMAConv(channels, activation='relu')
        self.conv5 = ARMAConv(channels, activation='relu')
        self.conv6 = ARMAConv(channels, activation='relu')
        self.conv7 = ARMAConv(channels, activation='relu')
        self.dense = Dense(n_labels, activation="linear")

    def call(self, inputs):
        x, a = inputs
        # Ensure adjacency is an array
        a = np.asarray(a)
        x = self.conv1([x, a])
        x = self.conv2([x, a])
        x = self.conv3([x, a])
        x = self.conv4([x, a])
        x = self.conv5([x, a])
        x = self.conv6([x, a])
        x = self.conv7([x, a])
        output = self.dense(x)
        return output


def measure_runtime():
    days_list = [30, 60, 90]
    num_predictions_list = [1, 2, 4, 8, 16, 32, 64, 128]
    iterations_ode = 3
    iterations_gnn = 50
    path_weights_base = '/PATH/saved_models_GNN/'

    # Initialize graph for ODE
    data_dir = os.path.join(os.getcwd(), 'data')
    mobility_dir = transform_mobility_directory()
    graph = get_graph(6, data_dir, mobility_dir)

    ode_measured_times = {}

    print("Starting ODE benchmark...")
    for day in days_list:
        times = []
        for i in range(iterations_ode):
            _, t_run = run_secir_groups_simulation(day, graph)
            times.append(t_run)
        avg_time = np.mean(times)
        ode_measured_times[day] = avg_time
        print(f"  ODE {day} days: {avg_time:.2f}s")

    results = []
    # Create dummy binary adjacency for 400 regions (as in paper)
    adj = np.random.randint(0, 2, (400, 400)).astype(np.float32)
    # Normalize adjacency for ARMAConv
    adj_norm = spektral.utils.convolution.normalized_adjacency(adj)

    for day in days_list:
        print(f"\nBenchmarking GNN for {day} simulation days...")
        n_labels = day * 8 * 6  # 8 compartments, 6 age groups
        model = Net(n_labels=n_labels)

        # Load pretrained weights
        path_weight = os.path.join(
            path_weights_base, f"GNN_{day}days_nodeswithvariance_1k_test.pickle")

        # Input features: 5 lead days * 8 compartments * 6 age groups
        # Shape: (Batch, Nodes, Features)
        input_dim = 5 * 8 * 6

        if os.path.exists(path_weight):
            with open(path_weight, "rb") as fp:
                weights = pickle.load(fp)
            # Dummy call to initialize variables
            dummy_x = np.random.rand(1, 400, input_dim).astype(np.float32)
            model([dummy_x, adj_norm])
            model.set_weights(weights)
            print(f"  Weights loaded: {os.path.basename(path_weight)}")
        else:
            print(f"  WARNING: weights not found at {path_weight}. Using random weights.")
            dummy_x = np.random.rand(1, 400, input_dim).astype(np.float32)
            model([dummy_x, adj_norm])

        for num_pred in num_predictions_list:
            print(f"  Batch Size: {num_pred}...", end="", flush=True)
            x_batch = np.random.rand(
                num_pred, 400, input_dim).astype(np.float32)

            _ = model([x_batch, adj_norm], training=False)

            times = []
            for _ in range(iterations_gnn):
                start = time.perf_counter()
                _ = model([x_batch, adj_norm], training=False)
                # Ensure computation finished (important for GPU); measure wall-clock
                end = time.perf_counter()
                times.append(end - start)

            mean_gnn = np.mean(times)
            median_gnn = np.median(times)

            # Compute ODE time (assume linear scaling)
            mean_ode = ode_measured_times[day] * num_pred

            results.append({
                'Num_Pred': num_pred,
                'Days': day,
                'Mean_Time_GNN': mean_gnn,
                'Median_Time_GNN': median_gnn,
                'Mean_Time_ODE': mean_ode,
                'Speedup': mean_ode / mean_gnn if mean_gnn > 0 else 0
            })
            print(f" Done. (GNN: {mean_gnn:.4f}s, ODE Ref: {mean_ode:.2f}s)")

    df = pd.DataFrame(results)
    output_csv = '/PATH/measurements_gnn_vs_ode.csv'
    df.to_csv(output_csv, index=False)
    print(f"\nResults saved to '{output_csv}'")


if __name__ == "__main__":
    # GPU Speicherverwaltung optimieren (falls vorhanden)
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

    measure_runtime()
