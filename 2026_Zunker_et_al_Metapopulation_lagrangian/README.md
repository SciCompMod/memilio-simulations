# Information

- **Title:**  
  Efficient handling of mobility and commuting subpopulations in metapopulation models: Mathematical theory and application to epidemic scenarios
- **DOI:** To be added soon
- **Authors:**  
  Henrik Zunker, René Schmieding, Jan Hasenauer, Martin Kühn

---

# Setup

1. **Build the project with CMake.**  
   The `CMakeLists.txt` will automatically download and build the required version of MEmilio.
   ```sh
   cmake -S . -B build
   cmake --build build --config Release
   ```

2. **Install Python dependencies for plotting:**
   ```sh
   pip install numpy pandas matplotlib
   ```

---

# Plotting

All plot scripts read their input data from a `saves/` directory (relative to the working directory) and write the generated figures into subdirectories of `saves/Figures/`.

---

## Figure 2

Run the following executable before plotting:

- `cpp/benchmarks/flow_simulation_ode_seir.cpp`

---

## Figure 3 — `plot_figure_3.py`

Run the following executables before plotting:

- `cpp/examples/ode_seir_convergence.cpp`
- `cpp/examples/ode_seir_error_compare_stress.cpp`
- `cpp/examples/ode_seir_error_compare_cp.cpp`

---

## Figure 4 — `plot_figure_4.py`

Run the following executables before plotting:

- `cpp/examples/ode_seir_flow_explicit_comp.cpp`
- `cpp/examples/ode_seir_flow_convergence_euler_euler.cpp`

---

## Figure 5 — `plot_figure_5.py`

Run `cpp/benchmarks/ode_seir_benchmark_w_sim.cpp` with the following parameters before plotting:

```cpp
const ScalarType t0    = 0.0;
const ScalarType t_max = 0.5;
const ScalarType dt    = 0.1;

const ScalarType t_max_phi = 0.5;
const ScalarType dt_phi    = 0.1;
```

---

## Figure 6 — `plot_figure_6.py`

Run `cpp/benchmarks/ode_seir_benchmark_w_sim.cpp` with the following parameters before plotting:

```cpp
const ScalarType t_max     = 50.0;
const ScalarType dt        = 0.5;
const ScalarType t_max_phi = 50.0;
const ScalarType dt_phi    = 0.5;
```