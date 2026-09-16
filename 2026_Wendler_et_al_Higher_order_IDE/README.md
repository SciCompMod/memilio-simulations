# Information

- **Title:**  
  Applied numerical analysis of higher-order schemes for integro-differential equation-based age-of-infection models with time-dependent contact rates
- **Authors:**  
  Anna Wendler, Jan Hasenauer, Martin Kühn

---

# Scripts

In this folder you can find the scripts for replicating the results from the paper. All executables write their output into a `simulation_results` folder (relative to the working directory).

- **`convergence_constant_contacts.cpp`** — Creates convergence results with a constant contact rate. The parameters set in `main` refer to the baseline scenario; adjusting the corresponding parameters in `main` recreates the remaining scenarios. Plot with `plot_convergence.py`.
- **`convergence_nonconstant_contacts.cpp`** — Creates convergence results with a nonconstant contact rate. The parameters set in `main` refer to the scenario with a C^4 polynomial smoother; adjusting the corresponding parameters in `main` recreates the remaining scenarios. Plot with `plot_convergence.py`.
- **`runtime_smoother.cpp`** — Runs a benchmark for the different smoother functions. Plot with `plot_runtime_comparison.py`.
- **`init_conditions_exp.cpp`** — Creates the figures regarding the epidemic outcome for different initialization horizons with exponentially distributed sojourn time distributions. Plot with `plot_init_conditions.py`.
- **`init_conditions_erlang.cpp`** — Creates the figures regarding the epidemic outcome for different initialization horizons with Erlang distributed sojourn time distributions. Plot with `plot_init_conditions.py`.
