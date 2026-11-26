# Information

- **Title:**  
  High-performance metapopulation models for epidemic emergencies: Scalability and performance of implicit and explicit mobility representations on CPUs and GPUs
- **Authors:**  
  Henrik Zunker, Carlotta Gerstein, Martin J. Kühn

---

# Requirements

- CMake >= 3.18 and a C++20 compiler (the paper used GCC 10.4)
- OpenMP
- For the GPU benchmarks: CUDA toolkit with cuBLAS (the paper used CUDA 12.4)
- Optional: `numactl` for the NUMA binding used in the paper
- For plotting: Python 3 with `numpy`, `pandas` and `matplotlib`

MEmilio and Google Benchmark are downloaded automatically. HDF5 is not needed.

---

# Build

```sh
cmake -S . -B build
cmake --build build -j 8
```

For the GPU benchmarks, add `-DHPC_METAPOP_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=90` (H100) or `=80` (A100).
All executables are written to `build/bin/`.

---

# Run and plot

`run_benchmarks.sh` runs an experiment and writes the Google benchmark JSON files to the directory `results/<experiment>_<cpu|gpu>/hsteps_<steps>/`.
The global application writes its results to the folder `results/world_<cpu|gpu>/`.
`hsteps_1` and `hsteps_32` correspond to step sizes of h = 1/2 day and h = 1/64 day.
Each run is validated at the end (schedule, accuracy checks, complete case grid, repetitions).
The scripts for generating figures read from these folders and write the figures to the `figures/` folder. Run all commands from this folder.

| Figure | Run | Plot |
| --- | --- | --- |
| Operation counts | none | `python3 plots/plot_operation_counts.py` |
| Mobility outcomes | `./run_benchmarks.sh mobility` | `python3 plots/plot_mobility_outcomes.py` |
| Worldwide application | see below | `python3 plots/plot_world_application.py` |
| Runtime comparison and age groups | `./run_benchmarks.sh runtime cpu` and `./run_benchmarks.sh runtime gpu` | `python3 plots/plot_runtime_comparison.py` |
| Strong and weak scaling | `./run_benchmarks.sh scaling cpu` | `python3 plots/plot_parallel_scalability.py` |
| Phase timings | `./run_benchmarks.sh phases cpu` and `./run_benchmarks.sh phases-weak cpu` | `python3 plots/plot_explicit_phases.py` |
| GPU roofline | see below | `python3 plots/plot_gpu_roofline.py` |
| CPU roofline | see below | `python3 plots/plot_cpu_roofline.py` |

Notes:

- The CPU experiments are based on a node with 128 physical cores (two AMD EPYC 9555 processors)
  The NUMA configuration is defined in `numa_nodes()` in the file `run_benchmarks.sh`; please adjust this for other machines.
- The serial runtimes are measured using the OpenMP implementations on a single thread.
- The case with the longest runtime requires approximately 52 GB of RAM on the CPU and approximately 50 GB of GPU memory.
- The entire CPU runtime experiment takes approximately 40 hours. Set `STEPS=1` or `STEPS=32` to execute only one step,
  and `REPETITIONS` to change the number of repetitions.
- All scripts for plotting also allow different input and output paths.

## Worldwide application

The worldwide application needs two public data sets:

- Population: GPWv4.11 population count for 2020 on the 1-degree grid in ASCII format
  (CIESIN, NASA SEDAC, https://doi.org/10.7927/H4JW8BX5),
  saved as `world/gpw_v4_population_count_rev11_2020_1_deg.asc`.
- Country borders for the map only: Natural Earth 1:50m Admin 0 countries as GeoJSON
  (e.g., `geojson/ne_50m_admin_0_countries.geojson` from https://github.com/nvkelso/natural-earth-vector),
  saved as `world/ne_50m_admin_0_countries.geojson`.

```sh
python3 world/preprocess_world.py      # writes world/data/world_inputs.bin (1.3 GB) and world/data/world_cells.npz
./run_benchmarks.sh world gpu          # implicit and explicit mobility on one GPU
./run_benchmarks.sh world cpu          # 128 threads and serial, implicit and explicit mobility
python3 plots/plot_world_application.py
```

During pre-processing, all 12,865 cells with at least 1,000 inhabitants are retained, the mobility matrix based on the gravity method is generated,
and 100 infectious individuals are located in the Frankfurt cell. All runs simulate 150 days with h = 1/64 of a day
(`WORLD_DAYS` changes the time horizon). The explicit mobility calculation requires approximately 33 GB of GPU memory.
On the CPU, the runs with 128 threads each take a few minutes, whilst the serial runs each take around three hours.

## Roofline measurements

The roofline figures use hardware performance counters and were measured separately from the runtimes.
The executables contain the required profiling hooks when built with `-DHPC_METAPOP_ROOFLINE=ON`.

- **GPU (Nsight Compute 2025.1.1):** day 2 of the 32-day runtime scenario, six age groups, 1024 and 8192 patches,
  both step sizes, three repetitions each. Each case is profiled with
  ```sh
  MEMILIO_ROOFLINE=1 ROOFLINE_DAY=2 MEMILIO_RUNTIME_SCENARIO=1 RUNTIME_EXPERIMENT=runtime \
  RUNTIME_DAYS=32 RUNTIME_HALF_DAY_STEPS=<1|32> OMP_NUM_THREADS=1 \
  ncu --profile-from-start off --graph-profiling node --kernel-name-base demangled --replay-mode application \
      --cache-control none --clock-control none \
      --section SpeedOfLight_HierarchicalDoubleRooflineChart --section MemoryWorkloadAnalysis --section SpeedOfLight \
      --metrics smsp__sass_thread_inst_executed_op_dadd_pred_on.sum,smsp__sass_thread_inst_executed_op_dmul_pred_on.sum,smsp__sass_thread_inst_executed_op_dfma_pred_on.sum,sm__inst_executed_pipe_tensor.sum,sm__ops_path_tensor_src_fp64.sum,gpu__time_duration.sum,dram__bytes_read.sum,dram__bytes_write.sum \
      --export <case>.ncu-rep \
      build/bin/ode_seir_metapop_benchmark_gpu '--benchmark_filter=^roofline/implicit/cuda/patches:<P>/age_groups:6/iterations:1/real_time$' \
      --benchmark_repetitions=1 --benchmark_out_format=json --benchmark_out=<case>.benchmark.json
  ```
  and analogously for `ode_seir_benchmark_gpu` with `roofline/explicit/cuda/...`.
  Each report is exported to `<case>.ncu.csv` with
  `ncu --import <case>.ncu-rep --page raw --csv --print-units base --print-kernel-base demangled`,
  where `<case>` is `<implicit|explicit>_p<P>_h<1|32>_r<repetition>`.
  `plot_gpu_roofline.py --job <folder>` reads all such files in a folder (`--gpu A100` for the A100 limits).
- **CPU (LIKWID):** two AMD EPYC 7702 with 128 cores; `plot_cpu_roofline.py` reads the exported aggregate tables.
