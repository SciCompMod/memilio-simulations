
"""
Benchmark the C++ SEIR metapopulation model via the Python bindings.

The script mirrors ``ode_seir_metapop_benchmark.cpp``: it varies
the number of regions, simulates with fixed step size, records runtime, and
writes a CSV summary. Optionally, the simulated trajectories can be exported.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import statistics
import time
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np

import memilio.simulation as sim
from memilio.simulation import AgeGroup, Region
from memilio.simulation.oseir_metapop import InfectionState as State
from memilio.simulation.oseir_metapop import Model, simulate

# Default grid used by the C++ benchmark.
DEFAULT_REGION_GRID: Sequence[int] = (1, 2, 4, 8, 16, 32, 64, 128, 256)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        "ode_seir_metapop_benchmark",
        description=(
            "Replicate the C++ metapopulation benchmark via Python bindings. "
            "Accepts optional --dt/--tmax overrides and region counts."
        ),
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=0.1,
        help="Fixed time step for the integrator. Matches the C++ default (0.1).",
    )
    parser.add_argument(
        "--tmax",
        type=float,
        default=500.0,
        help="Simulation horizon in days. Matches the C++ default (500.0).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("compare_results/python_benchmark_rk4.csv"),
        help="Destination CSV path for the benchmark summary.",
    )
    parser.add_argument(
        "--timeseries-dir",
        type=Path,
        help=(
            "Optional directory; if set, write time series CSV per run using "
            "the same layout as the C++ export (S/E/I/R ordered per region)."
        ),
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=40,
        help="Number of repeated simulations per configuration; median runtime is recorded.",
    )
    parser.add_argument(
        "regions",
        nargs="*",
        type=int,
        help="List of region counts to benchmark. Defaults to the C++ grid.",
    )
    return parser.parse_args()


def prepare_model(num_regions: int, num_agegroups: int = 1) -> Model:
    model = Model(num_regions, num_agegroups)

    for region_idx in range(num_regions):
        region = Region(region_idx)
        for age_idx in range(num_agegroups):
            age = AgeGroup(age_idx)
            model.populations[region, age, State.Susceptible] = 10000.0
            model.populations[region, age, State.Exposed] = 0.0
            model.populations[region, age, State.Infected] = 0.0
            model.populations[region, age, State.Recovered] = 0.0

    for age_idx in range(num_agegroups):
        age = AgeGroup(age_idx)
        model.parameters.TimeExposed[age] = 3.335
        model.parameters.TimeInfected[age] = 8.097612257
        model.parameters.TransmissionProbabilityOnContact[age] = 0.07333

    # Seed the first region with 100 exposed individuals.
    model.populations[Region(0), AgeGroup(0), State.Susceptible] = 9900.0
    model.populations[Region(0), AgeGroup(0), State.Exposed] = 100.0

    baseline = np.full((num_agegroups, num_agegroups), 2.7, dtype=float)
    zeros = np.zeros_like(baseline)
    model.parameters.ContactPatterns.cont_freq_mat[0].baseline = baseline
    model.parameters.ContactPatterns.cont_freq_mat[0].minimum = zeros

    model.set_commuting_strengths(np.eye(num_regions, dtype=float))

    model.check_constraints()
    return model


def run_benchmark(
    region_grid: Iterable[int],
    dt: float,
    tmax: float,
    summary_path: Path,
    timeseries_dir: Path | None,
    runs: int,
) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    if timeseries_dir is not None:
        timeseries_dir.mkdir(parents=True, exist_ok=True)

    rk4_cls = getattr(sim, "RungeKutta4IntegratorCore", None)
    integrator_factories: List[tuple[str, callable]] = []
    if rk4_cls is not None:
        integrator_factories.append(
            ("RungeKutta4IntegratorCore", lambda: rk4_cls())
        )
    else:
        integrator_factories.append(
            ("RKIntegratorCore", lambda: _fixed_step_rk_integrator(dt))
        )
    integrator_factories.append(("Euler", lambda: sim.EulerIntegratorCore()))

    for name, factory in integrator_factories:
        if name.lower().startswith("euler"):
            out_path = summary_path.with_name("python_benchmark_euler.csv")
        else:
            out_path = summary_path
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with out_path.open("w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Regions", "TimeSteps", "RuntimeSeconds", "TotalNoIOSeconds"])

            for num_regions in region_grid:
                sim_durations: List[float] = []
                total_durations: List[float] = []
                steps = None
                export_result = None

                for run_idx in range(max(1, runs)):
                    model = prepare_model(num_regions)
                    total_start = time.perf_counter()
                    integrator = factory()
                    sim_start = time.perf_counter()
                    result = simulate(0.0, tmax, dt, model, integrator=integrator)
                    sim_end = time.perf_counter()
                    total_end = time.perf_counter()

                    sim_durations.append(sim_end - sim_start)
                    total_durations.append(total_end - total_start)
                    if steps is None:
                        steps = result.get_num_time_points()
                        export_result = result

                runtime_sim = statistics.median(sim_durations)
                runtime_total = statistics.median(total_durations)
                writer.writerow([num_regions, steps, runtime_sim, runtime_total])

                print(
                    f"[{name}] Regions={num_regions} steps={steps} "
                    f"median_sim={runtime_sim:.6f}s median_total_no_io={runtime_total:.6f}s "
                    f"(dt={dt}, tmax={tmax}, runs={max(1, runs)})"
                )

                if timeseries_dir is not None and export_result is not None:
                    labels: List[str] = []
                    for region_idx in range(num_regions):
                        labels.extend(
                            [
                                f"S{region_idx + 1}",
                                f"E{region_idx + 1}",
                                f"I{region_idx + 1}",
                                f"R{region_idx + 1}",
                            ]
                        )
                    timeseries_path = timeseries_dir / (
                        f"python_metapop_regions_{num_regions}_{name.lower()}.csv"
                    )
                    export_result.export_csv(str(timeseries_path), labels, ",", 16)

        print(f"Benchmark CSV written to: {out_path}")


def _fixed_step_rk_integrator(dt: float):
    integrator = sim.RKIntegratorCore()
    integrator.dt_min = dt
    integrator.dt_max = dt
    return integrator


def main() -> None:
    args = parse_args()
    region_grid = args.regions or list(DEFAULT_REGION_GRID)

    sim.set_log_level(sim.LogLevel.Critical)

    run_benchmark(region_grid, args.dt, args.tmax,
                  args.out, args.timeseries_dir, args.runs)


if __name__ == "__main__":
    main()
