#!/usr/bin/env python3
"""Summarize daily commuting runtime, scaling or phase data."""
import argparse
import csv
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

PATCHES = {16, 32, 64, 128, 256, 512, 1024}
LARGE_PATCHES = {2048, 4096, 8192, 16384}
GROUPS = {1, 3, 6, 8}
BASE_SHAPES = {(p, g) for p in PATCHES for g in GROUPS}
SHAPE_SETS = {1: BASE_SHAPES, 2: BASE_SHAPES | {(p, 6) for p in LARGE_PATCHES}}
SCALING_THREADS = (1, 2, 4, 8, 16, 32, 64, 128)
SCALING_SHAPE_SETS = {3: (1, 16, 32, 64, 128), 7: SCALING_THREADS}
STRONG_PATCHES = 8192
WEAK_PATCHES = {1: 512, 2: 724, 4: 1024, 8: 1448, 16: 2048, 32: 2896, 64: 4096, 128: 5792}
PHASE_THREADS = (1, 16, 128)
PHASE_EXPERIMENTS = ("phase_strong", "phase_weak")
PHASE_WEAK_THREADS = (16, 32, 64)
PHASE_SHAPE_SETS = {4: (PHASE_EXPERIMENTS, PHASE_THREADS),
                    6: (("phase_weak",), PHASE_WEAK_THREADS)}
PHASE_MODES = {"phases": 4, "phases-weak": 6}
PHASE_SECONDS = ("home_seconds", "away_seconds", "integration_seconds", "departure_seconds",
                 "return_seconds", "profiled_seconds", "unattributed_seconds", "control_seconds")
PHASE_COMPONENTS = ("home", "away", "integration", "departure", "return", "unattributed")
NAME = re.compile(r"^(runtime|strong|weak|phase_strong|phase_weak)/(implicit|explicit)/(serial|openmp|cuda)/"
                  r"patches:(\d+)/age_groups:(\d+)(?:/threads:(\d+))?(?:/iterations:(\d+))?/real_time$")
FACTORS = {"ns": 1e-9, "us": 1e-6, "ms": 1e-3, "s": 1.0}
SETTINGS = ("scenario_version", "shape_set_version", "simulation_days", "step_days", "half_day_steps", "steps",
            "event_interval_days", "home_fraction", "accuracy_tolerance")


def quantile(values, probability):
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower, upper = math.floor(position), math.ceil(position)
    return ordered[lower] + (position - lower) * (ordered[upper] - ordered[lower])


def validate_phases(entry, path):
    required = (*PHASE_SECONDS, "phase_validation_passed", "iterations")
    if any(not isinstance(entry.get(k), (int, float)) or isinstance(entry[k], bool) or
           not math.isfinite(entry[k]) for k in required):
        raise ValueError(f"Missing/non-finite phase metadata: {path}")
    if entry["phase_validation_passed"] != 1 or entry["iterations"] != 1:
        raise ValueError(f"Phase diagnosis requires validation and exactly one trajectory per repetition: {path}")
    if any(entry[k] < 0 for k in PHASE_SECONDS) or min(entry["profiled_seconds"], entry["control_seconds"]) <= 0:
        raise ValueError(f"Invalid phase timing: {path}")
    if not math.isclose(entry["integration_seconds"], entry["home_seconds"] + entry["away_seconds"],
                        rel_tol=1e-8, abs_tol=1e-9):
        raise ValueError(f"Inconsistent integration phase sum: {path}")
    accounted = sum(entry[k] for k in ("integration_seconds", "departure_seconds", "return_seconds",
                                     "unattributed_seconds"))
    if not math.isclose(entry["profiled_seconds"], accounted, rel_tol=1e-8, abs_tol=1e-9):
        raise ValueError(f"Inconsistent profiled phase sum: {path}")
    if not math.isfinite(entry["profiled_seconds"] / entry["control_seconds"]):
        raise ValueError(f"Invalid phase/control timer ratio: {path}")
    if entry["shape_set_version"] == 6:
        rep, control_first = entry.get("repetition_index"), entry.get("control_first")
        if (not isinstance(rep, int) or isinstance(rep, bool) or rep < 0 or
                not isinstance(control_first, (int, float)) or isinstance(control_first, bool) or
                control_first not in (0, 1)):
            raise ValueError(f"Missing/invalid paired control order for phase shape set 6: {path}")
        if control_first != int(rep % 2 == 0):
            raise ValueError(f"Phase shape set 6 requires alternating paired control order by repetition_index: {path}")


def phase_statistics(samples):
    metrics = {key: [r[key] for r in samples] for key in PHASE_SECONDS}
    for phase in PHASE_COMPONENTS:
        metrics[phase + "_fraction"] = [r[phase + "_seconds"] / r["profiled_seconds"] for r in samples]
    metrics["profiling_overhead_fraction"] = [r["profiled_seconds"] / r["control_seconds"] - 1 for r in samples]
    result = dict(diagnostic=1, phase_validation_passed=1, phase_medians_additive=0)
    for key, values in metrics.items():
        q1, median, q3 = (quantile(values, q) for q in (0.25, 0.5, 0.75))
        result.update({"median_" + key: median, "q1_" + key: q1, "q3_" + key: q3, "iqr_" + key: q3 - q1})
    return result


def load(paths):
    files = set()
    for path in paths:
        if path.is_dir():
            files.update(p.resolve() for p in path.rglob("*.json"))
        elif path.is_file() and path.suffix == ".json":
            files.add(path.resolve())
        else:
            raise ValueError(f"Not a result file/directory: {path}")
    if not files:
        raise ValueError("No runtime JSON files found.")
    records = []
    for path in sorted(files):
        with path.open(encoding="utf-8") as handle:
            document = json.load(handle)
        raw = 0
        for entry in document.get("benchmarks", []):
            if entry.get("run_type") == "aggregate" or "aggregate_name" in entry:
                continue
            raw += 1
            if entry.get("error_occurred"):
                raise ValueError(f"{path}: {entry.get('name')}: {entry.get('error_message')}")
            match = NAME.match(entry.get("name", ""))
            if not match:
                raise ValueError(f"Not a daily runtime/scaling/phase benchmark (do not mix kernel results): {path}")
            experiment, model, backend, named_p, named_g, named_threads, named_iterations = match.groups()
            phases = experiment in PHASE_EXPERIMENTS
            if (phases and named_iterations != "1") or (not phases and named_iterations is not None):
                raise ValueError(f"Unexpected named iteration count for {experiment}: {path}")
            entry.setdefault("shape_set_version", 1)
            required = (*SETTINGS, "patches", "age_groups", "cpu_threads", "mobility_events", "events_timed",
                        "validation_passed", "refinement_error", "refinement_validation_patches", "real_time")
            if any(not isinstance(entry.get(k), (int, float)) or not math.isfinite(entry[k]) for k in required):
                raise ValueError(f"Missing/non-finite scenario metadata: {path}: {entry.get('name')}")
            if entry["scenario_version"] != 1 or entry["validation_passed"] != 1:
                raise ValueError(f"Unsupported/unvalidated runtime scenario: {path}")
            allowed_versions = SHAPE_SETS if experiment == "runtime" else (PHASE_SHAPE_SETS if phases else SCALING_SHAPE_SETS)
            if entry["shape_set_version"] not in allowed_versions:
                raise ValueError(f"Unsupported {experiment} shape set: {path}")
            if entry["refinement_error"] < 0 or entry["refinement_error"] > entry["accuracy_tolerance"]:
                raise ValueError(f"Step refinement failed: {path}")
            if entry["accuracy_tolerance"] != 1e-6:
                raise ValueError(f"Unexpected accuracy tolerance: {path}")
            if entry["refinement_validation_patches"] != 8:
                raise ValueError(f"Unexpected refinement validation setup: {path}")
            days, half = entry["simulation_days"], entry["half_day_steps"]
            if days < 1 or days != int(days) or half < 1 or half != int(half):
                raise ValueError(f"Invalid schedule: {path}")
            if entry["steps"] != 2 * days * half or not math.isclose(entry["step_days"] * half, 0.5, abs_tol=1e-14):
                raise ValueError(f"Inconsistent RK4/event schedule: {path}")
            if entry["event_interval_days"] != 0.5 or entry["home_fraction"] != 0.5:
                raise ValueError(f"Unexpected commuting cycle: {path}")
            explicit = model == "explicit"
            if entry["mobility_events"] != (2 * days if explicit else 0) or entry["events_timed"] != int(explicit):
                raise ValueError(f"Missing/untimed mobility events: {path}")
            p, g = entry["patches"], entry["age_groups"]
            threads = entry["cpu_threads"]
            if (p, g) != (int(named_p), int(named_g)) or (
                    backend == "openmp" and (named_threads is None or int(named_threads) != threads)) or (
                    backend != "openmp" and named_threads is not None):
                raise ValueError(f"Benchmark name and shape/thread metadata disagree: {path}")
            if experiment == "runtime":
                if backend == "openmp" and threads == 1:
                    backend = "serial"
                expected_threads = {"serial": 1, "openmp": 128, "cuda": 0}[backend]
                if threads != expected_threads:
                    raise ValueError(f"Expected {expected_threads} CPU threads for {backend}: {path}")
                if (p, g) not in SHAPE_SETS[entry["shape_set_version"]]:
                    raise ValueError(f"Unexpected runtime shape ({p}, {g}): {path}")
            elif phases:
                phase_kinds, phase_threads = PHASE_SHAPE_SETS[entry["shape_set_version"]]
                if model != "explicit" or backend != "openmp" or threads not in phase_threads:
                    raise ValueError(f"Phase diagnosis requires explicit OpenMP with threads in {phase_threads}: {path}")
                if experiment not in phase_kinds:
                    raise ValueError(f"Unsupported {experiment} for phase shape set {entry['shape_set_version']}: {path}")
                expected_p = STRONG_PATCHES if experiment == "phase_strong" else WEAK_PATCHES[threads]
                if (p, g) != (expected_p, 6):
                    raise ValueError(f"Unexpected {experiment} shape ({p}, {g}) for {threads} threads: {path}")
                validate_phases(entry, path)
            else:
                scaling_threads = SCALING_SHAPE_SETS[entry["shape_set_version"]]
                if backend != "openmp" or threads not in scaling_threads:
                    raise ValueError(f"Scaling requires OpenMP with threads in {scaling_threads}: {path}")
                expected_p = STRONG_PATCHES if experiment == "strong" else WEAK_PATCHES[threads]
                if (p, g) != (expected_p, 6):
                    raise ValueError(f"Unexpected {experiment} shape ({p}, {g}) for {threads} threads: {path}")
            unit = entry.get("time_unit")
            if unit not in FACTORS or entry["real_time"] <= 0:
                raise ValueError(f"Invalid timing: {path}")
            records.append(dict(entry, experiment=experiment, model=model, backend=backend, source=str(path),
                                seconds=entry["real_time"] * FACTORS[unit]))
        if not raw:
            raise ValueError(f"No raw measurements in {path} (empty filter or incomplete job).")
    return records


def summarize(records, minimum=10, suite="all", experiment="runtime"):
    if experiment not in ("runtime", "strong", "weak", "scaling", *PHASE_MODES):
        raise ValueError(f"Unknown experiment: {experiment}")
    if suite not in ("cpu", "gpu", "all") or (experiment != "runtime" and suite == "gpu"):
        raise ValueError("Scaling requires the cpu/all suite; phase diagnosis is also CPU-only.")
    if minimum < 1:
        raise ValueError("Repetition count must be positive.")
    settings = {tuple(r[k] for k in SETTINGS) for r in records}
    if len(settings) != 1:
        raise ValueError("Mixed scenario/shape-set versions, horizons, step sizes or accuracy settings; use separate summaries.")
    phases = experiment in PHASE_MODES
    if phases:
        if records[0]["shape_set_version"] != PHASE_MODES[experiment]:
            raise ValueError(f"Wrong shape set for {experiment}; expected version {PHASE_MODES[experiment]}.")
        phase_kinds, phase_threads = PHASE_SHAPE_SETS[PHASE_MODES[experiment]]
        experiments = set(phase_kinds)
    else:
        experiments = {"scaling": {"strong", "weak"}}.get(experiment, {experiment})
    if {r["experiment"] for r in records} != experiments:
        raise ValueError(f"Mixed/incomplete experiments; expected {sorted(experiments)}. Select matching JSON files.")
    grouped = defaultdict(list)
    for r in records:
        grouped[(r["experiment"], r["model"], r["backend"], int(r["patches"]),
                 int(r["age_groups"]), int(r["cpu_threads"]))].append(r)
    if experiment == "runtime":
        expected_backends = {"cpu": {"serial", "openmp"}, "gpu": {"cuda"}, "all": {"serial", "openmp", "cuda"}}[suite]
        shapes = SHAPE_SETS[records[0]["shape_set_version"]]
        expected = {("runtime", m, b, p, g, {"serial": 1, "openmp": 128, "cuda": 0}[b])
                    for m in ("implicit", "explicit") for b in expected_backends for p, g in shapes}
    elif phases:
        expected = {(e, "explicit", "openmp", STRONG_PATCHES if e == "phase_strong" else WEAK_PATCHES[n], 6, n)
                    for e in phase_kinds for n in phase_threads}
    else:
        scaling_threads = SCALING_SHAPE_SETS[records[0]["shape_set_version"]]
        expected = {(e, m, "openmp", STRONG_PATCHES if e == "strong" else WEAK_PATCHES[n], 6, n)
                    for e in experiments for m in ("implicit", "explicit") for n in scaling_threads}
    if set(grouped) != expected:
        raise ValueError(f"Incomplete/wrong suite: missing {sorted(expected - set(grouped))}; unexpected {sorted(set(grouped) - expected)}")
    rows = []
    for (kind, model, backend, p, g, threads), samples in sorted(grouped.items()):
        if len(samples) < minimum:
            raise ValueError(f"Only {len(samples)} repetitions for {(kind, model, backend, p, g, threads)}; expected at least {minimum}.")
        if len({r['source'] for r in samples}) != 1:
            raise ValueError(f"Duplicate measurements from multiple files/jobs for {(kind, model, backend, p, g, threads)}; select one source per case.")
        repetition_indices = [r.get("repetition_index") for r in samples]
        if any(i is not None for i in repetition_indices) and (
                any(not isinstance(i, int) or isinstance(i, bool) or i < 0 for i in repetition_indices) or
                len(set(repetition_indices)) != len(samples)):
            raise ValueError(f"Duplicate/invalid repetition indices for {(kind, model, backend, p, g, threads)}.")
        values = [r["seconds"] for r in samples]
        q1, median, q3 = (quantile(values, q) for q in (0.25, 0.5, 0.75))
        row = dict(model=model, backend=backend, patches=p, age_groups=g, cpu_threads=threads,
                         **{k: samples[0][k] for k in SETTINGS}, mobility_events=samples[0]["mobility_events"],
                         repetitions=len(samples), median_seconds=median, q1_seconds=q1, q3_seconds=q3,
                         iqr_seconds=q3-q1, max_refinement_error=max(r["refinement_error"] for r in samples))
        if experiment != "runtime":
            row["experiment"] = kind
        if phases:
            row.update(phase_statistics(samples))
        rows.append(row)
    if experiment == "runtime":
        medians = {(r["model"], r["backend"], r["patches"], r["age_groups"]): r["median_seconds"] for r in rows}
        for r in rows:
            serial = medians.get((r["model"], "serial", r["patches"], r["age_groups"]))
            implicit = medians[("implicit", r["backend"], r["patches"], r["age_groups"])]
            r["speedup_vs_own_serial"] = "" if serial is None else serial / r["median_seconds"]
            r["implicit_time_over_this_time"] = implicit / r["median_seconds"]
    elif not phases:
        baselines = {(r["experiment"], r["model"]): r["median_seconds"] for r in rows if r["cpu_threads"] == 1}
        for r in rows:
            baseline = baselines[(r["experiment"], r["model"])]
            ratio = baseline / r["median_seconds"]
            if r["experiment"] == "strong":
                r["speedup_vs_openmp_1"] = ratio
                r["parallel_efficiency"] = ratio / r["cpu_threads"]
            else:
                r["weak_scaling_efficiency"] = ratio
                r["normalized_time"] = r["median_seconds"] / baseline
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--expect-min-repetitions", type=int, default=10)
    parser.add_argument("--require-suite", choices=("cpu", "gpu", "all"), default="all")
    parser.add_argument("--experiment", choices=("runtime", "strong", "weak", "scaling", *PHASE_MODES), default="runtime")
    args = parser.parse_args()
    if args.expect_min_repetitions < 1:
        raise ValueError("Repetition count must be positive.")
    rows = summarize(load(args.inputs), args.expect_min_repetitions, args.require_suite, args.experiment)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    experiments = ("strong", "weak") if args.experiment == "scaling" else (args.experiment,)
    for experiment in experiments:
        selected = rows if experiment == "runtime" or experiment in PHASE_MODES else [r for r in rows if r["experiment"] == experiment]
        filename = ("phase_breakdown.csv" if experiment in PHASE_MODES else
                    "runtime_comparison.csv" if experiment == "runtime" else f"{experiment}_scaling.csv")
        output = args.output_dir / filename
        with output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(selected[0]))
            writer.writeheader()
            writer.writerows(selected)
        print(f"Validated {len(selected)} daily {experiment} cases; wrote {output}")
        if experiment in PHASE_MODES:
            print("Diagnostic only: phase medians are not additive; fractions/overhead use paired raw samples.")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        sys.exit(1)
