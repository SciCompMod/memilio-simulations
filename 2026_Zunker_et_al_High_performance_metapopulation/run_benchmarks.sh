#!/usr/bin/env bash
# Runs the experiments of the paper and writes Google Benchmark JSON files.
#
# Usage: ./run_benchmarks.sh <runtime|scaling|phases|phases-weak|mobility|world> [cpu|gpu]
#
#   runtime      daily commuting runtimes; cpu: 1 and 128 OpenMP threads, gpu: CUDA
#   scaling      strong and weak scaling with 1-128 OpenMP threads (cpu only)
#   phases       phase timings of explicit mobility with 1, 16, 128 threads (cpu only)
#   phases-weak  phase timings of explicit mobility, weak sizes for 16, 32, 64 threads (cpu only)
#   mobility     epidemic outcomes of implicit and explicit mobility
#   world        worldwide application; cpu: serial and 128 threads, gpu: CUDA
#
# Optional environment variables (defaults in brackets):
#   BUILD_DIR [./build], OUT_DIR [./results], REPETITIONS [10], MIN_TIME [0.1],
#   STEPS ["1 32"]: RK4 steps per half-day, i.e., h = 1/2 day and h = 1/64 day,
#   NUMA_BINDING [1]: set to 0 to run without numactl (see numa_nodes below),
#   WORLD_INPUT [./world/data/world_inputs.bin], WORLD_DAYS [150]: input and horizon of the world application.
#
# Results are written to OUT_DIR/<experiment>_<cpu|gpu>/hsteps_<steps>/ and OUT_DIR/world_<cpu|gpu>/.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
experiment="${1:?Usage: $0 <runtime|scaling|phases|phases-weak|mobility|world> [cpu|gpu]}"
suite="${2:-cpu}"
BUILD_DIR="${BUILD_DIR:-${here}/build}"
OUT_DIR="${OUT_DIR:-${here}/results}"
REPETITIONS="${REPETITIONS:-10}"
MIN_TIME="${MIN_TIME:-0.1}"
STEPS="${STEPS:-1 32}"
NUMA_BINDING="${NUMA_BINDING:-1}"
bin="${BUILD_DIR}/bin"

if [[ "${experiment}" == "mobility" ]]; then
    "${bin}/ode_seir_mobility_outcomes" "${OUT_DIR}/mobility_outcomes"
    exit 0
fi
case "${experiment}:${suite}" in
    runtime:cpu | runtime:gpu | scaling:cpu | phases:cpu | phases-weak:cpu | world:cpu | world:gpu) ;;
    *) echo "Unsupported combination: ${experiment} ${suite}" >&2; exit 2 ;;
esac

export MEMILIO_RUNTIME_SCENARIO=1
export RUNTIME_EXPERIMENT="${experiment}"
export RUNTIME_DAYS=32
export OMP_DYNAMIC=false OMP_PLACES=cores OMP_PROC_BIND=close

# NUMA nodes used for a given thread count. The paper used two AMD EPYC 9555
# with eight NUMA nodes of 16 cores each.
numa_nodes()
{
    case "$1" in
        1 | 2 | 4 | 8 | 16) echo 0 ;;
        32) echo 0-1 ;;
        64) echo 0-3 ;;
        128) echo 0-7 ;;
    esac
}

# run <executable> <benchmark filter> <output file> [OpenMP threads]
run()
{
    local executable="$1" filter="$2" output="$3" threads="${4:-}"
    local command=("${bin}/${executable}" "--benchmark_filter=${filter}"
        "--benchmark_repetitions=${REPETITIONS}" "--benchmark_min_time=${MIN_TIME}"
        --benchmark_report_aggregates_only=false --benchmark_out_format=json "--benchmark_out=${output}")
    if [[ -n "${threads}" ]]; then
        export OMP_NUM_THREADS="${threads}"
        if [[ "${NUMA_BINDING}" != 0 ]] && command -v numactl > /dev/null; then
            # Explicit mobility relies on first-touch placement, implicit mobility interleaves memory.
            local memory="--interleave=$(numa_nodes "${threads}")"
            if [[ "${executable}" == "ode_seir_benchmark_stage_aligned" ]]; then memory="--localalloc"; fi
            command=(numactl "--cpunodebind=$(numa_nodes "${threads}")" "${memory}" "${command[@]}")
        fi
    fi
    echo "$(date --iso-8601=seconds): ${command[*]}"
    "${command[@]}"
}

# Worldwide application with h = 1/64 day. Each program writes the daily infectious residents
# per patch to WORLD_OUTPUT_DIR and prints its integration time.
if [[ "${experiment}" == "world" ]]; then
    export WORLD_INPUT="${WORLD_INPUT:-${here}/world/data/world_inputs.bin}"
    export WORLD_DAYS="${WORLD_DAYS:-150}" RUNTIME_HALF_DAY_STEPS=32
    export WORLD_OUTPUT_DIR="${OUT_DIR}/world_${suite}"
    if [[ ! -f "${WORLD_INPUT}" ]]; then
        echo "World input not found: ${WORLD_INPUT}. Run world/preprocess_world.py first." >&2
        exit 2
    fi
    if [[ -e "${WORLD_OUTPUT_DIR}" ]]; then
        echo "Refusing to overwrite ${WORLD_OUTPUT_DIR}." >&2
        exit 2
    fi
    mkdir -p "${WORLD_OUTPUT_DIR}"
    {
        date --iso-8601=seconds
        hostname
        lscpu
        if [[ "${suite}" == "gpu" ]]; then nvidia-smi -q; fi
    } > "${WORLD_OUTPUT_DIR}/environment.txt" 2>&1 || true

    if [[ "${suite}" == "gpu" ]]; then
        "${bin}/ode_seir_metapop_benchmark_gpu" | tee "${WORLD_OUTPUT_DIR}/implicit.log"
        "${bin}/ode_seir_benchmark_gpu" | tee "${WORLD_OUTPUT_DIR}/explicit.log"
    else
        # Without WORLD_THREADS, the CPU programs use their serial implementations.
        for run in explicit_openmp implicit_openmp explicit_serial implicit_serial; do
            executable=ode_seir_metapop_benchmark_cpu
            if [[ "${run}" == explicit_* ]]; then executable=ode_seir_benchmark_stage_aligned; fi
            threads=1
            unset WORLD_THREADS
            if [[ "${run}" == *_openmp ]]; then threads=128; export WORLD_THREADS=128; fi
            command=("${bin}/${executable}")
            if [[ "${NUMA_BINDING}" != 0 ]] && command -v numactl > /dev/null; then
                memory="--interleave=$(numa_nodes "${threads}")"
                if [[ "${executable}" == "ode_seir_benchmark_stage_aligned" ]]; then memory="--localalloc"; fi
                command=(numactl "--cpunodebind=$(numa_nodes "${threads}")" "${memory}" "${command[@]}")
            fi
            echo "$(date --iso-8601=seconds): ${command[*]}"
            "${command[@]}" | tee "${WORLD_OUTPUT_DIR}/${run}.log"
        done
    fi
    exit 0
fi

for steps in ${STEPS}; do
    export RUNTIME_HALF_DAY_STEPS="${steps}"
    data="${OUT_DIR}/${experiment}_${suite}/hsteps_${steps}"
    if [[ -e "${data}" ]]; then
        echo "Refusing to overwrite ${data}." >&2
        exit 2
    fi
    mkdir -p "${data}"

    # Record the machine.
    {
        date --iso-8601=seconds
        hostname
        lscpu
        if [[ "${suite}" == "gpu" ]]; then nvidia-smi -q; fi
    } > "${data}/environment.txt" 2>&1 || true

    case "${experiment}" in
        runtime)
            if [[ "${suite}" == "cpu" ]]; then
                # The serial baseline is the OpenMP implementation on one thread.
                for backend in serial openmp; do
                    threads=1
                    if [[ "${backend}" == "openmp" ]]; then threads=128; fi
                    run ode_seir_metapop_benchmark_cpu "^runtime/implicit/openmp/" \
                        "${data}/runtime_implicit_${backend}.json" "${threads}"
                    run ode_seir_benchmark_stage_aligned "^runtime/explicit/openmp/" \
                        "${data}/runtime_explicit_${backend}.json" "${threads}"
                done
            else
                run ode_seir_metapop_benchmark_gpu "^runtime/implicit/cuda/" "${data}/runtime_implicit_cuda.json"
                run ode_seir_benchmark_gpu "^runtime/explicit/cuda/" "${data}/runtime_explicit_cuda.json"
            fi
            ;;
        scaling)
            for threads in 1 2 4 8 16 32 64 128; do
                run ode_seir_metapop_benchmark_cpu "^(strong|weak)/implicit/openmp/" \
                    "${data}/scaling_implicit_openmp_${threads}.json" "${threads}"
                run ode_seir_benchmark_stage_aligned "^(strong|weak)/explicit/openmp/" \
                    "${data}/scaling_explicit_openmp_${threads}.json" "${threads}"
            done
            ;;
        phases | phases-weak)
            thread_counts=(1 16 128)
            filter="^phase_(strong|weak)/explicit/openmp/"
            if [[ "${experiment}" == "phases-weak" ]]; then
                thread_counts=(16 32 64)
                filter="^phase_weak/explicit/openmp/"
            fi
            for threads in "${thread_counts[@]}"; do
                run ode_seir_benchmark_stage_aligned "${filter}" \
                    "${data}/phases_explicit_openmp_${threads}.json" "${threads}"
            done
            ;;
    esac

    # Validate the complete run (schedule, accuracy checks, case grid, repetitions).
    python3 "${here}/plots/summarize_runtime_comparison.py" "${data}" --output-dir "${data}/summary" \
        --expect-min-repetitions "${REPETITIONS}" --require-suite "${suite}" --experiment "${experiment}"
done
