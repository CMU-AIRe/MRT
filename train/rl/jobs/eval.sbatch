#!/bin/bash
#SBATCH --job-name=mrt-eval
#SBATCH --account=nvr_lacr_llm
#SBATCH --partition=batch_block1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=64
#SBATCH --exclusive
#SBATCH --time=03:55:00
#SBATCH --output=/lustre/fs1/portfolios/nvr/projects/nvr_lacr_llm/users/yuxiaoq/workspace/AI-Reproduction/sandboxes/paper_002_mrt/scratch_impl/logs/%x_%j.out
#
# eval.sbatch — vLLM evaluation of an HF checkpoint on the 5 MRT benchmarks
# (AIME24/AIME25/AMC23/Minerva/MATH500), pass@1 over N_SAMPLES, decoding per
# assumption A3 (temp 0.6 / top_p 0.95, handled inside vllm_eval.py).
# Runs in the HOST conda env (no container) — vLLM lives in
# repro_paper_002_scratch_impl.
#
# VERIFY-S4: code/eval/vllm_eval.py does not exist yet; this job is written
# against the CLI contract in implementation_plan.md:
#   vllm_eval.py --model-path --benchmarks-dir --output-dir --benchmarks --n-samples
# Shards are resumable per (problem_id, sample_idx), so requeueing/chaining
# this job on the same OUT_DIR continues where it left off.
#
# Submit:
#   sbatch --export=ALL,MODEL_PATH=$RUNS/arm_a/hf/iter_62 eval.sbatch
#   sbatch --export=ALL,MODEL_PATH=...,BENCHMARKS="aime24 math500",N_SAMPLES=4 eval.sbatch   # probe

set -euo pipefail

SANDBOX=/lustre/fs1/portfolios/nvr/projects/nvr_lacr_llm/users/yuxiaoq/workspace/AI-Reproduction/sandboxes/paper_002_mrt
CODE=${SANDBOX}/scratch_impl/code
RUNS=${SANDBOX}/scratch_impl/runs

: "${MODEL_PATH:?submit with --export=ALL,MODEL_PATH=<hf model dir or HF id>}"
BENCHMARKS=${BENCHMARKS:-"aime24,aime25,amc23,minerva,math500"}
# Slurm --export splits values on commas — accept colon/space separators too
# (submit with BENCHMARKS=aime25:amc23:math500) and normalize to commas here.
BENCHMARKS="${BENCHMARKS//[: ]/,}"
N_SAMPLES=${N_SAMPLES:-64}
NUM_SHARDS=${NUM_SHARDS:-8}   # problem-level data parallelism, one process per GPU
OUT_DIR=${OUT_DIR:-${RUNS}/eval/$(basename "${MODEL_PATH}")}

source /lustre/fsw/portfolios/nvr/projects/nvr_lacr_llm/users/yuxiaoq/envs/miniconda3/bin/activate repro_paper_002_scratch_impl
source /lustre/fsw/portfolios/nvr/projects/nvr_lacr_llm/users/yuxiaoq/workspace/Research-skills/.env
export HF_HOME=/lustre/fs1/portfolios/nvr/projects/nvr_lacr_llm/users/yuxiaoq/tmp/huggingface
export PYTHONUNBUFFERED=1

mkdir -p "${OUT_DIR}" "${SANDBOX}/scratch_impl/logs"

echo "===================================================================="
echo "Job:        ${SLURM_JOB_NAME} (${SLURM_JOB_ID}) on $(hostname)"
echo "Model:      ${MODEL_PATH}"
echo "Benchmarks: ${BENCHMARKS} (n=${N_SAMPLES})"
echo "Output:     ${OUT_DIR}"
echo "Env:        $(which python)"
echo "===================================================================="

# Trim Megatron vocab padding from local HF exports (idempotent; HF-hub ids skipped).
if [[ -d "${MODEL_PATH}" ]]; then
    python "${CODE}/eval/trim_vocab_padding.py" "${MODEL_PATH}"
fi

# One vLLM process per GPU, problems sharded round-robin; all shards resumable.
PIDS=()
for SHARD in $(seq 0 $((NUM_SHARDS - 1))); do
    CUDA_VISIBLE_DEVICES=${SHARD} python "${CODE}/eval/vllm_eval.py" \
        --model-path "${MODEL_PATH}" \
        --benchmarks-dir "${SANDBOX}/scratch_impl/data/benchmarks" \
        --output-dir "${OUT_DIR}" \
        --benchmarks "${BENCHMARKS}" \
        --n-samples "${N_SAMPLES}" \
        --num-shards "${NUM_SHARDS}" --shard-id "${SHARD}" \
        > "${OUT_DIR}/shard${SHARD}.log" 2>&1 &
    PIDS+=($!)
done

FAIL=0
for pid in "${PIDS[@]}"; do
    wait "${pid}" || FAIL=1
done
if [[ ${FAIL} -ne 0 ]]; then
    echo "WARNING: at least one shard failed — job is resumable, resubmit to continue" >&2
fi

# Merge shards into per-benchmark summary.json + final table (CPU only).
python "${CODE}/eval/vllm_eval.py" \
    --model-path "${MODEL_PATH}" \
    --benchmarks-dir "${SANDBOX}/scratch_impl/data/benchmarks" \
    --output-dir "${OUT_DIR}" \
    --benchmarks "${BENCHMARKS}" \
    --n-samples "${N_SAMPLES}" \
    --summarize-only

[[ ${FAIL} -eq 0 ]] || exit 1
echo "Eval job ${SLURM_JOB_ID} finished. Results in ${OUT_DIR}"
