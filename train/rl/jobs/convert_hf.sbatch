#!/bin/bash
#SBATCH --job-name=mrt-convert-hf
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
# convert_hf.sbatch — FALLBACK Megatron torch_dist -> HF conversion via
# miles/tools/convert_to_hf.py. Primary path is --save-hf during training
# (runs/<arm>/hf/iter_<N>/); use this only if those dirs are empty/corrupt.
#
# Tool argparse (tools/convert_to_hf.py): miles parse_args (full trainer arg
# set: MODEL_ARGS, --hf-checkpoint, --load, ...) + --output-dir, --check-same.
# It loads the LATEST iteration recorded in
# <CKPT_DIR>/latest_checkpointed_iteration.txt; to convert an earlier step,
# temporarily edit that tracker file.
#
# Submit:
#   sbatch --export=ALL,CKPT_DIR=$RUNS/arm_a/ckpts,OUT_DIR=$RUNS/arm_a/hf_fallback \
#       convert_hf.sbatch

set -euo pipefail

SANDBOX=/lustre/fs1/portfolios/nvr/projects/nvr_lacr_llm/users/yuxiaoq/workspace/AI-Reproduction/sandboxes/paper_002_mrt
CODE=${SANDBOX}/scratch_impl/code
MILES_DIR=${CODE}/miles
SELF=${SANDBOX}/scratch_impl/jobs/convert_hf.sbatch
CONTAINER=${CONTAINER:-/lustre/fsw/portfolios/nvr/projects/nvr_lacr_llm/users/yuxiaoq/lustre/images/miles_mirror.sqsh}
BASE_HF_MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B

: "${CKPT_DIR:?submit with --export=ALL,CKPT_DIR=<megatron ckpt dir>,OUT_DIR=<hf output dir>}"
: "${OUT_DIR:?submit with --export=ALL,CKPT_DIR=<megatron ckpt dir>,OUT_DIR=<hf output dir>}"

if [[ "${IN_CONTAINER:-0}" == "1" ]]; then
    # ---------------- runs inside the container ----------------
    source /lustre/fsw/portfolios/nvr/projects/nvr_lacr_llm/users/yuxiaoq/workspace/Research-skills/.env
    export HF_HOME=/lustre/fs1/portfolios/nvr/projects/nvr_lacr_llm/users/yuxiaoq/tmp/huggingface
    # VERIFY-S4: /root/Megatron-LM must exist inside miles_mirror.sqsh
    export PYTHONPATH=/root/Megatron-LM:${MILES_DIR}:${CODE}
    export PYTHONUNBUFFERED=1

    cd "${MILES_DIR}"

    # tools/convert_to_hf.py is dead at miles@9437366e (imports the removed
    # update_weight_utils — job 4767859). Use the maintained CPU reader of
    # torch_dist checkpoints instead; it also strips vocab padding itself.
    ITER=$(cat "${CKPT_DIR}/latest_checkpointed_iteration.txt")
    ITER_DIR=$(printf "%s/iter_%07d" "${CKPT_DIR}" "${ITER}")
    SNAP=$(ls -d ${HF_HOME}/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B/snapshots/*/ | head -1)
    echo "converting ${ITER_DIR} (origin ${SNAP})"

    python3 tools/convert_torch_dist_to_hf.py \
        --input-dir "${ITER_DIR}" \
        --output-dir "${OUT_DIR}" \
        --origin-hf-dir "${SNAP}" \
        --vocab-size 151936 \
        --force

    echo "---- converted HF dir ----"
    ls -la "${OUT_DIR}"
    exit 0
fi

# ---------------- sbatch prologue (host side) ----------------
if [[ ! -f "${CONTAINER}" ]]; then
    echo "ERROR: container image not found: ${CONTAINER}" >&2
    exit 1
fi
if [[ ! -f "${CKPT_DIR}/latest_checkpointed_iteration.txt" ]]; then
    echo "ERROR: no latest_checkpointed_iteration.txt in CKPT_DIR=${CKPT_DIR}" >&2
    exit 1
fi
mkdir -p "${SANDBOX}/scratch_impl/logs"

echo "===================================================================="
echo "Job:   ${SLURM_JOB_NAME} (${SLURM_JOB_ID}) on $(hostname)"
echo "Task:  Megatron -> HF conversion"
echo "From:  ${CKPT_DIR} (iter $(cat ${CKPT_DIR}/latest_checkpointed_iteration.txt))"
echo "To:    ${OUT_DIR}"
echo "===================================================================="

srun \
    --container-image="${CONTAINER}" \
    --container-mounts=/lustre:/lustre \
    --no-container-mount-home \
    --no-container-entrypoint \
    --export=ALL,IN_CONTAINER=1,CKPT_DIR="${CKPT_DIR}",OUT_DIR="${OUT_DIR}" \
    --nodes=1 --ntasks=1 \
    bash "${SELF}"

echo "Conversion job ${SLURM_JOB_ID} finished."
