#!/bin/bash
#SBATCH --job-name=mrt-prep-ckpt
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
# prep_megatron_ckpt.sbatch — one-off: convert DeepSeek-R1-Distill-Qwen-1.5B
# (HF) to a Megatron torch_dist checkpoint at runs/r1d15b_torch_dist.
# Invocation pattern from miles docs/getting-started/quick-start.md:50
# (single process is fine for 1.5B; bigger models use torchrun).
#
# Submit:  sbatch prep_megatron_ckpt.sbatch
# Success: runs/r1d15b_torch_dist/latest_checkpointed_iteration.txt contains
#          "release" (written last by the tool, after dist.barrier).
#
# Self-exec trick: the sbatch body re-runs THIS file (by absolute path, NOT
# $0 — Slurm's spool copy is invisible inside the container) under srun with
# IN_CONTAINER=1.

set -euo pipefail

SANDBOX=/lustre/fs1/portfolios/nvr/projects/nvr_lacr_llm/users/yuxiaoq/workspace/AI-Reproduction/sandboxes/paper_002_mrt
CODE=${SANDBOX}/scratch_impl/code
MILES_DIR=${CODE}/miles
RUNS=${SANDBOX}/scratch_impl/runs
SELF=${SANDBOX}/scratch_impl/jobs/prep_megatron_ckpt.sbatch
CONTAINER=${CONTAINER:-/lustre/fsw/portfolios/nvr/projects/nvr_lacr_llm/users/yuxiaoq/lustre/images/miles_mirror.sqsh}
BASE_HF_MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
SAVE_DIR=${RUNS}/r1d15b_torch_dist

if [[ "${IN_CONTAINER:-0}" == "1" ]]; then
    # ---------------- runs inside the container ----------------
    source /lustre/fsw/portfolios/nvr/projects/nvr_lacr_llm/users/yuxiaoq/workspace/Research-skills/.env
    export HF_HOME=/lustre/fs1/portfolios/nvr/projects/nvr_lacr_llm/users/yuxiaoq/tmp/huggingface
    # VERIFY-S4: /root/Megatron-LM must exist inside miles_mirror.sqsh
    export PYTHONPATH=/root/Megatron-LM:${MILES_DIR}:${CODE}
    export PYTHONUNBUFFERED=1

    cd "${MILES_DIR}"
    # MODEL_ARGS already pins --vocab-size 151936 (the padded vocab), so the
    # tool's optional --padded-vocab-size is NOT needed.
    source scripts/models/qwen2.5-1.5B.sh
    # R1-Distill has tie_word_embeddings=false (unlike base Qwen2.5-1.5B):
    MODEL_ARGS+=(--untie-embeddings-and-output-weights)

    mkdir -p "${SAVE_DIR}"

    # Tool argparse (tools/convert_hf_to_torch_dist.py): megatron parse_args
    # (accepts MODEL_ARGS and --save) + --hf-checkpoint (required),
    # --megatron-to-hf-mode, --padded-vocab-size. Single process => WORLD_SIZE
    # falls back to SLURM_NTASKS=1, pp auto-resolves to 1.
    python3 tools/convert_hf_to_torch_dist.py \
        ${MODEL_ARGS[@]} \
        --hf-checkpoint "${BASE_HF_MODEL}" \
        --save "${SAVE_DIR}"

    echo "---- tracker file ----"
    cat "${SAVE_DIR}/latest_checkpointed_iteration.txt"
    exit 0
fi

# ---------------- sbatch prologue (host side) ----------------
if [[ ! -f "${CONTAINER}" ]]; then
    echo "ERROR: container image not found: ${CONTAINER}" >&2
    exit 1
fi
mkdir -p "${SANDBOX}/scratch_impl/logs"

echo "===================================================================="
echo "Job:    ${SLURM_JOB_NAME} (${SLURM_JOB_ID}) on $(hostname)"
echo "Task:   HF -> Megatron torch_dist conversion of ${BASE_HF_MODEL}"
echo "Save:   ${SAVE_DIR}"
echo "===================================================================="

srun \
    --container-image="${CONTAINER}" \
    --container-mounts=/lustre:/lustre \
    --no-container-mount-home \
    --no-container-entrypoint \
    --export=ALL,IN_CONTAINER=1 \
    --nodes=1 --ntasks=1 \
    bash "${SELF}"

echo "Conversion job ${SLURM_JOB_ID} finished."
