#!/bin/bash
#SBATCH --job-name=mrt-train
#SBATCH --account=nvr_lacr_llm
#SBATCH --partition=batch_block1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=64
#SBATCH --exclusive
#SBATCH --time=03:55:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/lustre/fs1/portfolios/nvr/projects/nvr_lacr_llm/users/yuxiaoq/workspace/AI-Reproduction/sandboxes/paper_002_mrt/scratch_impl/logs/%x_%j.out
#
# submit_train.sbatch — one <=4h training segment for paper_002_mrt on
# oci-nrt-cs-001 (batch_block1 has a HARD 4h walltime; 03:55:00 leaves margin
# for teardown). Adapted from the proven dfw wrapper
# AI-Scientist/train/RL/dfw/aicsi-rubric-v01.00.sh.
#
# Required (via sbatch --export, see chain.sh):
#   ARM_CONFIG  absolute path to mrt_plugin/configs/arm_*.sh
# Optional:
#   CONTAINER   override container image
#   JOB_NAME    set via `sbatch --job-name` (header default: mrt-train)
#
# Segments share the per-arm --load/--save dir, so each segment auto-resumes
# from the last checkpoint (--save-interval 2 => a TIMEOUT loses <=1 step).

set -euo pipefail

SANDBOX=/lustre/fs1/portfolios/nvr/projects/nvr_lacr_llm/users/yuxiaoq/workspace/AI-Reproduction/sandboxes/paper_002_mrt
CONTAINER=${CONTAINER:-/lustre/fsw/portfolios/nvr/projects/nvr_lacr_llm/users/yuxiaoq/lustre/images/miles_mirror.sqsh}
LAUNCHER=${SANDBOX}/scratch_impl/jobs/launch_train.sh

: "${ARM_CONFIG:?submit with: sbatch --export=ALL,ARM_CONFIG=<abs path to arm_*.sh> submit_train.sbatch}"

if [[ ! -f "${CONTAINER}" ]]; then
    echo "ERROR: container image not found: ${CONTAINER}" >&2
    exit 1
fi
if [[ ! -f "${ARM_CONFIG}" ]]; then
    echo "ERROR: ARM_CONFIG not found: ${ARM_CONFIG}" >&2
    exit 1
fi

mkdir -p "${SANDBOX}/scratch_impl/logs"

echo "===================================================================="
echo "Job:        ${SLURM_JOB_NAME} (${SLURM_JOB_ID}) on $(hostname)"
echo "Stack:      miles (Megatron + SGLang colocated, 1 node x 8 H100)"
echo "Arm config: ${ARM_CONFIG}"
echo "Variant:    MRT_VARIANT=${MRT_VARIANT:-<config default>} MRT_ALPHA=${MRT_ALPHA:-<config default>}"
echo "Container:  ${CONTAINER}"
echo "===================================================================="

srun \
    --container-image="${CONTAINER}" \
    --container-mounts=/lustre:/lustre \
    --no-container-mount-home \
    --no-container-entrypoint \
    --export=ALL \
    --nodes=1 --ntasks=1 \
    bash "${LAUNCHER}"

echo "===================================================================="
echo "Slurm job ${SLURM_JOB_ID} finished."
echo "===================================================================="
