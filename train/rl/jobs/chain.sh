#!/bin/bash
# chain.sh — submit N consecutive <=4h training segments as a Slurm singleton
# chain (batch_block1's hard 4h walltime forces segmentation; checkpoints +
# miles auto-resume stitch the segments into one run).
#
# Usage:
#   ./chain.sh <ARM_CONFIG> <N_SEGMENTS> [JOB_NAME]
#
#   ARM_CONFIG  path to mrt_plugin/configs/arm_*.sh (resolved to absolute)
#   N_SEGMENTS  number of submit_train.sbatch copies to queue
#   JOB_NAME    optional; default mrt-<config basename>. All segments share
#               this name + --dependency=singleton, so Slurm runs them one
#               at a time in submission order regardless of how the previous
#               segment ended (COMPLETED / TIMEOUT / FAILED all release the
#               singleton). To abort the chain: scancel --name=<JOB_NAME>.
#
# Examples:
#   ./chain.sh ../code/mrt_plugin/configs/arm_a.sh 8
#   MRT_ALPHA=0.5 ./chain.sh ../code/mrt_plugin/configs/arm_b.sh 10 mrt-arm_b-a0.5
#
# Not an sbatch itself — run from a login/DC node.

set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <ARM_CONFIG> <N_SEGMENTS> [JOB_NAME]" >&2
    exit 1
fi

ARM_CONFIG=$(readlink -f "$1")
N=$2
JOB_NAME=${3:-mrt-$(basename "${ARM_CONFIG}" .sh)}

SANDBOX=/lustre/fs1/portfolios/nvr/projects/nvr_lacr_llm/users/yuxiaoq/workspace/AI-Reproduction/sandboxes/paper_002_mrt
SBATCH_SCRIPT=${SANDBOX}/scratch_impl/jobs/submit_train.sbatch

if [[ ! -f "${ARM_CONFIG}" ]]; then
    echo "ERROR: ARM_CONFIG not found: ${ARM_CONFIG}" >&2
    exit 1
fi
if [[ ! -f "${SBATCH_SCRIPT}" ]]; then
    echo "ERROR: sbatch script not found: ${SBATCH_SCRIPT}" >&2
    exit 1
fi

# Forward MRT_* overrides (if the caller set them) into the jobs; arm_*.sh
# applies its own defaults when they are absent from the environment.
EXPORTS="ALL,ARM_CONFIG=${ARM_CONFIG}"
[[ -n "${MRT_VARIANT:-}" ]] && EXPORTS+=",MRT_VARIANT=${MRT_VARIANT}"
[[ -n "${MRT_ALPHA:-}" ]] && EXPORTS+=",MRT_ALPHA=${MRT_ALPHA}"

echo "================================================================"
echo "Singleton chain: ${N} segments of ${SBATCH_SCRIPT}"
echo "  job name:   ${JOB_NAME}"
echo "  arm config: ${ARM_CONFIG}"
echo "  exports:    ${EXPORTS}"
echo "================================================================"

JOB_IDS=()
for i in $(seq 1 "${N}"); do
    JID=$(sbatch --parsable \
        --job-name="${JOB_NAME}" \
        --dependency=singleton \
        --export="${EXPORTS}" \
        "${SBATCH_SCRIPT}")
    JOB_IDS+=("${JID}")
    echo "  segment $(printf '%02d' "${i}"): jobid ${JID}"
done

echo "================================================================"
echo "Submitted job IDs: ${JOB_IDS[*]}"
echo "Watch:  squeue -u \$USER --name=${JOB_NAME} -o '%.12i %.24j %.8T %.10M %.10l %R'"
echo "Abort:  scancel --name=${JOB_NAME}"
echo "================================================================"
