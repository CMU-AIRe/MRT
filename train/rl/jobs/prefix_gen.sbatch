#!/bin/bash
#SBATCH --job-name=mrt-prefix-gen
#SBATCH --account=nvr_lacr_llm
#SBATCH --partition=batch_block1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=64
#SBATCH --exclusive
#SBATCH --time=03:55:00
#SBATCH --requeue
#SBATCH --output=/lustre/fs1/portfolios/nvr/projects/nvr_lacr_llm/users/yuxiaoq/workspace/AI-Reproduction/sandboxes/paper_002_mrt/scratch_impl/logs/%x_%j.out
#
# prefix_gen.sbatch — offline Arm-B prefix generation (assumption A11: T=1 so
# pi_ref = base model; all 4,000 prefixes generated once). Resumable shards —
# safe to resubmit until numinamath4k_arm_b_prefix.jsonl exists.
# vLLM data-parallel via 1 process, tp=1 is enough for 1.5B; we use tp=1 and
# let vLLM batch all prompts on one GPU per engine — simplest correct setup:
# single engine tp=1 uses GPU 0 only; acceptable (~25M tokens).

set -euo pipefail

SANDBOX=/lustre/fs1/portfolios/nvr/projects/nvr_lacr_llm/users/yuxiaoq/workspace/AI-Reproduction/sandboxes/paper_002_mrt
CODE=${SANDBOX}/scratch_impl/code
DATA=${SANDBOX}/scratch_impl/data

source /lustre/fsw/portfolios/nvr/projects/nvr_lacr_llm/users/yuxiaoq/workspace/Research-skills/.env
export HF_HOME=/lustre/fs1/portfolios/nvr/projects/nvr_lacr_llm/users/yuxiaoq/tmp/huggingface
source /lustre/fsw/portfolios/nvr/projects/nvr_lacr_llm/users/yuxiaoq/envs/miniconda3/bin/activate repro_paper_002_scratch_impl

cd "${CODE}"
python -m mrt_plugin.prefix_gen \
    --input  "${DATA}/numinamath4k_arm_a.jsonl" \
    --output "${DATA}/numinamath4k_arm_b_prefix.jsonl" \
    --model-path deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    --seed 42 --tp 1 \
    --max-tokens 16384 --temperature 0.9 --top-p 1.0 \
    --shard-size 500

echo "prefix_gen done"
