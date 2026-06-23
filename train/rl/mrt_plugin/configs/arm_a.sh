# arm_a.sh — Arm A: vanilla GRPO baseline (stock miles rollout, no custom
# generate fn). Sourced by jobs/launch_train.sh via ARM_CONFIG.
#
# Batch identity (implementation_plan.md): 4000 prompts, --rollout-batch-size 64,
# --n-samples-per-prompt 4 => 256 samples/rollout = --global-batch-size 256
# => 1 optimizer step per rollout, num_rollout = 1 * 4000 // 64 = 62 steps.

# RUN_TAG (env) suffixes the run dir + wandb group so a longer-schedule rerun
# (RUN_TAG=long NUM_EPOCH=4 → ~248 steps, repair ATT-001) gets its own ckpts/hf
# without clobbering the 62-step baseline. NUM_EPOCH (env) sets training length;
# the cosine LR schedule spans the resulting train_iters automatically.
ARM=arm_a${RUN_TAG:+_${RUN_TAG}}

_CFG_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${_CFG_DIR}/common.sh"

ROLLOUT_ARGS=(
   --prompt-data ${DATA}/numinamath4k_arm_a.jsonl
   "${ROLLOUT_COMMON[@]}"
   --rollout-batch-size 64
   --n-samples-per-prompt 4
   --global-batch-size 256
   --num-epoch ${NUM_EPOCH:-1}
)

# Arm A uses the stock per-sample generation path — no custom generate fn,
# no reward post-process (stock GRPO group normalization applies).
CUSTOM_ARGS=()
