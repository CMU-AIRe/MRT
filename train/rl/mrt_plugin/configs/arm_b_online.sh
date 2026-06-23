# arm_b_online.sh — MRT-online extension (INT-003/004): on-policy break.
# Generates fresh thinking traces from the CURRENT policy, segments online, and
# assigns per-episode dense progress rewards (vs arm B's offline stale prefixes).
#
# Prompt data = BARE questions (arm_a jsonl) — online generates its own prefixes,
# so we do NOT use the offline prefix file. Each of n-samples-per-prompt on-policy
# rollouts is split into a VARIABLE number of episode-samples, hence dynamic batch.
export MRT_VARIANT=${MRT_VARIANT:-explicit}
export MRT_ALPHA=${MRT_ALPHA:-1.0}

ARM=arm_b_online${RUN_TAG:+_${RUN_TAG}}

_CFG_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${_CFG_DIR}/common.sh"

# SMOKE-VERIFY (online smoke): with --use-dynamic-global-batch-size, variable
# episode counts per rollout produce one optimizer step per rollout; confirm
# train_iters and LR schedule resolve sanely (no lr_decay_steps assert).
ROLLOUT_ARGS=(
   --prompt-data ${DATA}/numinamath4k_arm_a.jsonl
   "${ROLLOUT_COMMON[@]}"
   --rollout-batch-size 64
   --n-samples-per-prompt 4
   --global-batch-size 256
   --use-dynamic-global-batch-size
   --num-epoch ${NUM_EPOCH:-1}
)

CUSTOM_ARGS=(
   --custom-generate-function-path mrt_plugin.rollout_mrt_online.generate_online
   --custom-reward-post-process-path mrt_plugin.reward_post.mrt_reward_post
)
