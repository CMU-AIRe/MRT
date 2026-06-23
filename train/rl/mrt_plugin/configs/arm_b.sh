# arm_b.sh — Arm B: MRT (Algorithm 2, T=1), prefix dataset + termination
# siblings. Sourced by jobs/launch_train.sh via ARM_CONFIG.
#
# Variant switch (read by mrt_plugin/reward_post.py at reward time):
#   MRT_VARIANT=explicit (default) — variant-i, Alg-2 literal: r + alpha*(r - J_term),
#                                    NO group normalization; terminations zeroed.
#   MRT_VARIANT=mixed              — variant-ii: stock-style group mean/std
#                                    normalization over conts+terms together.
# MRT_ALPHA (default 1.0) — progress-bonus weight (assumption A4).
export MRT_VARIANT=${MRT_VARIANT:-explicit}
export MRT_ALPHA=${MRT_ALPHA:-1.0}

# Keep checkpoints / wandb groups of the two variants apart. RUN_TAG (env)
# adds a schedule suffix (e.g. RUN_TAG=long + NUM_EPOCH=4 → ~248 steps, ATT-001).
if [[ "${MRT_VARIANT}" == "explicit" ]]; then
   ARM=arm_b
else
   ARM=arm_b_${MRT_VARIANT}
fi
ARM=${ARM}${RUN_TAG:+_${RUN_TAG}}

_CFG_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${_CFG_DIR}/common.sh"

# Batch identity: 64 prompts x 8 samples (positions 0-3 continuations, 4-7
# forced terminations, assigned by generate_cont_and_term) = 512 samples =
# --global-batch-size 512 => exactly 1 optimizer step per rollout and
# train_iters = 1*4000//64 = 62 (miles formula stays exact; the earlier
# sibling-spawning design broke lr_decay_steps — smoke 4762538).
ROLLOUT_ARGS=(
   --prompt-data ${DATA}/numinamath4k_arm_b_prefix.jsonl
   "${ROLLOUT_COMMON[@]}"
   --rollout-batch-size 64
   --n-samples-per-prompt 8
   --global-batch-size 512
   --num-epoch ${NUM_EPOCH:-1}
)

CUSTOM_ARGS=(
   # Plugs into the stock sglang generate_rollout per-sample hook
   # (arguments.py:447, sglang_rollout.py:262).
   --custom-generate-function-path mrt_plugin.rollout_mrt.generate_cont_and_term
   # Applies the MRT progress bonus / variant logic after custom_rm scoring
   # (arguments.py:1549).
   --custom-reward-post-process-path mrt_plugin.reward_post.mrt_reward_post
)
