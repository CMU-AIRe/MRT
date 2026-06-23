# common.sh — shared miles arg arrays for paper_002_mrt (sourceable, bash).
#
# NOT executable on its own. Sourced by arm_a.sh / arm_b.sh, which MUST set
# ARM (and, for arm B, MRT_VARIANT/MRT_ALPHA) *before* sourcing this file,
# because CKPT_ARGS / WANDB_ARGS are keyed on ARM.
#
# Flag provenance: every flag below was verified to exist either in
#   code/miles/miles/utils/arguments.py            (miles-side flags)
# or in miles/scripts/run-qwen3-4B.sh / run-qwen3-235B-A22B-sft.sh (Megatron
# pass-through flags: --lr-decay-style, --lr-warmup-fraction, --weight-decay,
# --bf16, recompute/dropout/attention flags).

: "${ARM:?common.sh requires ARM to be set before sourcing (source arm_a.sh / arm_b.sh instead)}"

SANDBOX=/lustre/fs1/portfolios/nvr/projects/nvr_lacr_llm/users/yuxiaoq/workspace/AI-Reproduction/sandboxes/paper_002_mrt
CODE=${SANDBOX}/scratch_impl/code
MILES_DIR=${CODE}/miles
RUNS=${SANDBOX}/scratch_impl/runs
DATA=${SANDBOX}/scratch_impl/data
BASE_HF_MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B

mkdir -p "${RUNS}/${ARM}/ckpts" "${RUNS}/${ARM}/hf"

# MODEL_ARGS — R1-Distill-Qwen-1.5B shares the Qwen2.5-1.5B architecture
# (verified in implementation_plan.md). Pins --vocab-size 151936 (padded).
source "${MILES_DIR}/scripts/models/qwen2.5-1.5B.sh"
# R1-Distill config diverges from base Qwen2.5-1.5B in ONE field:
# tie_word_embeddings=false (verified vs HF config.json; smoke 4761804 caught the
# mismatch via miles hf_validate_args). Untie embeddings and lm_head:
MODEL_ARGS+=(--untie-embeddings-and-output-weights)

CKPT_ARGS=(
   --hf-checkpoint ${BASE_HF_MODEL}
   --ref-load ${RUNS}/r1d15b_torch_dist
   --load ${RUNS}/${ARM}/ckpts
   --save ${RUNS}/${ARM}/ckpts
   --save-interval 2
   # save_hf is .format(rollout_id=...)-ed by miles (megatron_utils/model.py:763);
   # without the placeholder every save would overwrite the same dir.
   --save-hf "${RUNS}/${ARM}/hf/iter_{rollout_id}"
)

# Everything rollout-related that is identical across arms. Arm files build
# ROLLOUT_ARGS = ( --prompt-data <arm jsonl> "${ROLLOUT_COMMON[@]}" <batch sizes> ).
# NOTE: deliberately NO --apply-chat-template — prompts are pre-templated
# offline by data_prep.py (assumption A20; miles encodes add_special_tokens=False).
ROLLOUT_COMMON=(
   --input-key input
   --label-key label
   --metadata-key metadata
   --rollout-shuffle
   --rollout-seed 42
   --rollout-temperature 0.9
   --rollout-max-response-len 16384
   # Arm B prompts carry prefixes up to 12,288 tok + template/question headroom.
   # This flag filters longer prompts at dataset init (arguments.py:325).
   --rollout-max-prompt-len 14336
   --custom-rm-path mrt_plugin.reward_math.custom_rm
)

# GRPO + optimizer per implementation_plan.md. KL is OFF by miles defaults
# (verified: --kl-coef default 0.0, --use-kl-loss default False, --entropy-coef
# default 0.0 in arguments.py) — do not pass any KL flag (assumption A12).
# Adam betas / grad clip stay at Megatron/miles defaults (A16; --clip-grad
# default 1.0 via reset_arg in arguments.py).
GRPO_ARGS=(
   --advantage-estimator grpo
   --lr 1e-6
   --lr-decay-style cosine
   --lr-warmup-fraction 0.1
   --weight-decay 0.01
)

PERF_ARGS=(
   # TP2: vocab-parallel logits/CE (per-rank vocab 75,968) + halved resident
   # model/optimizer memory. The OOM lineage at TP1 (4762648 → 4764304 →
   # 4764577 → 4764587 → 4764755/6) kept hitting ~10 GiB full-vocab buffers in
   # fwd/bwd of 16-17K-token sequences despite dist-opt, logprob chunking, and
   # the div_ patch — 80 GB/rank is structurally tight for V=151,936 × T=17K.
   # torch_dist ckpts reshard on load (TP1 iter-25 ckpt resumes under TP2).
   --tensor-model-parallel-size 2
   --pipeline-model-parallel-size 1
   # CP2: TP2 alone did not shrink the ~10 GiB backward buffers (4765193 —
   # the loss path gathers FULL-vocab logits per rank), so split the SEQUENCE:
   # per-rank tokens 17.5K -> ~8.8K halves every per-token buffer the TP gather
   # re-materializes. get_responses has native CP support. DP becomes 2.
   --context-parallel-size 2

   # Shard Adam/master-fp32 states across DP8 (frees ~19 GiB/GPU vs replicated;
   # arm A OOM'd in the fp32 CE logits with only ~9 GiB free — job 4764304).
   # Convention verified in miles/scripts/run-kimi-k25.sh.
   --use-distributed-optimizer
   # Chunk full-vocab log-prob computation over the sequence (the transient
   # fp32 151,936-vocab buffers for a 16K-token sequence need >40 GiB unchunked
   # — jobs 4764304/4764577). 2048-token chunks ≈ 1.2 GiB per buffer.
   --log-probs-chunk-size 2048

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --use-dynamic-batch-size
   # Must exceed the longest single packed sequence. Worst cases: arm B
   # termination = prompt cap 14,336 + 4,096 response = 18,432; arm B/A
   # continuation = long question (≤~3K) + 16,384 budget. 19,456 covers both;
   # under CP2 per-rank tokens stay <10K — within the proven memory envelope
   # (OOM history: 32768 → smoke 4762648; 20480 at TP1 → 4764304/4764577/
   # 4764587; see div_ patch + F002).
   --max-tokens-per-gpu 19456

   # bf16 is also the miles default (backends/megatron_utils/arguments.py:16),
   # pinned here explicitly.
   --bf16
)

# Megatron misc, copied verbatim from miles/scripts/run-qwen3-4B.sh.
MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

SGLANG_ARGS=(
   # TP1 engines -> 8 engines on the 8-GPU node.
   --rollout-num-gpus-per-engine 1
   # VERIFY-S4: 0.7 copied from run-qwen3-4B.sh (colocate); plenty for 1.5B.
   --sglang-mem-fraction-static 0.7
)

# Wandb run name = group + random suffix (miles/utils/tracking_utils/wandb_utils.py:51-56);
# there is no --wandb-run-name flag, the group IS the name stem.
WANDB_ARGS=()
if [[ -n "${WANDB_API_KEY:-}" ]]; then
   # ARM already encodes the variant for arm B (arm_b / arm_b_mixed); only
   # alpha needs appending there.
   WANDB_GROUP="${ARM}"
   [[ -n "${MRT_VARIANT:-}" ]] && WANDB_GROUP="${ARM}-a${MRT_ALPHA:-1.0}"
   WANDB_ARGS=(
      --use-wandb
      --wandb-project repro-paper002-mrt
      --wandb-group "${WANDB_GROUP}"
      --wandb-key "${WANDB_API_KEY}"
   )
fi
