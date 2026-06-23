#!/bin/bash
# launch_train.sh — in-container miles launcher for paper_002_mrt.
# Executed by srun inside miles_mirror.sqsh (see submit_train.sbatch).
# Structure copied from miles/scripts/run-qwen3-4B.sh (ray head + ray job
# submit on a single node), adapted to ARM_CONFIG-driven arg arrays.
#
# Required env: ARM_CONFIG = absolute path to mrt_plugin/configs/arm_*.sh
# Auto-resume: train.py resumes from --load when
# <load>/latest_checkpointed_iteration.txt exists — keep per-arm dirs stable
# across chain segments and resume is automatic.

# ---- cleanup from any previous attempt on this node (pre set -e: pkill may
# return nonzero when nothing matches) ------------------------------------
pkill -9 sglang
sleep 3
ray stop --force
pkill -9 ray
pkill -9 python
sleep 3
pkill -9 ray
pkill -9 python

set -ex

: "${ARM_CONFIG:?launch_train.sh requires ARM_CONFIG (path to arm_*.sh)}"
[[ -f "${ARM_CONFIG}" ]] || { echo "ERROR: ARM_CONFIG not found: ${ARM_CONFIG}" >&2; exit 1; }

# ---- secrets + caches -----------------------------------------------------
source /lustre/fsw/portfolios/nvr/projects/nvr_lacr_llm/users/yuxiaoq/workspace/Research-skills/.env
export HF_HOME=/lustre/fs1/portfolios/nvr/projects/nvr_lacr_llm/users/yuxiaoq/tmp/huggingface

# ---- config: defines ARM, MODEL_ARGS, CKPT_ARGS, ROLLOUT_ARGS, GRPO_ARGS,
#      CUSTOM_ARGS, PERF_ARGS, MISC_ARGS, SGLANG_ARGS, WANDB_ARGS, and paths
#      SANDBOX/CODE/MILES_DIR/RUNS/DATA -------------------------------------
source "${ARM_CONFIG}"

# ROLLOUT_ARGS[0]=--prompt-data, [1]=<jsonl path> by construction in arm_*.sh
[[ -f "${ROLLOUT_ARGS[1]}" ]] || { echo "ERROR: prompt data missing: ${ROLLOUT_ARGS[1]}" >&2; exit 1; }
[[ -f "${RUNS}/r1d15b_torch_dist/latest_checkpointed_iteration.txt" ]] || {
   echo "ERROR: base torch_dist ckpt missing — run prep_megatron_ckpt.sbatch first" >&2; exit 1; }

# will prevent ray from buffering stdout/stderr
export PYTHONBUFFERED=16
export PYTHONUNBUFFERED=1

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

cd "${MILES_DIR}"

# ---- ray head (single node => loopback master, as in run-qwen3-4B.sh) ------
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 8 --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

# PYTHONPATH inside ray workers:
#   /root/Megatron-LM — Megatron in the container image (run-qwen3-4B.sh pattern)
#                       VERIFY-S4: confirm this path exists in miles_mirror.sqsh
#   ${MILES_DIR}      — our miles clone shadows any pip-installed miles
#   ${CODE}           — makes mrt_plugin importable
# MRT_VARIANT/MRT_ALPHA must reach the reward post-process running inside ray
# workers, hence they go through the runtime env too.
RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM:${MILES_DIR}:${CODE}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"HF_HOME\": \"${HF_HOME}\",
    \"HF_TOKEN\": \"${HF_TOKEN:-}\",
    \"WANDB_API_KEY\": \"${WANDB_API_KEY:-}\",
    \"MRT_VARIANT\": \"${MRT_VARIANT:-}\",
    \"MRT_ALPHA\": \"${MRT_ALPHA:-}\"
  }
}"

# ---- SMOKE mode (S4): 1 tiny rollout + debug dumps, separate ckpt dir ------
# argparse lets later occurrences override earlier ones, so SMOKE_ARGS is
# appended LAST. num-rollout overrides num-epoch (arguments.py:618).
SMOKE_ARGS=()
if [[ "${SMOKE:-0}" == "1" ]]; then
   SMOKE_DIR=${RUNS}/${ARM}/smoke
   mkdir -p "${SMOKE_DIR}"
   # global-batch-size must equal launched samples (8 prompts × n_samples) so
   # train_iters = num_rollout > 0 (else lr_decay_steps==0 asserts). Derive
   # n_samples from the config's ROLLOUT_ARGS instead of guessing from CUSTOM_ARGS
   # (arm B=8, arm A=4, online=4 — the old CUSTOM_ARGS heuristic mis-set online).
   _NS=4
   for _i in "${!ROLLOUT_ARGS[@]}"; do
      [[ "${ROLLOUT_ARGS[$_i]}" == "--n-samples-per-prompt" ]] && _NS=${ROLLOUT_ARGS[$((_i+1))]}
   done
   GBS=$((8 * _NS))
   SMOKE_ARGS=(
      --num-rollout "${SMOKE_NUM_ROLLOUT:-1}"
      --rollout-batch-size 8
      --global-batch-size ${GBS}
      --rollout-max-response-len 2048
      --save-interval 1
      --load "${SMOKE_DIR}/ckpts"
      --save "${SMOKE_DIR}/ckpts"
      --save-hf "${SMOKE_DIR}/hf/iter_{rollout_id}"
      --save-debug-rollout-data "${SMOKE_DIR}/debug_rollout_{rollout_id}.pt"
      --save-debug-train-data "${SMOKE_DIR}/debug_train_{rollout_id}_rank{rank}.pt"
   )
   WANDB_ARGS=()   # no wandb noise from smokes
fi

# --rollout-function-path is pinned to the stock sglang rollout (this IS the
# default in arguments.py:284 when the experimental refactor flag is off);
# arm B customizes per-sample generation via CUSTOM_ARGS instead.
# --rollout-num-gpus is ignored under --colocate (arguments.py:54: set to
# actor_num_nodes * actor_num_gpus_per_node = 8), passed for explicitness.
ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 8 \
   --rollout-num-gpus 8 \
   --colocate \
   --rollout-function-path miles.rollout.sglang_rollout.generate_rollout \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${CUSTOM_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]} \
   ${SMOKE_ARGS[@]}
