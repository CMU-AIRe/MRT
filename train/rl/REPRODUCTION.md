# MRT RL — Training & Reproduction (v0.1)

This directory contains the RL training and evaluation code for the open-ended setting
of *Optimizing Test-Time Compute via Meta Reinforcement Fine-Tuning* (Table 1,
DeepSeek-R1-Distill-Qwen-1.5B), plus the **MRT-online** variant.

It is built on [`miles`](https://github.com/radixark/miles) (Megatron-LM + SGLang). The
framework is vendored pristine at [`miles/`](miles/) (commit `9437366e`, Apache-2.0) in a
**separate commit**, and all MRT-specific code is added on top — so
`git show <the MRT commit>` (or `git diff` against the vendor commit) is exactly the MRT
delta over miles. The only change to miles itself is a one-line in-place
temperature-division fix (`miles_clone.patch`) that avoids a full-vocab fp32 copy OOM on
long sequences.

> Note for reproducers: hyperparameters the paper does not state are filled with the
> explicit choices in the **Assumptions** table below — treat those as this release's
> defaults. Absolute pass@1 depends on the answer grader (we use one grader end-to-end
> for train and eval); the headline metric is the **gain over the base model**, which is
> grader-internal and robust.

## Result (5-benchmark avg pass@1 gain over base)

DeepSeek-R1-Distill-Qwen-1.5B, 4,000 NuminaMath problems, eval pass@1 (mean of 64
samples) at 16K budget on AIME24 / AIME25 / AMC23 / MinervaMATH / MATH500.

| training length | GRPO gain | MRT gain | MRT/GRPO ratio | paper |
|---|---:|---:|---:|---|
| 62 steps (literal "1 epoch") | +0.09 | +0.41 | — (noise) | direction only |
| **248 steps** (4 epochs) | **+0.48** | **+1.20** | **2.50×** | +1.1 / +2.2, 2–3× |
| MRT-online, 248 steps | — | +0.62 | 1.29× | (extension) |

At ~248 steps both arms land within ±1.0 pt of the paper's gains and the MRT/GRPO ratio
(2.50×) is inside the paper's 2–3×; ordering MRT > GRPO > base holds. The 62-step run
reproduces the *effect* but at compressed magnitude — an undertraining artifact of the
literal "1 epoch, batch 256" reading (see assumption **A1**; the paper's Fig. 10 implies
a much longer schedule). Per-benchmark numbers at the end of this file.

## Method

Two trained arms from the same base, identical except for the reward:

- **GRPO baseline** (`mrt_plugin/configs/arm_a.sh`): vanilla outcome-reward GRPO, 0/1
  math correctness, group size 4.
- **MRT** (`mrt_plugin/configs/arm_b.sh`): Algorithm 2 — per prompt, sample a
  random-length thought **prefix from the previous policy**, then G=4 on-policy
  continuations + G=4 forced-termination ("Time is up… `</think>`") rollouts. The
  terminations' mean accuracy `J_term` estimates the meta-prover value; the continuation
  reward is `r + α·(r − J_term)` applied as a single end-of-trace scalar. Implemented as a
  per-sample generate hook (`mrt_plugin/rollout_mrt.py`) + a reward post-processor
  (`mrt_plugin/reward_post.py`), role-by-position over `--n-samples-per-prompt 8`.

`mrt_plugin/prefix_gen.py` produces the offline base-model prefixes (T=1);
`mrt_plugin/data_prep.py` builds the NuminaMath JSONL; `segmentation.py` / `templates.py`
hold the episode-marker rule and the termination prompt. CPU unit tests:
`tests/test_plugin.py` (reward wiring, segmentation, token tiling).

### Assumptions (not fixed by the paper)

| id | choice | note |
|---|---|---|
| A1 | 248 steps (4 epochs) | appendix "1 epoch, batch 256" ⇒ 62 steps; Fig. 10's step axis implies ~1,400; 62 reproduces direction only, 248 reproduces magnitude |
| A2 | 16K thinking budget (incl. prefix) | main text "16K"; appendix table says `max_completion_length 24576` |
| A3 | eval temp 0.6 / top_p 0.95, n=64 | "DeepScaleR protocol"; n raised from 20 for variance |
| A4 | α = 1.0 | progress-bonus weight |
| A6 | explicit bonus, no extra group-norm | Alg-2 "in place of Â"; a `mixed` variant (terminations inside the GRPO group) is `arm_b_variant2.sh` |
| A15 | one grader (math-verify/deepscaler) for train **and** eval | grader convention dominates minerva (≈8-pt swing); using one grader end-to-end keeps train and eval consistent and the gain metric robust |

## MRT-online (highlighted variant)

The paper applies the progress bonus over an **off-policy stale prefix** as a **single
end-of-trace scalar**, and lists as an open problem: *"How can we implement branched
rollouts, from a different meta-prover policy, in a computationally-efficient manner?"*

`mrt_plugin/rollout_mrt_online.py` does this: generate the trace **on-policy**, segment it
**online**, fork G forced-termination branches at **each** episode boundary to estimate
progress `J_{j+1} − J_j`, and assign a **per-episode dense reward**. SGLang radix prefix
caching keeps the per-boundary branches cheap. (`online_segments.py` holds the
CPU-testable segmentation / token-tiling / progress math.)

**Finding:** MRT-online beats outcome-reward GRPO (+0.62 vs +0.48) but **trails the
offline MRT** (+1.20). The on-policy per-episode signal helps, but is higher-variance
(each episode's progress from only G=4 branches), so it is noisier at this budget — empirical
support for the single off-policy scalar the paper uses. Caveats: single seed; ≤6
episodes/rollout; G=4 gives coarse J ∈ {0,¼,½,¾,1}; larger G or longer training may close the gap.

## How to run

Prereqs: build an env/container from the vendored `miles/` (its README + the
`miles_clone.patch` already applied here); a base→Megatron-torch_dist checkpoint of the
base model. **Paths in `mrt_plugin/configs/common.sh` and `jobs/*` are cluster-specific —
set `SANDBOX` / `HF_HOME` / `RUNS` for your environment.**

```bash
# 1. data + benchmarks
python -m mrt_plugin.data_prep    --output-dir data --n 4000 --seed 42
python eval/prep_benchmarks.py    --output-dir data/benchmarks
# 2. base model -> Megatron torch_dist
sbatch jobs/prep_megatron_ckpt.sbatch
# 3a. GRPO baseline (248 steps = 4 epochs)
RUN_TAG=long NUM_EPOCH=4 jobs/chain.sh mrt_plugin/configs/arm_a.sh 3 mrt-arma-long
# 3b. MRT (offline prefixes first)
sbatch jobs/prefix_gen.sbatch
RUN_TAG=long NUM_EPOCH=4 jobs/chain.sh mrt_plugin/configs/arm_b.sh 3 mrt-armb-long
# 3c. MRT-online
RUN_TAG=long NUM_EPOCH=4 jobs/chain.sh mrt_plugin/configs/arm_b_online.sh 6 mrt-online-long
# 4. eval: convert Megatron->HF offline, then eval n=64
sbatch --export=ALL,CKPT_DIR=runs/arm_b_long/ckpts,OUT_DIR=runs/arm_b_long/hf jobs/convert_hf.sbatch
sbatch --export=ALL,MODEL_PATH=runs/arm_b_long/hf,N_SAMPLES=64,OUT_DIR=runs/eval/arm_b_long jobs/eval.sbatch
```

### Infra notes (large-vocab + long-context RL on 1.5B)

- **Train OOM** (full-vocab 151,936 fp32 logit/CE buffers for 16K+ sequences): use TP=2 +
  context-parallel=2 + `--use-distributed-optimizer` + `--log-probs-chunk-size 2048` + the
  in-place `div_` patch; cap `--max-tokens-per-gpu` just above the longest sequence. Fix
  parallelism *before* the production run — distributed-optimizer state can't reshard
  across a TP change.
- **Checkpoint export**: the in-training HF export was unreliable under TP2+CP2; convert
  the Megatron torch_dist checkpoint **offline** (`jobs/convert_hf.sbatch`) and trim the
  Megatron vocab padding (`eval/trim_vocab_padding.py`) before serving in vLLM.
- **Env pins** (eval/data side): a CUDA-12.4 driver needs `vllm==0.8.5.post1`
  (torch 2.6.0+cu124) and `transformers==4.51.3`.

## Per-benchmark numbers (248 steps, n=64)

| benchmark | base | GRPO | MRT | MRT-online | paper base/GRPO/MRT |
|---|---:|---:|---:|---:|---|
| AIME 2024 | 27.34 | 28.12 | 28.75 | 28.59 | 28.7 / 29.8 / 30.3 |
| AIME 2025 | 22.86 | 22.97 | 23.59 | 22.24 | 26.0 / 27.3 / 29.3 |
| AMC 2023 | 67.89 | 67.77 | 70.86 | 68.79 | 69.9 / 70.5 / 72.9 |
| MinervaMATH | 24.94 | 26.45 | 24.96 | 25.87 | 19.8 / 22.1 / 22.5 |
| MATH500 | 81.71 | 81.85 | 82.61 | 82.37 | 80.1 / 80.3 / 80.4 |
| **Avg** | **44.95** | **45.43** | **46.16** | **45.57** | **44.9 / 46.0 / 47.1** |

## Provenance

- training framework: `radixark/miles` @ `9437366e` (Apache-2.0), vendored at `miles/`
- base model: `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` @ `ad9f0ae0`
- training data: `AI-MO/NuminaMath-CoT`, seed-42 4,000-problem sample
- benchmarks: deepscaler/rllm test JSONs (AIME24/MATH500/Minerva), `math-ai/amc23`,
  `opencompass/AIME2025`
- single seed per arm.
