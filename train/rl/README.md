# MRT RL training (v0.1)

RL training + evaluation for the open-ended setting (Table 1, R1-Distill-Qwen-1.5B),
implemented concretely on a vendored [`miles`](miles/) (Megatron-LM + SGLang). This
supersedes the earlier framework-agnostic guideline with runnable code.

**Full recipe → [`REPRODUCTION.md`](REPRODUCTION.md)** (method, assumptions, results,
run commands, infra notes, and the **MRT-online** variant).

## The core idea

Standard GRPO computes advantages against a group baseline (mean reward across rollouts):

$$A(s, a) = R(s, a) - R_{group}$$

MRT adds a **progress bonus**: it rewards each thought episode by how much it raises the
probability that a meta-prover (the model forced to terminate and answer *now*) reaches
the correct answer. With the previous policy's prefix `c` and meta-prover `μ`,

$$r_{\text{prg}}(z_j; c) = J_r\big(\mu(\cdot\mid z_j, c)\big) - J_r\big(\mu(\cdot\mid c)\big),$$

and the trained reward is the outcome reward plus an α-weighted progress term. In code we
estimate `J_r` from G forced-termination ("Time is up… `</think>`") rollouts and apply the
bonus as a single end-of-trace scalar (the paper's plug-and-play form).

## Layout

- `mrt_plugin/` — GRPO baseline (`configs/arm_a.sh`), MRT (`arm_b.sh`, Algorithm 2), and
  **MRT-online** (`arm_b_online.sh`, on-policy branched per-episode progress); reward
  (`reward_post.py`, `reward_math.py`), episode segmentation, prompts, prefix/data prep.
- `eval/` — vLLM pass@1 harness (n=64), benchmark builders, grader, vocab-trim.
- `jobs/` — Slurm sbatch wrappers (4h singleton chains, offline checkpoint conversion).
- `miles/` — vendored radixark/miles @ `9437366e` (Apache-2.0), pristine in its own
  commit; `miles_clone.patch` is the one-line MRT change to it (an in-place
  temperature-div that avoids a long-context OOM).
- `tests/` — CPU unit tests (reward wiring, segmentation, token tiling).
