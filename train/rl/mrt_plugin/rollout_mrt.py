"""Arm B (MRT) generation hook: role-by-position continuations + forced terminations.

Wiring (verified against miles @ 9437366e):
- Arm B runs with `--n-samples-per-prompt 8`: the data source builds groups of 8
  identical prefix-prompts with CONSECUTIVE sample indices (data_source.py:104-113),
  so `position = sample.index % 8` is a stable within-group coordinate.
- `--custom-generate-function-path mrt_plugin.rollout_mrt.generate_cont_and_term`
  replaces the per-sample `generate()` call inside `generate_and_rm`
  (sglang_rollout.py:260-272). Positions 0..G-1 generate CONTINUATIONS (budget =
  16K thinking budget minus prefix tokens, A2); positions G..2G-1 mutate their
  prompt with the forced-termination suffix and generate the meta-prover answer
  (eq:mrt_meta_prover), G = n_samples_per_prompt // 2 = 4.
- Termination samples carry `remove_sample=True`: excluded from the loss (loss_mask
  zeroed in train_data_conversion.py:53-54) but present in the train batch — exactly
  what variant-ii's mixed-group normalization needs; for variant-i they are inert
  passengers whose reward the post-process zeroes.
- Keeping every launched sample 1:1 with a returned sample keeps miles' batch/LR
  arithmetic exact: 64 prompts x 8 samples = 512 = global_batch_size -> 1 optimizer
  step per rollout, train_iters = 62 (the sibling-spawning design broke
  `train_iters = num_rollout*rbs*n//gbs` -> lr_decay_steps assert; smoke 4762538).
- The MRT reward itself is assembled in `mrt_plugin.reward_post` from group metadata.

Each prompt row (built by prefix_gen.py) already contains question + "<think>\n" +
z_{0:j}; `metadata.prefix_tokens` is the token length of the thought prefix.
"""

from miles.rollout.base_types import GenerateFnInput, GenerateFnOutput
from miles.rollout.sglang_rollout import generate

from .templates import (
    MIN_CONTINUATION_TOKENS,
    TERMINATION_MAX_TOKENS,
    TOTAL_THINKING_BUDGET,
    termination_prompt,
)


def _continuation_budget(prefix_tokens: int) -> int:
    return max(MIN_CONTINUATION_TOKENS, min(TOTAL_THINKING_BUDGET - prefix_tokens, TOTAL_THINKING_BUDGET))


async def generate_cont_and_term(input: GenerateFnInput) -> GenerateFnOutput:
    args = input.args
    sample = input.sample
    sampling_params = input.sampling_params.copy()
    if input.evaluation:
        out = await generate(args, sample, sampling_params)
        return GenerateFnOutput(samples=out)

    n = args.n_samples_per_prompt
    assert n % 2 == 0, f"arm B needs even n_samples_per_prompt (got {n})"
    position = sample.index % n
    is_termination = position >= n // 2

    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    if is_termination:
        sample.prompt = termination_prompt(sample.prompt)
        sample.metadata = {**metadata, "mrt_kind": "term"}
        sample.remove_sample = True
        sampling_params["max_new_tokens"] = TERMINATION_MAX_TOKENS
    else:
        sample.metadata = {**metadata, "mrt_kind": "cont"}
        prefix_tokens = int(metadata.get("prefix_tokens", 0))
        sampling_params["max_new_tokens"] = _continuation_budget(prefix_tokens)

    out = await generate(args, sample, sampling_params)
    return GenerateFnOutput(samples=out)
