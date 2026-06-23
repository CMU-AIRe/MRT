"""Training reward: 0/1 math correctness (assumptions A15/A19).

In-training RM wraps miles' built-in deepscaler grader (signal-safe inside ray
actors; battle-tested in this framework). The offline eval harness uses math-verify;
S2 reports the agreement rate between the two graders.

miles calls a custom RM two ways (verified miles/rollout/rm_hub/__init__.py):
- async_rm(args, sample)            -> custom_rm(args, sample)
- batched_async_rm(args, samples)   -> custom_rm(args, samples)   (batch mode)
so `custom_rm` must accept Sample | list[Sample].
"""

from miles.rollout.rm_hub.deepscaler import get_deepscaler_rule_based_reward
from miles.utils.types import Sample


def _score_one(sample: Sample) -> float:
    response = sample.response or ""
    label = sample.label
    if label is None:
        return 0.0
    # Truncated / empty responses earn 0 (A19). The deepscaler grader itself returns
    # 0 when no answer is extractable, but short-circuit the degenerate cases.
    if not response.strip():
        return 0.0
    # The grader only looks at text after "</think>". For forced-termination rollouts
    # the "</think>" lives in the PROMPT suffix, so the whole response IS the
    # post-think solution — re-add the tag so the grader sees it.
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    if metadata.get("mrt_kind") == "term" and "</think>" not in response:
        response = "</think>\n" + response
    try:
        return float(get_deepscaler_rule_based_reward(response, label))
    except Exception:
        return 0.0


async def custom_rm(args, sample, **kwargs):
    if isinstance(sample, list):
        return [_score_one(s) for s in sample]
    return _score_one(sample)
