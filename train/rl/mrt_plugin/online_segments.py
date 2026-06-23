"""Pure (CPU-testable) logic for MRT-online: split an on-policy rollout into
per-episode training samples with per-episode progress rewards.

Kept separate from rollout_mrt_online.py (which needs the GPU rollout engine) so
the token-accounting and progress math can be unit-tested without miles/SGLang.

MRT-online (INT-003 extension, user-proposed): unlike arm B (offline stale
prefixes), here the thinking trace is generated ON-POLICY in one shot, segmented
online, and each episode gets a dense progress reward
  progress_j = J_after_j − J_before_j
where J_* is the meta-prover (forced-termination) success rate conditioned on the
episodes so far. The episode's tokens are trained with reward
  r_episode_j = outcome_r + alpha * progress_j.
"""

from .segmentation import episode_boundaries, split_steps
from .templates import STEP_DELIMITER


def episode_token_spans(full_text, episode_texts, tokenizer, prompt_text):
    """Map each episode's TEXT to a (start, end) span in the response token list.

    Re-encodes cumulative prefixes and diffs their lengths so each episode-sample
    can be emitted with response = just that episode's tokens. add_special_tokens
    is False to match miles' rollout encoding (sglang_rollout.py:154).

    Returns: list of (tok_start, tok_end) offsets INTO THE RESPONSE (0 = first
    generated token), one per episode. Concatenated they tile [0, n_response_tok).

    # SMOKE-VERIFY: re-encoding prefix(question+think_so_far) then subtracting the
    # encoded prompt length must reproduce the rollout's own response_length within
    # tokenizer round-trip tolerance; assert exact tiling in the online smoke.
    """
    prompt_len = len(tokenizer.encode(prompt_text, add_special_tokens=False))
    spans = []
    prev_end = 0
    cum = prompt_text
    for ep in episode_texts:
        cum = cum + ep
        cum_total = len(tokenizer.encode(cum, add_special_tokens=False))
        end = cum_total - prompt_len
        spans.append((prev_end, end))
        prev_end = end
    return spans


def segment_trace(think_text):
    """Split a thinking trace into episode TEXTS (each ends with the step
    delimiter so concatenation is lossless). Returns list of episode strings;
    a trace with no marker boundaries yields a single episode (the whole trace)."""
    steps = split_steps(think_text)
    bounds = episode_boundaries(steps)
    cuts = [0] + bounds + [len(steps)]
    episodes = []
    for a, b in zip(cuts[:-1], cuts[1:]):
        seg = STEP_DELIMITER.join(steps[a:b])
        # preserve the trailing delimiter between episodes (not after the last step)
        if b < len(steps):
            seg = seg + STEP_DELIMITER
        episodes.append(seg)
    return episodes


def progress_rewards(j_estimates, outcome_r, alpha):
    """Per-episode dense reward from meta-prover success estimates.

    j_estimates[i] = J after episode i (success rate of forced-termination at the
    boundary FOLLOWING episode i). j_before for episode 0 is j_estimates pre-context
    (success with empty thinking) passed as j_estimates[-1]-style? No: we pass the
    full list [J_before_0, J_after_0, J_after_1, ...] of length n_ep+1, so
    progress_j = J[j+1] − J[j].

    Returns list of per-episode rewards: outcome_r + alpha * progress_j.
    """
    assert len(j_estimates) >= 2, "need J_before_0 plus one J per episode"
    rewards = []
    for j in range(len(j_estimates) - 1):
        progress = j_estimates[j + 1] - j_estimates[j]
        rewards.append(outcome_r + alpha * progress)
    return rewards
