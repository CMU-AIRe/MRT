"""MRT reward post-processing (replaces miles' group normalization wholesale).

Hook: `--custom-reward-post-process-path mrt_plugin.reward_post.mrt_reward_post`.
Called as f(args, samples) where `samples` is the FLAT, group-contiguous list of all
samples in the rollout (verified miles/ray/rollout/train_data_conversion.py:96-98);
must return (raw_rewards, rewards). With `--advantage-estimator grpo`, kl_coef 0 and
no advantage whitening, `rewards[i]` is broadcast as the per-token advantage of
sample i (verified backends/training_utils/loss.py + loss_hub/math_utils.py).

Variants (env MRT_VARIANT, default "explicit"; alpha via MRT_ALPHA, default 1.0):
- explicit (variant-i, Alg-2 literal "in place of A_hat"): continuation i gets
  r_i + alpha * (r_i - J_term) with NO normalization; terminations get 0 (they are
  loss-masked anyway). Note: group-normalizing AFTER this bonus would cancel it
  exactly (J_term is constant within a group) — hence no normalization here.
- mixed (variant-ii, Fig-6 caption reading): mean/std-normalize raw 0/1 rewards over
  the mixed group (continuations + terminations together); terminations act as the
  implicit progress baseline and are loss-masked out of the gradient.

Grouping uses metadata `qid` (stamped by prefix_gen/data_prep), falling back to
`group_index`; groups are contiguous in the flat list.
"""

import os
from statistics import mean


def _group_spans(samples):
    spans = []
    start = 0
    for i in range(1, len(samples) + 1):
        if i == len(samples) or _gkey(samples[i]) != _gkey(samples[start]):
            spans.append((start, i))
            start = i
    return spans


def _gkey(sample):
    md = sample.metadata if isinstance(sample.metadata, dict) else {}
    return md.get("qid", sample.group_index)


def _kind(sample):
    md = sample.metadata if isinstance(sample.metadata, dict) else {}
    return md.get("mrt_kind", "cont")


def mrt_reward_post(args, samples):
    alpha = float(os.environ.get("MRT_ALPHA", "1.0"))
    variant = os.environ.get("MRT_VARIANT", "explicit")
    raw = [float(s.reward) for s in samples]
    rewards = [0.0] * len(samples)

    for start, end in _group_spans(samples):
        idxs = list(range(start, end))
        # MRT-online: rollout fn already finalized per-episode rewards
        # (outcome + alpha*progress). Pass them through unchanged (explicit, no
        # group norm — consistent with arm B variant-i).
        if any(_kind(samples[i]).startswith("online") for i in idxs):
            for i in idxs:
                rewards[i] = raw[i]
            continue
        conts = [i for i in idxs if _kind(samples[i]) == "cont"]
        terms = [i for i in idxs if _kind(samples[i]) == "term"]
        if variant == "explicit":
            j_term = mean(raw[i] for i in terms) if terms else 0.0
            for i in conts:
                rewards[i] = raw[i] + alpha * (raw[i] - j_term)
            for i in terms:
                rewards[i] = 0.0
        elif variant == "mixed":
            vals = [raw[i] for i in idxs]
            m = mean(vals)
            var = mean((v - m) ** 2 for v in vals)
            std = var**0.5
            for i in idxs:
                rewards[i] = (raw[i] - m) / (std + 1e-6)
        else:
            raise ValueError(f"Unknown MRT_VARIANT: {variant}")

    return raw, rewards
