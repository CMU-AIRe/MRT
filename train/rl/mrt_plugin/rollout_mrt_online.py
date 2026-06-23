"""MRT-online generation hook (INT-003 extension; user-proposed "break online").

Wiring: `--custom-generate-function-path mrt_plugin.rollout_mrt_online.generate_online`
with `--custom-reward-post-process-path mrt_plugin.reward_post.mrt_reward_post`
(MRT_VARIANT=explicit: per-episode scalar passes through as advantage, no group norm).

Per bare-question sample:
  1. Generate ONE full on-policy thinking trace + answer (current policy).
  2. Score the final answer → outcome_r (0/1).
  3. Segment the <think> trace online into episodes (same marker rule as arm B).
  4. At each episode boundary j (including the empty prefix), fork G forced-
     termination branches → J[j] = mean meta-prover success. J has len = n_ep+1.
  5. progress_j = J[j+1] − J[j]; per-episode reward = outcome_r + alpha*progress_j.
  6. Emit one training sample per episode: response = that episode's tokens,
     prompt = question + prior episodes, reward preset.

Contrast with arm B (rollout_mrt.py): there the prefix is a STALE offline base-model
sample and there is ONE break; here the prefix is the CURRENT policy's own fresh
trace and EVERY episode boundary gets a dense reward. This directly tests whether the
paper's off-policy/stale-prefix approximation costs anything (paper open problem #3).

SGLang radix prefix caching makes the J-estimation branches cheap (shared
question+prefix KV; short ≤512-token best-guess answers).

# SMOKE-VERIFY (online smoke, before any production run):
#   - episode_token_spans tiles [0, response_length) exactly (assert).
#   - termination branches emit short boxed answers (not re-opened thinking).
#   - emitted per-episode samples have loss_mask summing to their episode length
#     and reward == outcome_r + alpha*progress_j in the train-data debug dump.
#   - dynamic batch (variable episodes/rollout) works with
#     --use-dynamic-global-batch-size.
"""

import os

from miles.rollout.base_types import GenerateFnInput, GenerateFnOutput
from miles.rollout.rm_hub import async_rm
from miles.rollout.sglang_rollout import generate
from miles.utils.types import Sample

from .online_segments import episode_token_spans, progress_rewards, segment_trace
from .reward_math import _score_one
from .templates import (
    TERMINATION_MAX_TOKENS,
    TERMINATION_SUFFIX,
    THINK_OPEN,
    TOTAL_THINKING_BUDGET,
)

MAX_EPISODES = int(os.environ.get("MRT_MAX_EPISODES", "4"))  # cap branches/rollout (cost)
J_BRANCHES = int(os.environ.get("MRT_J_BRANCHES", "4"))      # forced-termination branches per boundary


def _think_and_answer(full_response):
    """Split a completed rollout into (think_text, has_close). The thinking trace
    is everything inside <think>...</think> (THINK_OPEN already in the prompt)."""
    if "</think>" in full_response:
        return full_response.split("</think>")[0], True
    return full_response, False


async def _estimate_J(args, state, question_prompt, think_prefix, label):
    """Meta-prover success rate at a boundary: append termination suffix to
    (question + think_prefix), generate J_BRANCHES short answers, mean 0/1."""
    base_prompt = question_prompt + think_prefix + "\n\n" + TERMINATION_SUFFIX
    accs = []
    for _ in range(J_BRANCHES):
        s = Sample(prompt=base_prompt, label=label, metadata={"mrt_kind": "term"})
        sp = state.sampling_params.copy()
        sp["max_new_tokens"] = TERMINATION_MAX_TOKENS
        s = await generate(args, s, sp)
        accs.append(_score_one(s))
    return sum(accs) / len(accs) if accs else 0.0


async def generate_online(input: GenerateFnInput) -> GenerateFnOutput:
    args = input.args
    sample = input.sample
    if input.evaluation:
        return GenerateFnOutput(samples=await generate(args, sample, input.sampling_params.copy()))

    state = input.state
    alpha = float(os.environ.get("MRT_ALPHA", "1.0"))
    question_prompt = sample.prompt  # already ends with THINK_OPEN (data_prep render)

    # 1. full on-policy rollout
    sp = input.sampling_params.copy()
    sp["max_new_tokens"] = TOTAL_THINKING_BUDGET
    full = await generate(args, sample, sp)
    outcome_r = _score_one(full)

    # 2. segment the think trace online
    think_text, _closed = _think_and_answer(full.response)
    episodes = segment_trace(think_text)
    if len(episodes) > MAX_EPISODES:  # merge tail into the last kept episode
        episodes = episodes[: MAX_EPISODES - 1] + ["".join(episodes[MAX_EPISODES - 1 :])]
    if len(episodes) < 2:
        # single-episode trace: no per-episode signal — fall back to one sample
        # carrying the plain outcome reward (so the prompt still trains).
        full.reward = outcome_r
        full.metadata = {**(full.metadata or {}), "mrt_kind": "online_single",
                         "qid": (sample.metadata or {}).get("qid", sample.index)}
        return GenerateFnOutput(samples=[full])

    # 3. J estimates at each boundary: J[0] = empty prefix, J[i+1] = after episode i
    j_est = [await _estimate_J(args, state, question_prompt, "", sample.label)]
    cum = ""
    for ep in episodes:
        cum += ep
        j_est.append(await _estimate_J(args, state, question_prompt, cum, sample.label))

    # 4. per-episode dense rewards and token spans
    ep_rewards = progress_rewards(j_est, outcome_r, alpha)
    spans = episode_token_spans(full.response, episodes, state.tokenizer, question_prompt)

    # 5. emit one training sample per episode by SLICING the full rollout.
    # miles treats the last `response_length` of `tokens` as the response, so
    # episode i = tokens[:prompt+b] with response_length (b-a) exposes exactly
    # response_tokens[a:b] as the trained span; rollout_log_probs[a:b] aligns
    # (it is indexed by response token). Slicing avoids re-tokenization drift.
    qid = (sample.metadata or {}).get("qid", sample.index)
    prompt_ids = state.tokenizer.encode(question_prompt, add_special_tokens=False)
    n_prompt = len(prompt_ids)
    response_tokens = full.tokens[n_prompt:]
    full_lp = full.rollout_log_probs or []
    out = []
    for i, (a, b) in enumerate(spans):
        if b <= a or b > len(response_tokens):
            continue
        lp = full_lp[a:b] if len(full_lp) >= b else None
        if lp is None or len(lp) != (b - a):
            continue  # log-prob misalignment (CP/truncation edge) — drop this episode
        s = Sample(
            group_index=sample.group_index,
            index=sample.index,
            prompt=question_prompt + "".join(episodes[:i]),
            tokens=full.tokens[: n_prompt + b],
            response=episodes[i],
            response_length=b - a,
            label=sample.label,
            loss_mask=[1] * (b - a),
            rollout_log_probs=lp,
            metadata={"mrt_kind": "online_ep", "qid": qid, "episode": i,
                      "progress": j_est[i + 1] - j_est[i]},
        )
        s.status = Sample.Status.COMPLETED
        s.reward = ep_rewards[i]   # finalized; reward_post (explicit) passes it through
        out.append(s)
    if not out:
        full.reward = outcome_r
        full.metadata = {**(full.metadata or {}), "mrt_kind": "online_single", "qid": qid}
        return GenerateFnOutput(samples=[full])
    return GenerateFnOutput(samples=out)
