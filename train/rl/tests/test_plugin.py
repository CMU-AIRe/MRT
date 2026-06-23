"""S2 smoke: unit tests for the MRT plugin (CPU-only; miles importable, no GPU).

Run from scratch_impl/code/:  PYTHONPATH=.:miles pytest tests/test_plugin.py -q
"""

import asyncio

import pytest

from mrt_plugin.reward_math import custom_rm
from mrt_plugin.reward_post import mrt_reward_post
from mrt_plugin.segmentation import episode_boundaries, prefix_at_boundary, split_steps
from mrt_plugin.templates import TERMINATION_SUFFIX, termination_prompt

from miles.utils.types import Sample


def make_sample(response, label, kind=None, qid="q0"):
    md = {"qid": qid}
    if kind:
        md["mrt_kind"] = kind
    return Sample(prompt="p", response=response, label=label, metadata=md)


def rm(sample):
    return asyncio.get_event_loop().run_until_complete(custom_rm(None, sample))


class TestRewardMath:
    def test_correct_after_think(self):
        s = make_sample("blah blah</think>\nThe answer is \\boxed{42}.", "42")
        assert rm(s) == 1.0

    def test_wrong_answer(self):
        s = make_sample("x</think>\n\\boxed{41}", "42")
        assert rm(s) == 0.0

    def test_unclosed_think_is_zero(self):
        s = make_sample("thinking forever, the answer is \\boxed{42}", "42")
        assert rm(s) == 0.0  # A19: truncated/unclosed -> 0

    def test_no_boxed_is_zero(self):
        s = make_sample("</think> the answer is 42", "42")
        assert rm(s) == 0.0

    def test_termination_sample_graded_without_think_tag(self):
        # term responses contain no </think> (it lives in the prompt suffix)
        s = make_sample("1. **Conclusion** \\boxed{42}", "42", kind="term")
        assert rm(s) == 1.0

    def test_equivalent_fraction(self):
        s = make_sample("</think>\\boxed{\\frac{1}{2}}", "0.5")
        assert rm(s) == 1.0

    def test_empty_response(self):
        s = make_sample("", "42")
        assert rm(s) == 0.0

    def test_batch_mode(self):
        batch = [make_sample("</think>\\boxed{1}", "1"), make_sample("</think>\\boxed{2}", "3")]
        out = asyncio.get_event_loop().run_until_complete(custom_rm(None, batch))
        assert out == [1.0, 0.0]


class TestSegmentation:
    def test_marker_starts_episode(self):
        steps = ["a", "b", "c", "Wait, hmm", "d", "e", "f", "Alternatively, x"]
        b = episode_boundaries(steps)
        assert b == [3, 7]

    def test_min_steps_rule(self):
        # marker at step 1 ignored (episode would have <3 steps)
        steps = ["a", "Wait b", "c", "d", "But wait e"]
        assert episode_boundaries(steps) == [4]

    def test_prefix_ends_with_delimiter(self):
        steps = ["a", "b", "c", "Wait d"]
        p = prefix_at_boundary(steps, 3)
        assert p == "a\n\nb\n\nc\n\n"

    def test_no_markers(self):
        assert episode_boundaries(["a", "b", "c", "d"]) == []

    def test_split_roundtrip(self):
        text = "a\n\nWait b\n\nc"
        assert split_steps(text) == ["a", "Wait b", "c"]


class TestTemplates:
    def test_termination_prompt(self):
        p = termination_prompt("question<think>\nsome thinking\n\n")
        assert p.endswith(TERMINATION_SUFFIX)
        assert "\n\nTime is up." in p
        assert not p.split("Time is up.")[0].endswith("\n\n\n")


def group8(qid, cont_rewards, term_rewards):
    samples = []
    for r in cont_rewards:
        s = make_sample("x</think>\\boxed{1}", "1", kind="cont", qid=qid)
        s.reward = r
        samples.append(s)
    for r in term_rewards:
        s = make_sample("\\boxed{1}", "1", kind="term", qid=qid)
        s.reward = r
        s.remove_sample = True
        samples.append(s)
    return samples


class TestRewardPost:
    def test_explicit_bonus(self, monkeypatch):
        monkeypatch.setenv("MRT_VARIANT", "explicit")
        monkeypatch.setenv("MRT_ALPHA", "1.0")
        samples = group8("q1", [1, 0, 1, 1], [1, 0, 0, 0])  # J_term = 0.25
        raw, rewards = mrt_reward_post(None, samples)
        assert raw == [1, 0, 1, 1, 1, 0, 0, 0]
        # cont: r + 1.0*(r - 0.25)
        assert rewards[:4] == pytest.approx([1.75, -0.25, 1.75, 1.75])
        assert rewards[4:] == [0.0] * 4

    def test_explicit_alpha(self, monkeypatch):
        monkeypatch.setenv("MRT_VARIANT", "explicit")
        monkeypatch.setenv("MRT_ALPHA", "0.5")
        samples = group8("q1", [1, 1, 1, 1], [0, 0, 0, 0])  # J_term = 0
        _, rewards = mrt_reward_post(None, samples)
        assert rewards[:4] == pytest.approx([1.5] * 4)

    def test_mixed_normalization(self, monkeypatch):
        monkeypatch.setenv("MRT_VARIANT", "mixed")
        samples = group8("q1", [1, 0, 1, 1], [1, 0, 0, 0])
        _, rewards = mrt_reward_post(None, samples)
        assert sum(rewards) == pytest.approx(0.0, abs=1e-6)
        assert rewards[0] > 0 > rewards[1]

    def test_two_groups_independent(self, monkeypatch):
        monkeypatch.setenv("MRT_VARIANT", "explicit")
        monkeypatch.setenv("MRT_ALPHA", "1.0")
        g1 = group8("q1", [1, 1, 1, 1], [1, 1, 1, 1])  # J=1: bonus 0 -> reward 1
        g2 = group8("q2", [0, 0, 0, 0], [0, 0, 0, 0])  # all 0
        _, rewards = mrt_reward_post(None, g1 + g2)
        assert rewards[:4] == pytest.approx([1.0] * 4)
        assert rewards[8:12] == pytest.approx([0.0] * 4)

    def test_arm_a_passthrough_unaffected(self):
        # reward_post is only wired for arm B; this just documents grouping by qid
        # fallback to group_index when qid missing.
        s = Sample(prompt="p", response="r", label="1", metadata={}, group_index=7)
        s.reward = 1.0
        raw, rewards = mrt_reward_post(None, [s])
        assert raw == [1.0]

    def test_online_passthrough(self, monkeypatch):
        monkeypatch.setenv("MRT_VARIANT", "explicit")
        # online episode samples carry finalized rewards; reward_post passes them through
        samples = []
        for r in [1.6, -0.4, 0.9]:
            s = make_sample("ep</think>", "1", kind="online_ep", qid="q1")
            s.reward = r
            samples.append(s)
        raw, rewards = mrt_reward_post(None, samples)
        assert rewards == pytest.approx([1.6, -0.4, 0.9])


class TestOnlineSegments:
    def test_segment_and_progress(self):
        from mrt_plugin.online_segments import segment_trace, progress_rewards
        trace = "a\n\nb\n\nc\n\nWait d\n\ne\n\nf\n\nAlternatively g\n\nh\n\ni"
        eps = segment_trace(trace)
        # 3 episodes: [a,b,c], [Wait d,e,f], [Alternatively g,h,i]
        assert len(eps) == 3
        assert "".join(eps) == trace  # lossless tiling
        # progress: J=[0.2,0.5,0.4,0.9] -> deltas [.3,-.1,.5], reward=outcome+alpha*delta
        rw = progress_rewards([0.2, 0.5, 0.4, 0.9], outcome_r=1.0, alpha=1.0)
        assert rw == pytest.approx([1.3, 0.9, 1.5])

    def test_single_episode_trace(self):
        from mrt_plugin.online_segments import segment_trace
        assert len(segment_trace("just one step\n\nand another")) == 1

    def test_token_spans_tile(self):
        from mrt_plugin.online_segments import episode_token_spans

        class FakeTok:
            # 1 token per char (deterministic) for tiling test
            def encode(self, s, add_special_tokens=False):
                return list(range(len(s)))
        eps = ["abc", "defg", "hi"]
        prompt = "Q?"
        spans = episode_token_spans("".join(eps), eps, FakeTok(), prompt)
        assert spans == [(0, 3), (3, 7), (7, 9)]  # tiles [0,9) exactly
