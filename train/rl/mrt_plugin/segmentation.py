"""Episode segmentation for R1-style thought traces (appendix sec:r1_analysis rule).

A "step" is a chunk separated by "\n\n". A new episode starts at a step that begins
with one of the marker phrases, provided the current episode already has >=3 steps
(assumption A10: the analysis rule is reused for training).
"""

from .templates import EPISODE_MARKERS, MIN_STEPS_PER_EPISODE, STEP_DELIMITER

# Longest-first so "But wait" wins over "Wait" prefix checks (same outcome either way,
# but keeps the match unambiguous).
_MARKERS = sorted(EPISODE_MARKERS, key=len, reverse=True)


def split_steps(thought: str) -> list[str]:
    return thought.split(STEP_DELIMITER)


def is_marker_step(step: str) -> bool:
    s = step.lstrip()
    return any(s.startswith(m) for m in _MARKERS)


def episode_boundaries(steps: list[str]) -> list[int]:
    """Step indices that BEGIN a new episode (excluding index 0).

    Enforces MIN_STEPS_PER_EPISODE: a marker step closes the current episode only if
    that episode already spans >= MIN_STEPS_PER_EPISODE steps; otherwise the marker
    step is absorbed into the current episode.
    """
    boundaries = []
    episode_start = 0
    for i in range(1, len(steps)):
        if is_marker_step(steps[i]) and (i - episode_start) >= MIN_STEPS_PER_EPISODE:
            boundaries.append(i)
            episode_start = i
    return boundaries


def prefix_at_boundary(steps: list[str], boundary: int) -> str:
    """The thought prefix z_{0:j} ending right before `boundary`, with a trailing
    delimiter so the continuation starts a fresh step."""
    return STEP_DELIMITER.join(steps[:boundary]) + STEP_DELIMITER


def segment_episodes(thought: str) -> tuple[list[str], list[int]]:
    steps = split_steps(thought)
    return steps, episode_boundaries(steps)
