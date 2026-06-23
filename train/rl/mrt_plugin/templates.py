"""Shared constants for the MRT reproduction (paper_002_mrt).

Sources are the paper's LaTeX only (method_spec.md is the contract):
- termination suffix: appendix fig:r1_stop, verbatim.
- episode markers: appendix fig:r1_wait, verbatim list.
"""

# Forced-termination suffix (meta-prover mu, eq:mrt_meta_prover + fig:r1_stop).
# Appended after the thought prefix; generation continues right after "1. **".
TERMINATION_SUFFIX = (
    "Time is up.\n\n"
    "Given the time I've spent and the approaches I've tried, I should stop thinking "
    "and formulate a final answer based on what I already have.\n"
    "</think>\n\n"
    "**Step-by-Step Explanation and Answer:**\n\n"
    "1. **"
)

# Episode-boundary marker phrases (fig:r1_wait). A step beginning with one of these
# starts a new episode (subject to the >=3-steps-per-episode rule).
EPISODE_MARKERS = [
    "Wait",
    "But wait",
    "Alternatively",
    "Is there another way to think about this?",
    "But let me double-check",
    "But hold on",
]

# Step delimiter within the thought block (appendix "Construction of episodes").
STEP_DELIMITER = "\n\n"

# Minimum steps per episode (appendix: "at least three steps").
MIN_STEPS_PER_EPISODE = 3

# A2: total thinking budget in tokens (main text: 16K; prefix counts against it).
TOTAL_THINKING_BUDGET = 16384
# Floor for the continuation budget when the prefix is long (assumption, ledger A2).
MIN_CONTINUATION_TOKENS = 2048
# Cap for forced-termination generations (assumption, ledger A13).
TERMINATION_MAX_TOKENS = 4096
# Prefix length cap used at prefix-generation time (keeps continuation budget sane).
MAX_PREFIX_TOKENS = 12288

# Marker that begins the thought block in R1-style prompts.
THINK_OPEN = "<think>\n"


def termination_prompt(prompt: str) -> str:
    """Build the meta-prover prompt: thought-so-far + forced termination."""
    return prompt.rstrip() + "\n\n" + TERMINATION_SUFFIX


def render_prompt(tokenizer, question: str) -> str:
    """Apply the model's chat template and ensure generation starts inside <think>.

    R1-distill chat templates differ on whether they already append "<think>\\n" after
    the assistant tag; normalize so every prompt ends with exactly one THINK_OPEN.
    """
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": question}],
        tokenize=False,
        add_generation_prompt=True,
    )
    if not rendered.rstrip().endswith("<think>"):
        rendered = rendered + THINK_OPEN
    elif not rendered.endswith("\n"):
        rendered = rendered + "\n"
    return rendered
