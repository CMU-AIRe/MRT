"""Offline thought-prefix generation for MRT Arm B (paper_002_mrt).

Because T = 1 outer iteration, pi_ref equals the base model, so ALL thought
prefixes z_{0:j} can be generated once, offline, by this script. Assumptions
implemented here (see scratch_impl/implementation_plan.md, "Assumptions"):

- A9:  j ~ uniform over episode boundaries (j >= 1 when boundaries exist); the
  prefix is capped at templates.MAX_PREFIX_TOKENS by dropping to the largest
  boundary whose prefix fits, else falling back to an empty prefix with j = 0
  (fallbacks are counted in the stats).
- A10: episode segmentation reuses the R1-analysis rule via
  segmentation.segment_episodes ("\\n\\n" steps, marker phrases, >=3 steps per
  episode).
- A11: T = 1 => pi_ref = base model; prefixes are pre-generated offline into a
  resumable sharded JSONL dataset.

For each arm-A row, ONE rollout is sampled with vLLM at the training
temperature (default 0.9). Prompts are the arm-A "input" strings used
VERBATIM — they already end inside "<think>\\n" — pre-tokenized with
add_special_tokens=False and passed as token IDs (avoids the double-BOS that
vLLM's string-prompt path can introduce for R1-distill tokenizers; matches how
miles encodes training inputs). Generation stops at "</think>" (only the
thought is needed); a trace that never closes the think block is kept whole
and marked trace_truncated. Per-request seeds (base seed + a stable qid hash)
make rollouts deterministic and independent of sharding.

Usage:
    python -m mrt_plugin.prefix_gen \
        --input DATA_DIR/numinamath4k_arm_a.jsonl \
        --output DATA_DIR/numinamath4k_arm_b_prefix.jsonl \
        --model-path PATH [--seed 42] [--tp 1] [--max-tokens 16384] \
        [--temperature 0.9] [--top-p 1.0] [--shard-size 500]

Resumability: rows are processed in shards of --shard-size; each shard is
written atomically to OUTPUT.shard_<k>.jsonl and skipped on rerun if it exists
with the expected line count. A final pass concatenates the shards into
--output and writes OUTPUT.stats.json (idempotent).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from collections import Counter
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer, PreTrainedTokenizerBase

from .segmentation import prefix_at_boundary, segment_episodes
from .templates import MAX_PREFIX_TOKENS

THINK_CLOSE = "</think>"


# --------------------------------------------------------------------------- io


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write via a temp file + rename so partial shards never look complete."""
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def count_lines(path: Path) -> int:
    with path.open(encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def shard_file(output: Path, k: int) -> Path:
    return output.with_name(f"{output.name}.shard_{k}.jsonl")


def shard_is_complete(output: Path, k: int, expected: int) -> bool:
    path = shard_file(output, k)
    return path.exists() and count_lines(path) == expected


# ------------------------------------------------------------------ determinism


def stable_qid_hash(qid: str) -> int:
    """Stable 32-bit hash of the qid (Python's hash() is salted per process)."""
    return int.from_bytes(hashlib.sha256(qid.encode("utf-8")).digest()[:4], "big")


def sampling_seed(seed: int, qid: str) -> int:
    """Per-request vLLM seed: base seed + stable qid hash (fits in int64)."""
    return seed + stable_qid_hash(qid)


# ------------------------------------------------------------- prefix selection


def extract_thought(completion: Any) -> tuple[str, bool]:
    """(thought, trace_truncated) from one vLLM CompletionOutput.

    The thought is the text up to THINK_CLOSE, exclusive. Generation uses
    stop=[THINK_CLOSE] (stop string excluded from the text and reported via
    stop_reason), but the in-text split is kept as a defensive fallback.
    """
    text: str = completion.text
    if THINK_CLOSE in text:
        return text.split(THINK_CLOSE, 1)[0], False
    if getattr(completion, "stop_reason", None) == THINK_CLOSE:
        return text, False
    return text, True  # hit max_tokens / EOS without ever closing the think block


def choose_prefix(
    thought: str,
    qid: str,
    seed: int,
    tokenizer: PreTrainedTokenizerBase,
) -> tuple[str, int, int, int]:
    """Pick the thought prefix z_{0:j} for one trace (A9 + A10).

    Returns (prefix, j, n_boundaries, prefix_tokens). j is drawn uniformly
    from range(1, n_boundaries + 1) with random.Random(seed ^ qid_hash) and
    mapped to boundaries[j - 1]. Prefix token counts grow monotonically with
    the boundary index, so when the chosen prefix exceeds MAX_PREFIX_TOKENS we
    scan downward and the first fitting boundary is the largest that fits.
    Fallback (no boundaries, or nothing fits): prefix = "" and j = 0.
    """
    steps, boundaries = segment_episodes(thought)
    n_boundaries = len(boundaries)
    if n_boundaries == 0:
        return "", 0, 0, 0
    rng = random.Random(seed ^ stable_qid_hash(qid))
    j = rng.randint(1, n_boundaries)  # uniform over range(1, n_boundaries + 1)
    while j >= 1:
        prefix = prefix_at_boundary(steps, boundaries[j - 1])
        n_tokens = len(tokenizer.encode(prefix, add_special_tokens=False))
        if n_tokens <= MAX_PREFIX_TOKENS:
            return prefix, j, n_boundaries, n_tokens
        j -= 1
    return "", 0, n_boundaries, 0


# ------------------------------------------------------------------- generation


def build_llm(model_path: str, tp: int, seed: int) -> Any:
    """Construct the vLLM engine (imported lazily: stats/concat-only reruns and
    plain imports of this module must not require vllm)."""
    from vllm import LLM

    return LLM(
        model=model_path,
        tensor_parallel_size=tp,
        trust_remote_code=True,
        seed=seed,
    )


def generate_shard(
    llm: Any,
    tokenizer: PreTrainedTokenizerBase,
    rows: list[dict[str, Any]],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    """One rollout per arm-A row, then prefix selection; returns arm-B rows."""
    from vllm import SamplingParams

    prompts = [
        {"prompt_token_ids": tokenizer.encode(row["input"], add_special_tokens=False)}
        for row in rows
    ]
    params = [
        SamplingParams(
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            stop=[THINK_CLOSE],
            seed=sampling_seed(args.seed, row["metadata"]["qid"]),
        )
        for row in rows
    ]
    request_outputs = llm.generate(prompts, params)

    out_rows: list[dict[str, Any]] = []
    for row, request_output in zip(rows, request_outputs, strict=True):
        qid = row["metadata"]["qid"]
        thought, truncated = extract_thought(request_output.outputs[0])
        prefix, j, n_boundaries, prefix_tokens = choose_prefix(
            thought, qid, args.seed, tokenizer
        )
        out_rows.append(
            {
                "input": row["input"] + prefix,
                "label": row["label"],
                "metadata": {
                    "qid": qid,
                    "prefix_tokens": prefix_tokens,
                    "j": j,
                    "n_boundaries": n_boundaries,
                    "trace_truncated": truncated,
                    "prefix_chars": len(prefix),
                },
            }
        )
    return out_rows


# ------------------------------------------------------------------------ stats


def percentile(sorted_values: list[int], q: float) -> int:
    """Nearest-rank percentile of a non-empty pre-sorted list."""
    idx = round(q * (len(sorted_values) - 1))
    return sorted_values[min(len(sorted_values) - 1, max(0, idx))]


def compute_stats(rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    metas = [row["metadata"] for row in rows]
    prefix_tokens = sorted(m["prefix_tokens"] for m in metas)
    j_counts = Counter(m["j"] for m in metas)
    return {
        "n_rows": len(rows),
        "prefix_tokens": {
            "min": prefix_tokens[0],
            "p50": percentile(prefix_tokens, 0.50),
            "p90": percentile(prefix_tokens, 0.90),
            "max": prefix_tokens[-1],
        },
        "j_distribution": {str(j): j_counts[j] for j in sorted(j_counts)},
        "fallback_no_boundaries": sum(
            1 for m in metas if m["j"] == 0 and m["n_boundaries"] == 0
        ),
        "fallback_token_cap": sum(
            1 for m in metas if m["j"] == 0 and m["n_boundaries"] > 0
        ),
        "trace_truncated": sum(1 for m in metas if m["trace_truncated"]),
        "config": {
            "model_path": args.model_path,
            "seed": args.seed,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_tokens": args.max_tokens,
            "shard_size": args.shard_size,
            "max_prefix_tokens": MAX_PREFIX_TOKENS,
        },
    }


# ------------------------------------------------------------------------- main


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--shard-size", type=int, default=500)
    args = parser.parse_args(argv)
    if args.shard_size <= 0:
        parser.error("--shard-size must be positive")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    rows = read_jsonl(args.input)
    if not rows:
        raise SystemExit(f"no rows in {args.input}")
    shards = [rows[k : k + args.shard_size] for k in range(0, len(rows), args.shard_size)]
    pending = [
        k for k in range(len(shards)) if not shard_is_complete(args.output, k, len(shards[k]))
    ]
    print(
        f"{len(rows)} rows -> {len(shards)} shards of <= {args.shard_size}; "
        f"{len(shards) - len(pending)} complete, {len(pending)} pending"
    )

    if pending:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        llm = build_llm(args.model_path, args.tp, args.seed)
        for k in pending:
            print(f"shard {k}: generating {len(shards[k])} rollouts")
            out_rows = generate_shard(llm, tokenizer, shards[k], args)
            write_jsonl_atomic(shard_file(args.output, k), out_rows)
            print(f"shard {k}: wrote {shard_file(args.output, k)}")

    # Final pass: concatenate shards (idempotent) and emit stats.
    all_rows: list[dict[str, Any]] = []
    for k, shard_rows in enumerate(shards):
        produced = read_jsonl(shard_file(args.output, k))
        if len(produced) != len(shard_rows):
            raise SystemExit(
                f"shard {k} has {len(produced)} rows, expected {len(shard_rows)} "
                f"(input or --shard-size changed? delete {shard_file(args.output, k)})"
            )
        all_rows.extend(produced)
    write_jsonl_atomic(args.output, all_rows)
    print(f"wrote {len(all_rows)} rows -> {args.output}")

    stats = compute_stats(all_rows, args)
    stats_path = args.output.with_name(args.output.name + ".stats.json")
    stats_path.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    print(f"wrote stats -> {stats_path}")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
