"""Arm-A training-data preparation for the MRT reproduction (paper_002_mrt).

Builds the 4,000-problem NuminaMath-CoT training subset and applies the chat
template offline. Assumptions implemented here (see
scratch_impl/implementation_plan.md, "Assumptions" ledger):

- A14: the 4,000-problem subset is a seed-42 uniform sample from the
  NuminaMath-CoT train split, filtered to boxed-extractable, non-degenerate
  answers (the paper only says the data is from NuminaMath).
- A20: the training prompt template is the same chat template as eval
  (templates.render_prompt), applied offline so miles can encode inputs with
  add_special_tokens=False; every prompt ends inside "<think>\\n".

Output (one JSON object per line in DATA_DIR/numinamath4k_arm_a.jsonl):
    {"input": <templated prompt>, "label": <boxed answer>,
     "metadata": {"qid": "numina-<original_row_index>", "source": <source col>}}

A sidecar DATA_DIR/numinamath4k_meta.json records the filter funnel, seed, n,
model path, and dataset fingerprint. HF_HOME / HF caches come from the
environment.

Usage:
    python -m mrt_plugin.data_prep --output-dir DATA_DIR \
        [--n 4000] [--seed 42] [--model-path PATH]
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any

from datasets import Dataset, load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from .templates import render_prompt

DATASET_NAME = "AI-MO/NuminaMath-CoT"
SPLIT = "train"
DEFAULT_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
BOXED_KEY = "\\boxed{"
MAX_ANSWER_CHARS = 64
OUTPUT_NAME = "numinamath4k_arm_a.jsonl"
META_NAME = "numinamath4k_meta.json"
_FILTER_BATCH = 1024


def extract_last_boxed(text: str) -> str | None:
    """Content of the LAST well-formed ``\\boxed{...}`` in ``text``.

    Uses brace matching (nested braces such as ``\\boxed{\\frac{1}{2}}`` are
    common, so a non-greedy regex is not enough). Scans occurrences from the
    end and returns the first one that closes properly; unbalanced
    occurrences are skipped.
    """
    end = len(text)
    while True:
        start = text.rfind(BOXED_KEY, 0, end)
        if start == -1:
            return None
        content_start = start + len(BOXED_KEY)
        depth = 1
        for i in range(content_start, len(text)):
            char = text[i]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return text[content_start:i]
        end = start  # unbalanced occurrence: try an earlier one


def build_funnel(ds: Dataset) -> tuple[list[tuple[int, str]], dict[str, int]]:
    """Sequential filter over the dataset; returns (eligible, funnel_counts).

    ``eligible`` is a list of (original_row_index, extracted_answer) for rows
    that pass every filter. Filters, in order: non-empty problem; solution
    contains "\\boxed{" with a non-empty brace-matched answer; answer at most
    MAX_ANSWER_CHARS characters (sanity cap against degenerate labels).
    """
    funnel = {
        "total": 0,
        "problem_nonempty": 0,
        "has_boxed": 0,
        "answer_extractable": 0,
        "answer_len_ok": 0,
    }
    eligible: list[tuple[int, str]] = []
    slim = ds.select_columns(["problem", "solution"])
    n_batches = math.ceil(len(slim) / _FILTER_BATCH)
    idx = 0
    for batch in tqdm(
        slim.iter(batch_size=_FILTER_BATCH), total=n_batches, desc="filter", unit="batch"
    ):
        for problem, solution in zip(batch["problem"], batch["solution"]):
            row_idx = idx
            idx += 1
            funnel["total"] += 1
            if not (problem or "").strip():
                continue
            funnel["problem_nonempty"] += 1
            if BOXED_KEY not in (solution or ""):
                continue
            funnel["has_boxed"] += 1
            answer = (extract_last_boxed(solution) or "").strip()
            if not answer:
                continue
            funnel["answer_extractable"] += 1
            if len(answer) > MAX_ANSWER_CHARS:
                continue
            funnel["answer_len_ok"] += 1
            eligible.append((row_idx, answer))
    return eligible, funnel


def sample_eligible(
    eligible: list[tuple[int, str]], n: int, seed: int
) -> list[tuple[int, str]]:
    """Deterministic sample of exactly ``n`` rows: shuffle with
    ``random.Random(seed)`` and take the first ``n`` (stable across runs)."""
    if len(eligible) < n:
        raise SystemExit(
            f"only {len(eligible)} eligible rows after filtering; need n={n}"
        )
    pool = list(eligible)
    random.Random(seed).shuffle(pool)
    return pool[:n]


def build_rows(
    ds: Dataset,
    sampled: list[tuple[int, str]],
    tokenizer: PreTrainedTokenizerBase,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row_idx, answer in tqdm(sampled, desc="template", unit="row"):
        record = ds[row_idx]
        rows.append(
            {
                "input": render_prompt(tokenizer, record["problem"]),
                "label": answer,
                "metadata": {
                    "qid": f"numina-{row_idx}",
                    "source": record.get("source"),
                },
            }
        )
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def print_preview(rows: list[dict[str, Any]], k: int = 3, width: int = 160) -> None:
    print(f"--- preview (first {k} rows, input truncated to {width} chars) ---")
    for row in rows[:k]:
        meta = row["metadata"]
        snippet = row["input"][:width].replace("\n", "\\n")
        print(f"[{meta['qid']} | source={meta['source']}] label={row['label']!r}")
        print(f"  input[:{width}]: {snippet}...")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--n", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-path", default=DEFAULT_MODEL)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"loading {DATASET_NAME} split={SPLIT} (HF caches from env)")
    ds = load_dataset(DATASET_NAME, split=SPLIT)
    print(f"loaded {len(ds)} rows")

    print(f"loading tokenizer from {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    eligible, funnel = build_funnel(ds)
    print("--- filter funnel ---")
    for stage, count in funnel.items():
        print(f"  {stage}: {count}")

    sampled = sample_eligible(eligible, args.n, args.seed)
    rows = build_rows(ds, sampled, tokenizer)

    out_path = args.output_dir / OUTPUT_NAME
    write_jsonl(out_path, rows)
    print(f"wrote {len(rows)} rows -> {out_path}")

    meta = {
        "dataset": DATASET_NAME,
        "split": SPLIT,
        "dataset_fingerprint": getattr(ds, "_fingerprint", None),
        "n": args.n,
        "seed": args.seed,
        "model_path": args.model_path,
        "max_answer_chars": MAX_ANSWER_CHARS,
        "funnel": funnel,
        "n_eligible": len(eligible),
        "output": str(out_path),
    }
    meta_path = args.output_dir / META_NAME
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"wrote meta -> {meta_path}")

    print_preview(rows)


if __name__ == "__main__":
    main()
