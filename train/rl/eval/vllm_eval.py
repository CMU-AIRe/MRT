"""Standalone vLLM pass@1 evaluation for paper_002_mrt (MRT reproduction).

Evaluates a DeepSeek-R1-Distill-Qwen-1.5B-family checkpoint on the 5 math
benchmarks built by eval/prep_benchmarks.py and reports pass@1 (mean over
n samples per problem, then mean over problems) plus the 5-benchmark average.

Pinned assumptions (see ../../implementation_plan.md, "Assumptions" table):
  - A3 (decoding): temperature 0.6, top_p 0.95, n=64 samples/problem,
    16,384 max completion tokens — DeepScaleR protocol / R1-Distill model card.
  - A15 (grading): must match the training-reward semantics (miles' deepscaler
    grader, see ../mrt_plugin/reward_math.py): grade ONLY the text after the
    LAST "</think>"; a response with no "</think>" is a truncated/unclosed
    thought and scores 0. From the graded segment extract the LAST \\boxed{...}
    by brace matching (no box -> 0), then check math_verify equivalence OR
    normalized-string equality against the gold answer.
  - A18 (prompting): chat template, no system prompt, generation starts inside
    "<think>\\n" — reuses mrt_plugin.templates.render_prompt verbatim.

Resumable: per benchmark OUT/<bench>/samples.jsonl holds one row per
(problem, sample). On restart, problems that already have n_samples rows are
skipped; incomplete problems are fully regenerated (their old rows replaced).

Usage:
  python vllm_eval.py --model-path PATH --benchmarks-dir BENCH_DIR --output-dir OUT \\
      --benchmarks aime24,aime25,amc23,minerva,math500 \\
      [--n-samples 64] [--temperature 0.6] [--top-p 0.95] [--max-tokens 16384] \\
      [--tp 1] [--seed 1234]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

# Make ../mrt_plugin importable so prompting is byte-identical to training.
_CODE_DIR = Path(__file__).resolve().parent.parent
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

from mrt_plugin.templates import render_prompt  # noqa: E402

DEFAULT_BENCHMARKS = "aime24,aime25,amc23,minerva,math500"


# ----------------------------------------------------------------------------
# Grading (A15) — mirrors the training reward in mrt_plugin/reward_math.py /
# miles' deepscaler grader: split on the LAST </think>, no tag => 0.
# ----------------------------------------------------------------------------

def extract_last_boxed(text: str) -> str | None:
    """Return the contents of the last well-formed \\boxed{...} via brace matching."""
    result = None
    i = 0
    while True:
        idx = text.find("\\boxed", i)
        if idx == -1:
            break
        j = idx + len("\\boxed")
        while j < len(text) and text[j] in " \t":
            j += 1
        if j >= len(text) or text[j] != "{":
            i = idx + 1
            continue
        depth = 0
        k = j
        while k < len(text):
            if text[k] == "{":
                depth += 1
            elif text[k] == "}":
                depth -= 1
                if depth == 0:
                    break
            k += 1
        if depth != 0:  # unbalanced to end of text (truncated box) — keep prior match
            break
        result = text[j + 1 : k]
        i = k + 1
    return result


def normalize_answer(ans: str) -> str:
    """Mild LaTeX/whitespace normalization for the string-equality fallback."""
    s = str(ans).strip()
    for token in ("\\left", "\\right", "\\!", "\\,", "\\;", "$", "\\$"):
        s = s.replace(token, "")
    s = "".join(s.split())  # remove all whitespace
    s = s.rstrip(".")
    if s.startswith("\\text{") and s.endswith("}"):
        s = s[len("\\text{") : -1].strip()
    return s


def math_verify_equal(gold: str, pred: str) -> bool:
    """math_verify equivalence, hardened: any internal error counts as not-equal."""
    try:
        from math_verify import parse, verify

        return bool(verify(parse(f"${gold}$"), parse(f"${pred}$")))
    except Exception:
        return False


def _deepscaler_grade(response_text: str, gold_answer: str) -> int:
    """PRIMARY verdict: the exact training-reward grader (miles' deepscaler RM),
    i.e. the grader of the protocol the paper cites. Imported lazily from the
    pinned miles clone (sibling of this file's parent dir)."""
    global _DS_GRADER
    if _DS_GRADER is None:
        miles_dir = str(Path(__file__).resolve().parent.parent / "miles")
        if miles_dir not in sys.path:
            sys.path.insert(0, miles_dir)
        from miles.rollout.rm_hub.deepscaler import get_deepscaler_rule_based_reward
        _DS_GRADER = get_deepscaler_rule_based_reward
    try:
        return int(_DS_GRADER(response_text, gold_answer))
    except Exception:
        return 0


_DS_GRADER = None


def grade_response(response_text: str, gold_answer: str) -> tuple[int, bool, str | None, int]:
    """Return (correct 0/1, closed_think, extracted_answer, correct_mv).

    `correct` (PRIMARY) = deepscaler grader on the raw completion — identical
    semantics to the training reward (A15) and to the protocol the paper cites;
    G1 calibration showed math-verify is ~8 pts more lenient on minerva (32.8 vs
    24.9; paper-side graders themselves disagree: MRT 19.8 vs DeepScaleR ~26).
    `correct_mv` (secondary, recorded) = math-verify-or-normalized-match on the
    extracted answer. The extracted boxed answer is persisted per row so grader
    changes can REGRADE old runs without regeneration.
    """
    closed = "</think>" in response_text
    if not closed:
        return 0, False, None, 0
    segment = response_text.split("</think>")[-1]
    extracted = extract_last_boxed(segment)
    strict = _deepscaler_grade(response_text, gold_answer)
    if extracted is None:
        return strict, True, None, 0
    gold = str(gold_answer)
    if "\\boxed" in gold:
        boxed_gold = extract_last_boxed(gold)
        if boxed_gold is not None:
            gold = boxed_gold
    correct_mv = math_verify_equal(gold, extracted) or (
        normalize_answer(gold) == normalize_answer(extracted)
    )
    return strict, True, extracted, int(correct_mv)


# ----------------------------------------------------------------------------
# IO helpers
# ----------------------------------------------------------------------------

def load_benchmark(bench_dir: Path, name: str) -> list[dict]:
    path = bench_dir / f"{name}.jsonl"
    if not path.exists():
        sys.exit(f"FATAL: benchmark file not found: {path} (run prep_benchmarks.py first)")
    problems = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                problems.append(json.loads(line))
    return problems


def load_existing_rows(samples_path: Path) -> dict[str, list[dict]]:
    """Group prior samples.jsonl rows by problem id (tolerates a torn last line)."""
    rows_by_id: dict[str, list[dict]] = defaultdict(list)
    if not samples_path.exists():
        return rows_by_id
    with samples_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue  # torn write from a killed run; that problem will be redone
            rows_by_id[row["id"]].append(row)
    return rows_by_id


def load_all_rows(out_bench: Path) -> dict[str, list[dict]]:
    """Merge rows from samples.jsonl and every samples_shard*.jsonl (disjoint
    problem ids per shard, so a plain merge is safe; resume works across
    sharded and unsharded runs)."""
    merged: dict[str, list[dict]] = defaultdict(list)
    for path in sorted(out_bench.glob("samples*.jsonl")):
        for pid, rows in load_existing_rows(path).items():
            merged[pid].extend(rows)
    return merged


def atomic_write_jsonl(rows: list[dict], path: Path) -> None:
    tmp = path.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    os.replace(tmp, path)


def summarize(bench: str, problems: list[dict], rows: list[dict], args) -> dict:
    """Per-benchmark summary: pass@1 = mean over problems of mean-correct over samples."""
    by_id: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_id[row["id"]].append(row)
    per_problem = [
        sum(r["correct"] for r in by_id[p["id"]]) / len(by_id[p["id"]])
        for p in problems
        if by_id.get(p["id"])
    ]
    n_graded = len(per_problem)
    if n_graded != len(problems):
        print(f"WARNING: {bench}: only {n_graded}/{len(problems)} problems have samples")
    return {
        "benchmark": bench,
        "n_problems": n_graded,
        "n_samples": args.n_samples,
        "pass1": sum(per_problem) / n_graded if n_graded else 0.0,
        "avg_completion_tokens": (
            sum(r["finish_len"] for r in rows) / len(rows) if rows else 0.0
        ),
        "frac_unclosed_think": (
            sum(1 for r in rows if not r["closed_think"]) / len(rows) if rows else 0.0
        ),
        "model_path": args.model_path,
        "decoding": {
            "n": args.n_samples,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_tokens": args.max_tokens,
            "seed": args.seed,
        },
    }


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--benchmarks-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--benchmarks", default=DEFAULT_BENCHMARKS,
                        help="comma-separated benchmark names")
    parser.add_argument("--n-samples", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--tp", type=int, default=1, help="tensor parallel size")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--num-shards", type=int, default=1,
                        help="problem-level data parallelism: run N processes, one per GPU")
    parser.add_argument("--shard-id", type=int, default=0,
                        help="which problem shard this process handles (0-based)")
    parser.add_argument("--summarize-only", action="store_true",
                        help="no generation: merge samples*.jsonl shards and write summary.json")
    return parser.parse_args()


def run_benchmark(bench: str, args, tokenizer, get_llm, sampling_params_cls) -> dict:
    """Evaluate one benchmark (resumable); returns its summary dict."""
    from tqdm import tqdm

    bench_dir = Path(args.benchmarks_dir)
    out_bench = Path(args.output_dir) / bench
    out_bench.mkdir(parents=True, exist_ok=True)

    all_problems = load_benchmark(bench_dir, bench)

    if args.summarize_only:
        merged = load_all_rows(out_bench)
        all_rows = [
            row for p in all_problems for row in merged.get(p["id"], [])[: args.n_samples]
        ]
        n_complete = sum(1 for p in all_problems if len(merged.get(p["id"], [])) >= args.n_samples)
        if n_complete < len(all_problems):
            print(f"[{bench}] WARNING: only {n_complete}/{len(all_problems)} problems "
                  "have full sample counts — summary is partial")
        summary = summarize(bench, all_problems, all_rows, args)
        (out_bench / "summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        print(f"[{bench}] pass@1 = {summary['pass1']:.4f} (summarize-only, "
              f"{n_complete}/{len(all_problems)} complete)")
        return summary

    sharded = args.num_shards > 1
    problems = [p for i, p in enumerate(all_problems) if i % args.num_shards == args.shard_id]
    samples_path = (
        out_bench / (f"samples_shard{args.shard_id}.jsonl" if sharded else "samples.jsonl")
    )

    valid_ids = {p["id"] for p in problems}
    existing = load_all_rows(out_bench)
    complete = {
        pid for pid, rows in existing.items()
        if pid in valid_ids and len(rows) >= args.n_samples
    }
    kept_rows = [row for pid in sorted(complete) for row in existing[pid][: args.n_samples]]
    todo = [p for p in problems if p["id"] not in complete]
    print(f"[{bench}] shard {args.shard_id}/{args.num_shards}: {len(problems)} problems; "
          f"{len(complete)} already complete, {len(todo)} to generate")

    new_rows: list[dict] = []
    if todo:
        # Rendered prompts contain the literal BOS from the chat template; pass
        # token ids (add_special_tokens=False) so vLLM doesn't prepend a second
        # BOS (same convention as prefix_gen.py and miles' rollout encoding).
        prompts = [
            {
                "prompt_token_ids": tokenizer.encode(
                    render_prompt(tokenizer, p["problem"]), add_special_tokens=False
                )
            }
            for p in todo
        ]
        sampling = sampling_params_cls(
            n=args.n_samples,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            seed=args.seed,
        )
        outputs = get_llm().generate(prompts, sampling)
        for problem, output in tqdm(
            zip(todo, outputs), total=len(todo), desc=f"grade {bench}"
        ):
            for sample_idx, completion in enumerate(output.outputs):
                correct, closed, extracted, correct_mv = grade_response(
                    completion.text, problem["answer"]
                )
                new_rows.append({
                    "id": problem["id"],
                    "sample_idx": sample_idx,
                    "correct": correct,
                    "correct_mv": correct_mv,
                    "finish_len": len(completion.token_ids),
                    "closed_think": closed,
                    "extracted": extracted,
                })

    all_rows = kept_rows + new_rows
    atomic_write_jsonl(all_rows, samples_path)
    summary = summarize(bench, problems, all_rows, args)
    if sharded:
        # partial view only; run --summarize-only after all shards finish
        print(f"[{bench}] shard {args.shard_id}: partial pass@1 = {summary['pass1']:.4f} "
              f"over {len(problems)} problems (no summary.json written)")
        return summary
    (out_bench / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[{bench}] pass@1 = {summary['pass1']:.4f}  "
          f"(avg_tokens {summary['avg_completion_tokens']:.0f}, "
          f"unclosed_think {summary['frac_unclosed_think']:.3f})")
    return summary


def main() -> None:
    args = parse_args()
    benchmarks = [b.strip() for b in args.benchmarks.split(",") if b.strip()]

    # Heavy imports kept inside main so grading helpers are importable/testable
    # without a GPU environment.
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    llm_holder: list = []

    def get_llm():
        if not llm_holder:  # instantiate only if some benchmark needs generation
            llm_holder.append(
                LLM(model=args.model_path, tensor_parallel_size=args.tp, seed=args.seed)
            )
        return llm_holder[0]

    summaries = {}
    for bench in benchmarks:
        summaries[bench] = run_benchmark(bench, args, tokenizer, get_llm, SamplingParams)

    print("\n=== pass@1 (mean over problems of mean-correct over "
          f"{args.n_samples} samples) ===")
    print(f"{'benchmark':<12} {'pass@1':>8}")
    for bench in benchmarks:
        print(f"{bench:<12} {summaries[bench]['pass1'] * 100:>7.2f}%")
    avg = sum(s["pass1"] for s in summaries.values()) / len(summaries)
    print(f"{'AVG':<12} {avg * 100:>7.2f}%   (over {len(benchmarks)} benchmarks)")


if __name__ == "__main__":
    main()
