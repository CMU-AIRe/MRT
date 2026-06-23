"""Aggregate vllm_eval.py runs into a cross-model comparison table (paper_002_mrt).

Reads each run's OUT/<bench>/summary.json (produced by eval/vllm_eval.py under the
A3 decoding protocol — temp 0.6 / top_p 0.95 / 16K budget — and the A15 grading
convention; see ../../implementation_plan.md) and emits two markdown tables:

  1. per-benchmark pass@1 (in %), 5-benchmark Avg, and delta vs the `base` run;
  2. our deltas vs the paper's deltas (paper Table: base 44.9, GRPO 46.0 (+1.1),
     MRT 47.1 (+2.2) — overridable with --paper).

pass1 in summary.json is a fraction in [0, 1]; all table values are percentages.
Missing benchmarks render as "-"; the Avg (and deltas) require all 5 benchmarks.

Usage:
  python aggregate.py --runs base=OUT1 grpo=OUT2 mrt=OUT3 \\
      [--paper base=44.9 grpo=46.0 mrt=47.1] [--output aggregate.md]
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path

BENCH_ORDER = ["aime24", "aime25", "amc23", "minerva", "math500"]
BENCH_LABEL = {
    "aime24": "AIME24",
    "aime25": "AIME25",
    "amc23": "AMC23",
    "minerva": "Minerva",
    "math500": "MATH500",
}


def parse_pairs(items: list[str], cast, flag: str) -> dict:
    """Parse name=value CLI pairs into an ordered dict."""
    out: dict = {}
    for item in items:
        if "=" not in item:
            sys.exit(f"FATAL: {flag} entries must be name=value, got: {item!r}")
        name, value = item.split("=", 1)
        try:
            out[name] = cast(value)
        except (TypeError, ValueError) as exc:
            sys.exit(f"FATAL: bad {flag} value {item!r}: {exc}")
    return out


def load_run(run_dir: Path) -> dict[str, float]:
    """Read pass1 (fraction) from every <bench>/summary.json under run_dir."""
    scores: dict[str, float] = {}
    if not run_dir.is_dir():
        print(f"WARNING: run dir not found: {run_dir}")
        return scores
    for summary_path in sorted(run_dir.glob("*/summary.json")):
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            scores[summary["benchmark"]] = float(summary["pass1"])
        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            print(f"WARNING: skipping unreadable {summary_path}: {exc}")
    return scores


def five_bench_avg(scores: dict[str, float]) -> float | None:
    """Mean over the 5 canonical benchmarks; None unless all 5 are present."""
    if all(b in scores for b in BENCH_ORDER):
        return sum(scores[b] for b in BENCH_ORDER) / len(BENCH_ORDER)
    return None


def fmt_pct(fraction: float | None) -> str:
    return "-" if fraction is None else f"{fraction * 100:.2f}"


def fmt_delta(points: float | None) -> str:
    return "-" if points is None else f"{points:+.2f}"


def build_results_table(runs: dict[str, dict[str, float]]) -> list[str]:
    """Table 1: rows = models, cols = 5 benchmarks + Avg + delta vs base (in points)."""
    base_avg = five_bench_avg(runs["base"]) if "base" in runs else None
    header = "| Model | " + " | ".join(BENCH_LABEL[b] for b in BENCH_ORDER) + \
        " | Avg | Δ vs base |"
    sep = "|---" * (len(BENCH_ORDER) + 3) + "|"
    lines = [header, sep]
    for name, scores in runs.items():
        avg = five_bench_avg(scores)
        delta = None
        if avg is not None and base_avg is not None:
            delta = (avg - base_avg) * 100
        cells = [fmt_pct(scores.get(b)) for b in BENCH_ORDER]
        lines.append(
            f"| {name} | " + " | ".join(cells) +
            f" | {fmt_pct(avg)} | {fmt_delta(delta)} |"
        )
    return lines


def build_paper_table(
    runs: dict[str, dict[str, float]], paper: dict[str, float]
) -> list[str]:
    """Table 2: our 5-bench-avg deltas vs the paper's deltas (both vs `base`)."""
    base_avg = five_bench_avg(runs["base"]) if "base" in runs else None
    paper_base = paper.get("base")
    lines = [
        "| Model | Ours Avg | Ours Δ vs base | Paper Avg | Paper Δ vs base | Gap (ours − paper Δ) |",
        "|---|---|---|---|---|---|",
    ]
    for name, scores in runs.items():
        avg = five_bench_avg(scores)
        ours_delta = (
            (avg - base_avg) * 100 if avg is not None and base_avg is not None else None
        )
        paper_avg = paper.get(name)
        paper_delta = (
            paper_avg - paper_base
            if paper_avg is not None and paper_base is not None
            else None
        )
        gap = (
            ours_delta - paper_delta
            if ours_delta is not None and paper_delta is not None
            else None
        )
        paper_avg_str = "-" if paper_avg is None else f"{paper_avg:.2f}"
        lines.append(
            f"| {name} | {fmt_pct(avg)} | {fmt_delta(ours_delta)} | "
            f"{paper_avg_str} | {fmt_delta(paper_delta)} | {fmt_delta(gap)} |"
        )
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", nargs="+", required=True, metavar="NAME=OUT_DIR",
                        help="e.g. base=out/base grpo=out/grpo mrt=out/mrt")
    parser.add_argument("--paper", nargs="*", metavar="NAME=PCT",
                        default=["base=44.9", "grpo=46.0", "mrt=47.1"],
                        help="paper 5-bench averages in percent (default: MRT paper)")
    parser.add_argument("--output", default="aggregate.md",
                        help="markdown output path (default: ./aggregate.md)")
    args = parser.parse_args()

    run_dirs = parse_pairs(args.runs, Path, "--runs")
    paper = parse_pairs(args.paper, float, "--paper")
    runs = {name: load_run(path) for name, path in run_dirs.items()}
    if "base" not in runs:
        print("WARNING: no run named 'base' — deltas vs base will be '-'")

    sections = [
        "# Eval aggregate (paper_002_mrt)",
        "",
        f"Generated {datetime.date.today().isoformat()} by eval/aggregate.py. "
        "Values are pass@1 in %, A3 decoding (temp 0.6 / top_p 0.95 / 16K), "
        "A15 grading.",
        "",
        "Runs: " + ", ".join(f"{n}={p}" for n, p in run_dirs.items()),
        "",
        "## Per-benchmark pass@1",
        "",
        *build_results_table(runs),
        "",
        "## Deltas vs paper",
        "",
        *build_paper_table(runs, paper),
        "",
    ]
    report = "\n".join(sections)
    print(report)
    Path(args.output).write_text(report, encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
