"""Build the 5 normalized eval benchmarks for paper_002_mrt (MRT reproduction).

Produces one JSONL file per benchmark in --output-dir, each line:
    {"id": str, "problem": str, "answer": str}

Benchmarks and primary sources (DeepScaleR's own eval sets where possible):
  - aime24  (30):  deepscaler repo test/aime.json    (fallback: HF Maxwell-Jia/AIME_2024)
  - amc23   (40):  deepscaler repo test/amc.json     (fallback: HF math-ai/amc23)
  - math500 (500): deepscaler repo test/math.json    (fallback: HF HuggingFaceH4/MATH-500)
  - minerva (272): deepscaler repo test/minerva.json (fallback: HF math-ai/minervamath)
  - aime25  (30):  HF opencompass/AIME2025, configs AIME2025-I + AIME2025-II

The deepscaler repo moved; we try agentica-project/deepscaler@main first, then the
`deepscaler` branch of agentica-project/rllm, then the HF fallback. Expected counts
are HARD-ASSERTED (30/40/500/272/30) so a silently changed upstream fails loudly.

Context (see ../../implementation_plan.md): these sets feed eval/vllm_eval.py, which
implements the A3 decoding protocol (temp 0.6 / top_p 0.95, 16K budget) and the
A15 grading convention (last \\boxed{} + math-verify).
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path

EXPECTED_COUNTS = {
    "aime24": 30,
    "amc23": 40,
    "math500": 500,
    "minerva": 272,
    "aime25": 30,
}

# Primary sources: raw JSON files from the DeepScaleR project (repo moved once).
_DEEPSCALER_FILE = {
    "aime24": "aime.json",
    "amc23": "amc.json",
    "math500": "math.json",
    "minerva": "minerva.json",
}
_GITHUB_BASES = [
    "https://raw.githubusercontent.com/agentica-project/deepscaler/main/deepscaler/data/test",
    "https://raw.githubusercontent.com/agentica-project/rllm/deepscaler/deepscaler/data/test",
]

# HF fallbacks if both GitHub locations fail: (dataset_id, config or None).
_HF_FALLBACK = {
    "aime24": ("Maxwell-Jia/AIME_2024", None),
    "amc23": ("math-ai/amc23", None),
    "math500": ("HuggingFaceH4/MATH-500", None),
    "minerva": ("math-ai/minervamath", None),
}

_PROBLEM_KEYS = ["problem", "Problem", "question", "Question", "prompt"]
_ANSWER_KEYS = ["answer", "Answer", "final_answer", "expected_answer", "gt_answer"]


def _pick_key(record: dict, candidates: list[str], what: str) -> str:
    for key in candidates:
        if key in record and record[key] is not None:
            return key
    raise KeyError(f"no {what} key among {candidates}; record keys = {sorted(record)}")


def _clean_answer(raw) -> str:
    """Unwrap list-typed answers (minerva in the deepscaler JSONs stores
    ['9.6\\n']-style python-list strings / lists) and strip whitespace."""
    import ast

    val = raw
    if isinstance(val, str):
        s = val.strip()
        # Only unwrap reprs of LISTS OF STRINGS (the minerva pattern "['9.6\\n']").
        # Bare brackets/parens may be genuine math answers ("[0, 1)", "(3, 4)").
        if s.startswith(("['", '["')) and s.endswith("]"):
            try:
                parsed = ast.literal_eval(s)
                if (
                    isinstance(parsed, list)
                    and parsed
                    and all(isinstance(x, str) for x in parsed)
                ):
                    val = parsed[0]
            except (ValueError, SyntaxError):
                pass
    if isinstance(val, (list, tuple)):
        val = val[0] if val else ""
    return str(val).strip()


def normalize_records(records: list[dict], bench: str) -> list[dict]:
    """Map source records (varying schemas) to {"id", "problem", "answer"} rows."""
    if not records:
        raise ValueError(f"{bench}: source returned zero records")
    problem_key = _pick_key(records[0], _PROBLEM_KEYS, "problem")
    answer_key = _pick_key(records[0], _ANSWER_KEYS, "answer")
    rows = []
    for i, rec in enumerate(records):
        rows.append(
            {
                "id": f"{bench}-{i:04d}",
                "problem": str(rec[problem_key]).strip(),
                "answer": _clean_answer(rec[answer_key]),
            }
        )
    return rows


def fetch_github_json(url: str) -> list[dict] | None:
    """Fetch a raw JSON list from GitHub; None (with a printed reason) on any failure."""
    import requests

    try:
        resp = requests.get(url, timeout=60)
    except requests.RequestException as exc:
        print(f"  [github] {url} -> network error: {exc}")
        return None
    if resp.status_code != 200:
        print(f"  [github] {url} -> HTTP {resp.status_code}")
        return None
    try:
        data = resp.json()
    except ValueError as exc:
        print(f"  [github] {url} -> invalid JSON: {exc}")
        return None
    if not isinstance(data, list):
        print(f"  [github] {url} -> unexpected top-level type {type(data).__name__}")
        return None
    return data


def _pick_split(dataset_dict) -> str:
    """Prefer 'test', then 'train', else the first available split."""
    splits = list(dataset_dict.keys())
    for preferred in ("test", "train"):
        if preferred in splits:
            return preferred
    return splits[0]


def load_hf_records(dataset_id: str, config: str | None) -> tuple[list[dict], str]:
    """Load an HF dataset and return (records, source description)."""
    from datasets import load_dataset

    dsd = load_dataset(dataset_id, config) if config else load_dataset(dataset_id)
    split = _pick_split(dsd)
    records = [dict(row) for row in dsd[split]]
    desc = f"HF {dataset_id}" + (f" config={config}" if config else "") + f" split={split}"
    return records, desc


def build_github_benchmark(bench: str) -> tuple[list[dict], str]:
    """aime24/amc23/math500/minerva: GitHub primaries, then HF fallback.

    A source whose problem count doesn't match EXPECTED_COUNTS is treated as a miss
    and the next source is tried (e.g. deepscaler's amc.json holds 83 problems
    spanning AMC 2022-23, while the paper's "AMC 2023" column is the standard
    40-problem set — math-ai/amc23)."""
    expected = EXPECTED_COUNTS[bench]
    fname = _DEEPSCALER_FILE[bench]
    for base in _GITHUB_BASES:
        url = f"{base}/{fname}"
        data = fetch_github_json(url)
        if data is not None:
            rows = normalize_records(data, bench)
            if len(rows) == expected:
                return rows, url
            print(
                f"  [{bench}] {url} has {len(rows)} problems, expected {expected}; "
                "trying next source"
            )
    dataset_id, config = _HF_FALLBACK[bench]
    print(f"  [{bench}] GitHub sources unusable; falling back to HF {dataset_id}")
    records, desc = load_hf_records(dataset_id, config)
    return normalize_records(records, bench), desc


def build_aime25() -> tuple[list[dict], str]:
    """aime25: opencompass/AIME2025, configs AIME2025-I + AIME2025-II concatenated."""
    from datasets import get_dataset_config_names

    dataset_id = "opencompass/AIME2025"
    available = get_dataset_config_names(dataset_id)
    wanted = [c for c in ("AIME2025-I", "AIME2025-II") if c in available]
    if len(wanted) != 2:
        raise RuntimeError(
            f"{dataset_id}: expected configs AIME2025-I/AIME2025-II, found {available}"
        )
    records: list[dict] = []
    descs = []
    for config in wanted:
        part, desc = load_hf_records(dataset_id, config)
        print(f"  [aime25] {desc}: {len(part)} records")
        records.extend(part)
        descs.append(desc)
    return normalize_records(records, "aime25"), " + ".join(descs)


def write_jsonl(rows: list[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def assert_count(bench: str, rows: list[dict]) -> None:
    expected = EXPECTED_COUNTS[bench]
    if len(rows) != expected:
        print(
            f"FATAL: {bench} has {len(rows)} problems, expected {expected}. "
            "Upstream source changed — inspect it before trusting any eval numbers.",
            file=sys.stderr,
        )
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-dir", required=True, help="directory for the JSONL files")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    readme_lines = [
        "# Eval benchmarks (paper_002_mrt)",
        "",
        f"Built by `eval/prep_benchmarks.py` on {datetime.date.today().isoformat()}.",
        "Schema per line: `{\"id\": str, \"problem\": str, \"answer\": str}`.",
        "",
        "| benchmark | count | source |",
        "|---|---|---|",
    ]

    for bench in ("aime24", "amc23", "math500", "minerva", "aime25"):
        print(f"== building {bench} ==")
        if bench == "aime25":
            rows, source = build_aime25()
        else:
            rows, source = build_github_benchmark(bench)
        assert_count(bench, rows)
        path = out_dir / f"{bench}.jsonl"
        write_jsonl(rows, path)
        print(f"  wrote {path} ({len(rows)} problems) from {source}")
        readme_lines.append(f"| {bench} | {len(rows)} | {source} |")

    readme = out_dir / "README.md"
    readme.write_text("\n".join(readme_lines) + "\n", encoding="utf-8")
    print(f"wrote {readme}")
    print("All 5 benchmarks built; counts verified (30/40/500/272/30).")


if __name__ == "__main__":
    main()
