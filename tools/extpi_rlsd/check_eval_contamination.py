#!/usr/bin/env python3
"""Check normalized overlap between training and evaluation JSONL files."""

from __future__ import annotations

import argparse

from tools.extpi_rlsd.common import compact_text, normalize_problem, read_jsonl, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_jsonl", required=True)
    parser.add_argument("--eval_jsonl", action="append", required=True)
    parser.add_argument("--output_jsonl", required=True)
    args = parser.parse_args()

    eval_norm = set()
    eval_compact = set()
    for path in args.eval_jsonl:
        for row in read_jsonl(path):
            problem = str(row.get("problem", ""))
            eval_norm.add(normalize_problem(problem))
            eval_compact.add(compact_text(problem))
    hits = []
    for row in read_jsonl(args.train_jsonl):
        problem = str(row.get("problem", ""))
        if normalize_problem(problem) in eval_norm or compact_text(problem) in eval_compact:
            hits.append(row)
    write_jsonl(args.output_jsonl, hits)


if __name__ == "__main__":
    main()
