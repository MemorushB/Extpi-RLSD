#!/usr/bin/env python3
"""Create a shuffled-PI control view of an ExtPI-RLSD JSONL file."""

from __future__ import annotations

import argparse
import random

from tools.extpi_rlsd.common import read_jsonl, write_jsonl
from verl.trainer.extpi_rlsd.prompt_assembly import DEFAULT_PI_TRACE_FIELD


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_jsonl", required=True)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pi_trace_field", default=DEFAULT_PI_TRACE_FIELD)
    args = parser.parse_args()

    rows = read_jsonl(args.input_jsonl)
    traces = [row.get(args.pi_trace_field) for row in rows]
    rng = random.Random(args.seed)
    shuffled = traces[:]
    rng.shuffle(shuffled)
    if len(shuffled) > 1 and any(a == b for a, b in zip(traces, shuffled, strict=True)):
        shuffled = shuffled[1:] + shuffled[:1]
    output = []
    for row, trace in zip(rows, shuffled, strict=True):
        new_row = dict(row)
        new_row[args.pi_trace_field] = trace
        new_row.setdefault("metadata", {})
        new_row["metadata"]["shuffled_pi_control"] = {
            "seed": args.seed,
            "source": args.input_jsonl,
            "pi_trace_field": args.pi_trace_field,
        }
        output.append(new_row)
    write_jsonl(args.output_jsonl, output)


if __name__ == "__main__":
    main()
