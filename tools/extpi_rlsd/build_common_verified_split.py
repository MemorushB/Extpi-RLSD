#!/usr/bin/env python3
"""Build common verified train/dev splits without uplift filtering."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from tools.extpi_rlsd.common import read_jsonl, write_json, write_jsonl
from verl.trainer.extpi_rlsd.prompt_assembly import DEFAULT_PI_TRACE_FIELD


def _verified_field(trace_field: str) -> str:
    return trace_field.replace("_trace", "_verified") if trace_field.endswith("_trace") else f"{trace_field}_verified"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_jsonl", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--train_size", type=int, default=3200)
    parser.add_argument("--dev_size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pi_trace_field", default=DEFAULT_PI_TRACE_FIELD)
    args = parser.parse_args()

    pi_verified_field = _verified_field(args.pi_trace_field)
    rows = [
        row
        for row in read_jsonl(args.input_jsonl)
        if bool(row.get(pi_verified_field, False)) and row.get(args.pi_trace_field)
    ]
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    train = rows[: args.train_size]
    dev = rows[args.train_size : args.train_size + args.dev_size]
    output_dir = Path(args.output_dir)
    write_jsonl(output_dir / "confirmation_train.jsonl", train)
    write_jsonl(output_dir / "confirmation_dev.jsonl", dev)
    write_json(
        output_dir / "split_manifest.json",
        {"seed": args.seed, "train": len(train), "dev": len(dev), "pi_trace_field": args.pi_trace_field},
    )


if __name__ == "__main__":
    main()
