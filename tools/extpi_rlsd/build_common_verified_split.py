#!/usr/bin/env python3
"""Build common verified train/dev splits without uplift filtering."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from tools.extpi_rlsd.common import read_jsonl, write_json, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_jsonl", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--train_size", type=int, default=3200)
    parser.add_argument("--dev_size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = [
        row
        for row in read_jsonl(args.input_jsonl)
        if bool(row.get("qwen8b_pi_verified", False)) and row.get("qwen8b_pi_trace")
    ]
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    train = rows[: args.train_size]
    dev = rows[args.train_size : args.train_size + args.dev_size]
    output_dir = Path(args.output_dir)
    write_jsonl(output_dir / "confirmation_train.jsonl", train)
    write_jsonl(output_dir / "confirmation_dev.jsonl", dev)
    write_json(output_dir / "split_manifest.json", {"seed": args.seed, "train": len(train), "dev": len(dev)})


if __name__ == "__main__":
    main()
