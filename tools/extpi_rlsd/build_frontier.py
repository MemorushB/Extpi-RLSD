#!/usr/bin/env python3
"""Build smoke, frontier, matched-dev, and confirmation splits."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from tools.extpi_rlsd.common import read_jsonl, write_json, write_jsonl
from verl.trainer.extpi_rlsd.prompt_assembly import DEFAULT_PI_TRACE_FIELD


def _verified_field(trace_field: str) -> str:
    return trace_field.replace("_trace", "_verified") if trace_field.endswith("_trace") else f"{trace_field}_verified"


def eligible_frontier(row: dict, relaxed: bool = False, pi_trace_field: str = DEFAULT_PI_TRACE_FIELD) -> bool:
    p_plain = float(row.get("p_plain", row.get("pS", 0.0)) or 0.0)
    p_pi = float(row.get("p_PI", row.get("pPI", row.get("pTPI", 0.0))) or 0.0)
    uplift = p_pi - p_plain
    verified = bool(row.get(_verified_field(pi_trace_field), row.get("pi_trace_verified", False)))
    has_trace = bool(row.get(pi_trace_field))
    truncated = bool(row.get("student_completion_truncated", False))
    if not verified or not has_trace or truncated:
        return False
    if relaxed:
        return 0.1 <= p_plain <= 0.75 and uplift > 0
    return p_plain in {0.25, 0.5, 0.75} and p_pi >= 0.75 and uplift >= 0.25


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_jsonl", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke", type=int, default=64)
    parser.add_argument("--mvp_train", type=int, default=800)
    parser.add_argument("--matched_dev", type=int, default=512)
    parser.add_argument("--confirmation_train", type=int, default=3200)
    parser.add_argument("--confirmation_dev", type=int, default=512)
    parser.add_argument("--pi_trace_field", default=DEFAULT_PI_TRACE_FIELD)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    rows = read_jsonl(args.input_jsonl)
    strict = [row for row in rows if eligible_frontier(row, relaxed=False, pi_trace_field=args.pi_trace_field)]
    relaxed = [row for row in rows if eligible_frontier(row, relaxed=True, pi_trace_field=args.pi_trace_field)]
    frontier = strict if len(strict) >= args.mvp_train + args.matched_dev else relaxed
    rng.shuffle(frontier)

    smoke = frontier[: args.smoke]
    mvp_train = frontier[: args.mvp_train]
    matched_dev = frontier[args.mvp_train : args.mvp_train + args.matched_dev]
    used_ids = {row["id"] for row in mvp_train + matched_dev}
    confirmation_pool = [
        row
        for row in rows
        if row.get("id") not in used_ids
        and bool(row.get(_verified_field(args.pi_trace_field), row.get("pi_trace_verified", False)))
        and row.get(args.pi_trace_field)
        and not bool(row.get("student_completion_truncated", False))
    ]
    rng.shuffle(confirmation_pool)
    confirmation_train = confirmation_pool[: args.confirmation_train]
    confirmation_dev = confirmation_pool[args.confirmation_train : args.confirmation_train + args.confirmation_dev]

    output_dir = Path(args.output_dir)
    write_jsonl(output_dir / "smoke.jsonl", smoke)
    write_jsonl(output_dir / "mvp_train.jsonl", mvp_train)
    write_jsonl(output_dir / "matched_dev.jsonl", matched_dev)
    write_jsonl(output_dir / "confirmation_train.jsonl", confirmation_train)
    write_jsonl(output_dir / "confirmation_dev.jsonl", confirmation_dev)
    write_json(
        output_dir / "split_manifest.json",
        {
            "seed": args.seed,
            "strict_frontier_count": len(strict),
            "relaxed_frontier_count": len(relaxed),
            "frontier_rule": "strict" if frontier is strict else "relaxed",
            "pi_trace_field": args.pi_trace_field,
            "counts": {
                "smoke": len(smoke),
                "mvp_train": len(mvp_train),
                "matched_dev": len(matched_dev),
                "confirmation_train": len(confirmation_train),
                "confirmation_dev": len(confirmation_dev),
            },
        },
    )


if __name__ == "__main__":
    main()
