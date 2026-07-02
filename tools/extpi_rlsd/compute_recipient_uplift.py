#!/usr/bin/env python3
"""Attach recipient PI-uplift statistics from plain and PI completion JSONL files."""

from __future__ import annotations

import argparse
from collections import defaultdict
from typing import Any

from recipes.extpi_rlsd.rewards.math_verify_reward import extract_boxed_answer, verify_answer
from tools.extpi_rlsd.common import read_jsonl, write_json, write_jsonl
from verl.trainer.extpi_rlsd.prompt_assembly import DEFAULT_PI_TRACE_FIELD

TEXT_KEYS = ("completion", "response", "solution_str", "text", "output")


def _verified_field(trace_field: str) -> str:
    return trace_field.replace("_trace", "_verified") if trace_field.endswith("_trace") else f"{trace_field}_verified"


def _sample_id(row: dict[str, Any]) -> str:
    for key in ("id", "problem_id", "sample_id"):
        if row.get(key) is not None:
            return str(row[key])
    extra = row.get("extra_info")
    if isinstance(extra, dict):
        for key in ("id", "problem_id", "sample_id"):
            if extra.get(key) is not None:
                return str(extra[key])
    raise ValueError(f"Completion row is missing an id field: keys={sorted(row)}")


def _completion_text(row: dict[str, Any]) -> str:
    for key in TEXT_KEYS:
        if row.get(key) is not None:
            return str(row[key])
    raise ValueError(f"Completion row is missing one of {TEXT_KEYS}: keys={sorted(row)}")


def _is_truncated(row: dict[str, Any]) -> bool:
    finish_reason = str(row.get("finish_reason", "")).lower()
    return bool(row.get("truncated", row.get("completion_truncated", False))) or finish_reason in {
        "length",
        "max_tokens",
    }


def _group_completions(path: str) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in read_jsonl(path):
        grouped[_sample_id(row)].append(row)
    return grouped


def _correct_count(samples: list[dict[str, Any]], gold_answer: str) -> tuple[int, int]:
    correct = 0
    truncated = 0
    for sample in samples:
        text = _completion_text(sample)
        correct += int(verify_answer(extract_boxed_answer(text), gold_answer))
        truncated += int(_is_truncated(sample))
    return correct, truncated


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_jsonl", required=True, help="Base ExtPI JSONL with id/gold_answer/pi trace fields.")
    parser.add_argument("--plain_jsonl", required=True, help="Problem-only recipient completions.")
    parser.add_argument("--pi_jsonl", required=True, help="PI-assisted recipient completions.")
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--expected_samples", type=int, default=4)
    parser.add_argument("--pi_trace_field", default=DEFAULT_PI_TRACE_FIELD)
    args = parser.parse_args()
    pi_verified_field = _verified_field(args.pi_trace_field)

    base_rows = read_jsonl(args.input_jsonl)
    plain = _group_completions(args.plain_jsonl)
    pi = _group_completions(args.pi_jsonl)

    output_rows = []
    stats = {
        "rows": 0,
        "with_plain": 0,
        "with_pi": 0,
        "with_uplift_positive": 0,
        "strict_frontier_eligible": 0,
        "relaxed_frontier_eligible": 0,
    }
    for row in base_rows:
        sample_id = str(row["id"])
        gold_answer = str(row["gold_answer"])
        plain_samples = plain.get(sample_id, [])
        pi_samples = pi.get(sample_id, [])
        new_row = dict(row)
        stats["rows"] += 1
        if plain_samples:
            stats["with_plain"] += 1
        if pi_samples:
            stats["with_pi"] += 1
        if len(plain_samples) != args.expected_samples or len(pi_samples) != args.expected_samples:
            new_row.setdefault("metadata", {})
            new_row["metadata"]["recipient_uplift_incomplete"] = {
                "plain_samples": len(plain_samples),
                "pi_samples": len(pi_samples),
                "expected_samples": args.expected_samples,
            }
            output_rows.append(new_row)
            continue

        correct_plain, truncated_plain = _correct_count(plain_samples, gold_answer)
        correct_pi, truncated_pi = _correct_count(pi_samples, gold_answer)
        p_plain = correct_plain / args.expected_samples
        p_pi = correct_pi / args.expected_samples
        uplift = p_pi - p_plain
        any_truncated = truncated_plain > 0 or truncated_pi > 0
        new_row.update(
            {
                "p_plain": p_plain,
                "p_PI": p_pi,
                "uplift": uplift,
                "correct_plain": correct_plain,
                "correct_PI": correct_pi,
                "student_completion_truncated": any_truncated,
            }
        )
        if uplift > 0:
            stats["with_uplift_positive"] += 1
        if (
            bool(new_row.get(pi_verified_field, False))
            and not any_truncated
            and p_plain in {0.25, 0.5, 0.75}
            and p_pi >= 0.75
            and uplift >= 0.25
        ):
            stats["strict_frontier_eligible"] += 1
        if (
            bool(new_row.get(pi_verified_field, False))
            and not any_truncated
            and 0.1 <= p_plain <= 0.75
            and uplift > 0
        ):
            stats["relaxed_frontier_eligible"] += 1
        output_rows.append(new_row)

    write_jsonl(args.output_jsonl, output_rows)
    if args.manifest:
        write_json(
            args.manifest,
            {
                "input_jsonl": args.input_jsonl,
                "plain_jsonl": args.plain_jsonl,
                "pi_jsonl": args.pi_jsonl,
                "output_jsonl": args.output_jsonl,
                "expected_samples": args.expected_samples,
                "pi_trace_field": args.pi_trace_field,
                "stats": stats,
            },
        )


if __name__ == "__main__":
    main()
