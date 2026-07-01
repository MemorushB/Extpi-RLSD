#!/usr/bin/env python3
"""Build generic off-policy teacher-response SFT parquet rows."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from hashlib import sha256
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.extpi_rlsd.common import read_jsonl, stable_problem_id, write_json  # noqa: E402
from verl.trainer.extpi_rlsd.prompt_assembly import STUDENT_USER_TEMPLATE  # noqa: E402

DEFAULT_RESPONSE_FIELDS = (
    "closed_teacher_output",
    "teacher_output",
    "response",
    "solution",
    "assistant",
)


def _parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y"}:
        return True
    if normalized in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean value, got {value!r}")


def _write_parquet(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import pandas as pd

        pd.DataFrame(rows).to_parquet(output_path)
    except Exception:
        from datasets import Dataset

        Dataset.from_list(rows).to_parquet(str(output_path))


def _pick_response(row: dict[str, Any], response_field: str | None) -> tuple[str | None, str | None]:
    fields = (response_field,) if response_field else DEFAULT_RESPONSE_FIELDS
    for field in fields:
        value = row.get(field)
        if value is not None and str(value).strip():
            return str(value), field
    return None, None


def build_rows(
    source_rows: list[dict[str, Any]],
    *,
    problem_field: str = "problem",
    response_field: str | None = None,
    answer_field: str = "gold_answer",
    verified_field: str = "verified",
    require_verified: bool = False,
    enable_thinking: bool = False,
    data_source: str = "extpi_rlsd/offpolicy_sft",
    max_rows: int | None = None,
) -> tuple[list[dict[str, Any]], Counter[str], int]:
    rows: list[dict[str, Any]] = []
    response_field_counts: Counter[str] = Counter()
    dropped_count = 0
    for source_row in source_rows:
        if require_verified and not bool(source_row.get(verified_field, False)):
            dropped_count += 1
            continue
        problem = source_row.get(problem_field)
        response, used_field = _pick_response(source_row, response_field)
        if problem is None or not str(problem).strip() or response is None or used_field is None:
            dropped_count += 1
            continue
        problem_text = str(problem)
        row_id = str(source_row.get("id") or stable_problem_id(problem_text))
        response_field_counts[used_field] += 1
        rows.append(
            {
                "messages": [
                    {"role": "user", "content": STUDENT_USER_TEMPLATE.format(problem=problem_text)},
                    {"role": "assistant", "content": response},
                ],
                "enable_thinking": bool(enable_thinking),
                "data_source": data_source,
                "reward_model": {"style": "rule", "ground_truth": source_row.get(answer_field)},
                "extra_info": {
                    "id": row_id,
                    "problem": problem_text,
                    "response_field": used_field,
                    "verified": bool(source_row.get(verified_field, False)),
                },
            }
        )
        if max_rows is not None and len(rows) >= max_rows:
            break
    return rows, response_field_counts, dropped_count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_jsonl", required=True)
    parser.add_argument("--output_parquet", required=True)
    parser.add_argument("--manifest_json", default=None)
    parser.add_argument("--problem_field", default="problem")
    parser.add_argument("--response_field", default=None)
    parser.add_argument("--answer_field", default="gold_answer")
    parser.add_argument("--verified_field", default="verified")
    parser.add_argument("--require_verified", type=_parse_bool, default=False)
    parser.add_argument("--allow_unverified", action="store_true")
    parser.add_argument("--enable_thinking", type=_parse_bool, default=False)
    parser.add_argument("--data_source", default="extpi_rlsd/offpolicy_sft")
    parser.add_argument("--max_rows", type=int, default=None)
    args = parser.parse_args()

    require_verified = False if args.allow_unverified else bool(args.require_verified)
    source_rows = read_jsonl(args.input_jsonl)
    rows, response_field_counts, dropped_count = build_rows(
        source_rows,
        problem_field=args.problem_field,
        response_field=args.response_field,
        answer_field=args.answer_field,
        verified_field=args.verified_field,
        require_verified=require_verified,
        enable_thinking=bool(args.enable_thinking),
        data_source=args.data_source,
        max_rows=args.max_rows,
    )
    if not rows:
        raise SystemExit("No teacher-response rows found; cannot build SFT parquet.")

    output_path = Path(args.output_parquet)
    _write_parquet(rows, output_path)
    manifest_path = Path(args.manifest_json) if args.manifest_json else output_path.with_name("manifest.json")
    write_json(
        manifest_path,
        {
            "input_jsonl": args.input_jsonl,
            "output_parquet": str(output_path),
            "source_count": len(source_rows),
            "written_count": len(rows),
            "dropped_count": dropped_count,
            "problem_field": args.problem_field,
            "response_field": args.response_field or list(DEFAULT_RESPONSE_FIELDS),
            "response_field_counts": dict(response_field_counts),
            "answer_field": args.answer_field,
            "verified_field": args.verified_field,
            "require_verified": require_verified,
            "enable_thinking": bool(args.enable_thinking),
            "data_source": args.data_source,
            "prompt_template_sha256": sha256(STUDENT_USER_TEMPLATE.encode("utf-8")).hexdigest(),
            "messages_key": "messages",
            "loss_mask_source": "verl MultiTurnSFTDataset masks non-assistant turns and assistant generation prompt.",
        },
    )


if __name__ == "__main__":
    main()
