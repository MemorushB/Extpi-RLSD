#!/usr/bin/env python3
"""Build verl SFT parquet rows from verified closed-teacher outputs."""

from __future__ import annotations

import argparse
import os
from hashlib import sha256
from pathlib import Path

from tools.extpi_rlsd.common import read_jsonl, write_json
from verl.trainer.extpi_rlsd.prompt_assembly import STUDENT_USER_TEMPLATE


def _write_parquet(rows: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import pandas as pd

        pd.DataFrame(rows).to_parquet(output_path)
    except Exception:
        from datasets import Dataset

        Dataset.from_list(rows).to_parquet(str(output_path))


def build_rows(source_rows: list[dict], max_rows: int | None = None) -> list[dict]:
    rows = []
    for row in source_rows:
        if not bool(row.get("closed_teacher_verified", False)):
            continue
        response = row.get("closed_teacher_output")
        if not response:
            continue
        problem = str(row["problem"])
        rows.append(
            {
                "messages": [
                    {"role": "user", "content": STUDENT_USER_TEMPLATE.format(problem=problem)},
                    {"role": "assistant", "content": str(response)},
                ],
                "enable_thinking": False,
                "data_source": "extpi_rlsd/closed_sft",
                "reward_model": {"style": "rule", "ground_truth": row["gold_answer"]},
                "extra_info": {
                    "id": row["id"],
                    "problem": problem,
                    "closed_teacher_verified": True,
                    "closed_teacher_model": (
                        row.get("metadata", {}).get("closed_teacher_generation", {}).get("model")
                        if isinstance(row.get("metadata"), dict)
                        else None
                    ),
                },
            }
        )
        if max_rows is not None and len(rows) >= max_rows:
            break
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_jsonl", required=True)
    parser.add_argument("--output_parquet", required=True)
    parser.add_argument("--manifest_json", default=None)
    parser.add_argument("--closed_teacher_model", default=os.environ.get("CLOSED_TEACHER_MODEL"))
    parser.add_argument("--max_rows", type=int, default=None)
    args = parser.parse_args()

    source_rows = read_jsonl(args.input_jsonl)
    rows = build_rows(source_rows, max_rows=args.max_rows)
    if not rows:
        raise SystemExit("No verified closed-teacher rows found; cannot build SFT parquet.")

    output_path = Path(args.output_parquet)
    _write_parquet(rows, output_path)

    manifest_path = Path(args.manifest_json) if args.manifest_json else output_path.with_name("manifest.json")
    verified_source_count = sum(
        1 for row in source_rows if bool(row.get("closed_teacher_verified", False)) and row.get("closed_teacher_output")
    )
    write_json(
        manifest_path,
        {
            "input_jsonl": args.input_jsonl,
            "output_parquet": str(output_path),
            "closed_teacher_model": args.closed_teacher_model,
            "source_count": len(source_rows),
            "verified_source_count": verified_source_count,
            "written_count": len(rows),
            "verified_rate": verified_source_count / len(source_rows) if source_rows else 0.0,
            "prompt_template_sha256": sha256(STUDENT_USER_TEMPLATE.encode("utf-8")).hexdigest(),
            "messages_key": "messages",
            "loss_mask_source": "verl MultiTurnSFTDataset masks non-assistant turns and assistant generation prompt.",
        },
    )


if __name__ == "__main__":
    main()
