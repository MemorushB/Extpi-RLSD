#!/usr/bin/env python3
"""Convert ExtPI-RLSD JSONL rows into verl-compatible parquet rows."""

from __future__ import annotations

import argparse
from pathlib import Path

from tools.extpi_rlsd.common import read_jsonl
from verl.trainer.extpi_rlsd.prompt_assembly import DEFAULT_PI_TRACE_FIELD


def _verified_field(trace_field: str) -> str:
    return trace_field.replace("_trace", "_verified") if trace_field.endswith("_trace") else f"{trace_field}_verified"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_jsonl", required=True)
    parser.add_argument("--output_parquet", required=True)
    parser.add_argument("--prompt_key", default="prompt")
    parser.add_argument("--pi_trace_field", default=DEFAULT_PI_TRACE_FIELD)
    args = parser.parse_args()

    rows = []
    pi_verified_field = _verified_field(args.pi_trace_field)
    for row in read_jsonl(args.input_jsonl):
        problem = row["problem"]
        pi_trace = row.get(args.pi_trace_field)
        pi_verified = row.get(pi_verified_field, False)
        rows.append(
            {
                args.prompt_key: [
                    {
                        "role": "user",
                        "content": (
                            "Solve the following mathematics problem. Give a concise, self-contained derivation.\n"
                            "End with the final answer in \\boxed{...}.\n\n"
                            f"{problem}"
                        ),
                    }
                ],
                "data_source": "extpi_rlsd/math",
                "reward_model": {"style": "rule", "ground_truth": row["gold_answer"]},
                "extra_info": {
                    "id": row["id"],
                    "problem": problem,
                    args.pi_trace_field: pi_trace,
                    pi_verified_field: pi_verified,
                    "qwen32b_pi_trace": row.get("qwen32b_pi_trace"),
                    "qwen32b_pi_verified": row.get("qwen32b_pi_verified", False),
                    "qwen8b_pi_trace": row.get("qwen8b_pi_trace"),
                    "qwen8b_pi_verified": row.get("qwen8b_pi_verified", False),
                    "pi_trace_field": args.pi_trace_field,
                },
            }
        )
    output = Path(args.output_parquet)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        import pandas as pd

        pd.DataFrame(rows).to_parquet(output)
    except Exception:
        from datasets import Dataset

        Dataset.from_list(rows).to_parquet(str(output))


if __name__ == "__main__":
    main()
