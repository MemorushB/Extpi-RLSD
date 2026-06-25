#!/usr/bin/env python3
"""Generate visible closed-teacher SFT targets through an OpenAI-compatible API."""

from __future__ import annotations

import argparse
import os
import time
from datetime import datetime, timezone

from recipes.extpi_rlsd.rewards.math_verify_reward import extract_boxed_answer, verify_answer
from tools.extpi_rlsd.common import read_jsonl, write_jsonl

PROMPT = """Solve the following mathematics problem. Give a concise, self-contained derivation.
End with the final answer in \\boxed{{...}}.

Do not provide hidden chain-of-thought. Provide only visible reasoning that can be audited.

{problem}"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_jsonl", required=True)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--max_attempts", type=int, default=4)
    parser.add_argument("--sleep", type=float, default=1.0)
    parser.add_argument("--max_rows", type=int, default=None)
    args = parser.parse_args()

    base_url = os.environ.get("CLOSED_TEACHER_BASE_URL")
    api_key = os.environ.get("CLOSED_TEACHER_API_KEY")
    model = os.environ.get("CLOSED_TEACHER_MODEL")
    if not base_url or not api_key or not model:
        raise SystemExit("Set CLOSED_TEACHER_BASE_URL, CLOSED_TEACHER_API_KEY, and CLOSED_TEACHER_MODEL.")

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise SystemExit("Install the openai package to generate closed-teacher SFT data.") from exc

    client = OpenAI(base_url=base_url, api_key=api_key)
    rows = read_jsonl(args.input_jsonl)
    if args.max_rows is not None:
        rows = rows[: args.max_rows]
    updated = []
    for row in rows:
        attempts = []
        selected = None
        for attempt_idx in range(args.max_attempts):
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": PROMPT.format(problem=row["problem"])}],
                temperature=0.6,
                top_p=0.95,
            )
            text = response.choices[0].message.content or ""
            boxed = extract_boxed_answer(text)
            ok = verify_answer(boxed, row["gold_answer"])
            attempts.append(
                {
                    "attempt_index": attempt_idx,
                    "request_id": getattr(response, "id", None),
                    "text": text,
                    "boxed_answer": boxed,
                    "verified": ok,
                }
            )
            if ok:
                selected = text
                break
            time.sleep(args.sleep * (2**attempt_idx))
        new_row = dict(row)
        new_row["closed_teacher_output"] = selected
        new_row["closed_teacher_verified"] = selected is not None
        new_row.setdefault("metadata", {})
        new_row["metadata"]["closed_teacher_generation"] = {
            "provider": "openai_compatible",
            "base_url": base_url,
            "model": model,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "attempts": attempts,
        }
        updated.append(new_row)
    write_jsonl(args.output_jsonl, updated)


if __name__ == "__main__":
    main()
