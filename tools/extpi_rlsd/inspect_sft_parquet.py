#!/usr/bin/env python3
"""Inspect off-policy SFT parquet files before launching verl SFT."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from verl.utils.py_functional import convert_nested_value_to_list_recursive  # noqa: E402
from verl.utils.tokenizer.chat_template import extract_system_prompt_and_generation  # noqa: E402

PI_LEAK_PATTERNS = (
    "qwen8b_pi_trace",
    "qwen32b_pi_trace",
    "privileged",
    "teacher trace",
)


def _parse_bool(value: str | bool | None) -> bool | None:
    if value is None or isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y"}:
        return True
    if normalized in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean value, got {value!r}")


def _token_ids_from_template(tokenizer: Any, messages: list[dict[str, Any]], enable_thinking: bool | None) -> list[int]:
    kwargs = {}
    if enable_thinking is not None:
        kwargs["enable_thinking"] = enable_thinking
    output = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=False,
        tokenize=True,
        return_dict=False,
        **kwargs,
    )
    if isinstance(output, dict):
        output = output["input_ids"]
    if hasattr(output, "tolist"):
        output = output.tolist()
    if output and isinstance(output[0], list):
        output = output[0]
    return [int(token_id) for token_id in output]


def _prompt_prefix_lengths(tokenizer: Any) -> tuple[int, int]:
    try:
        system_prompt, generation_prompt = extract_system_prompt_and_generation(tokenizer)
        return len(system_prompt), len(generation_prompt)
    except Exception:
        return 0, 0


def _message_content(message: dict[str, Any]) -> str:
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False)


def _row_id(row: dict[str, Any], index: int) -> str:
    if row.get("id") is not None:
        return str(row["id"])
    extra_info = row.get("extra_info")
    if isinstance(extra_info, dict) and extra_info.get("id") is not None:
        return str(extra_info["id"])
    return str(index)


def _percentiles(values: list[int]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {key: 0.0 for key in ("mean", "p50", "p90", "p95", "p99", "max")}
    return {
        "mean": float(array.mean()),
        "p50": float(np.percentile(array, 50)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "max": float(array.max()),
    }


def inspect_dataframe(
    dataframe: pd.DataFrame,
    *,
    tokenizer: Any,
    max_length: int,
    messages_key: str = "messages",
    enable_thinking_default: bool | None = False,
    max_rows: int | None = None,
) -> dict[str, Any]:
    if dataframe.empty:
        raise ValueError("SFT parquet is empty")
    if messages_key not in dataframe.columns:
        raise ValueError(f"SFT parquet missing messages field {messages_key!r}")
    if max_rows is not None:
        dataframe = dataframe.head(max_rows)

    system_prompt_len, generation_prompt_len = _prompt_prefix_lengths(tokenizer)
    total_lengths: list[int] = []
    assistant_loss_lengths: list[int] = []
    overlong_ids: list[str] = []
    pi_leak_ids: list[str] = []
    invalid_rows: list[str] = []

    for index, row in enumerate(dataframe.to_dict(orient="records")):
        row_id = _row_id(row, index)
        messages = convert_nested_value_to_list_recursive(row[messages_key])
        if not isinstance(messages, list) or len(messages) < 2:
            invalid_rows.append(row_id)
            continue
        roles = [message.get("role") for message in messages if isinstance(message, dict)]
        if "user" not in roles or "assistant" not in roles:
            invalid_rows.append(row_id)
            continue

        enable_thinking = row.get("enable_thinking", enable_thinking_default)
        if enable_thinking is not None:
            enable_thinking = bool(enable_thinking)

        total_length = 0
        assistant_loss_length = 0
        assistant_has_content = False
        for message_index, message in enumerate(messages):
            if not isinstance(message, dict):
                invalid_rows.append(row_id)
                continue
            content = _message_content(message)
            role = message.get("role")
            if message.get("role") == "user":
                lowered = content.lower()
                if any(pattern in lowered for pattern in PI_LEAK_PATTERNS):
                    pi_leak_ids.append(row_id)
            if role == "assistant" and content.strip():
                assistant_has_content = True
            token_ids = _token_ids_from_template(tokenizer, [message], enable_thinking)
            effective_length = len(token_ids)
            if message_index != 0 and role != "system":
                effective_length = max(0, effective_length - system_prompt_len)
            total_length += effective_length
            if role == "assistant":
                assistant_loss_length += max(0, effective_length - generation_prompt_len)

        if not assistant_has_content or assistant_loss_length <= 0:
            invalid_rows.append(row_id)
            continue
        total_lengths.append(total_length)
        assistant_loss_lengths.append(assistant_loss_length)
        if total_length > max_length:
            overlong_ids.append(row_id)

    unique_pi_leak_ids = list(dict.fromkeys(pi_leak_ids))
    unique_invalid_rows = list(dict.fromkeys(invalid_rows))
    return {
        "rows": int(len(dataframe)),
        "checked_rows": int(len(total_lengths)),
        "invalid_row_count": len(unique_invalid_rows),
        "invalid_row_ids": unique_invalid_rows[:20],
        "max_length": int(max_length),
        "total_tokens": _percentiles(total_lengths),
        "assistant_loss_tokens": _percentiles(assistant_loss_lengths),
        "overlong_count": len(overlong_ids),
        "overlong_ids": overlong_ids[:20],
        "pi_leak_count": len(unique_pi_leak_ids),
        "pi_leak_ids": unique_pi_leak_ids[:20],
        "pi_leak_patterns": list(PI_LEAK_PATTERNS),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--max_length", type=int, required=True)
    parser.add_argument("--messages_key", default="messages")
    parser.add_argument("--enable_thinking_default", type=_parse_bool, default=False)
    parser.add_argument("--max_rows", type=int, default=None)
    parser.add_argument("--fail_on_overlong", action="store_true")
    parser.add_argument("--allow_pi_leak", action="store_true")
    args = parser.parse_args()

    parquet_path = Path(args.parquet)
    if not parquet_path.exists():
        raise SystemExit(f"SFT parquet does not exist: {parquet_path}")

    from verl.utils import hf_tokenizer

    tokenizer = hf_tokenizer(args.model)
    summary = inspect_dataframe(
        pd.read_parquet(parquet_path),
        tokenizer=tokenizer,
        max_length=args.max_length,
        messages_key=args.messages_key,
        enable_thinking_default=args.enable_thinking_default,
        max_rows=args.max_rows,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))

    if summary["invalid_row_count"] > 0:
        raise SystemExit("SFT parquet contains invalid rows")
    if args.fail_on_overlong and summary["overlong_count"] > 0:
        raise SystemExit("SFT parquet contains overlong rows")
    if not args.allow_pi_leak and summary["pi_leak_count"] > 0:
        raise SystemExit("SFT parquet user prompts contain PI leakage markers")


if __name__ == "__main__":
    main()
