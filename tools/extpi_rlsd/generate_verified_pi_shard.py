#!/usr/bin/env python3
"""Generate a verified PI-trace shard for ExtPI-RLSD data preparation."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from recipes.extpi_rlsd.rewards.math_verify_reward import extract_boxed_answer, verify_answer
from tools.extpi_rlsd.common import read_jsonl

PI_GENERATION_PROMPT = """Solve the following mathematics problem. Provide self-contained visible reasoning.
End with the final answer in \\boxed{{...}}.

Do not mention any gold or reference solution.

{problem}"""


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def load_seen_ids(*paths: Path) -> set[str]:
    seen = set()
    for path in paths:
        if not path.exists():
            continue
        for row in read_jsonl(path):
            if row.get("id"):
                seen.add(str(row["id"]))
    return seen


def build_prompt(tokenizer: Any, problem: str) -> str:
    messages = [{"role": "user", "content": PI_GENERATION_PROMPT.format(problem=problem)}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_jsonl", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--worker_id", type=int, required=True)
    parser.add_argument("--num_workers", type=int, required=True)
    parser.add_argument("--target_successes", type=int, required=True)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_attempts", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--max_model_len", type=int, default=8192)
    parser.add_argument("--max_num_seqs", type=int, default=4)
    parser.add_argument("--max_input_rows", type=int, default=None)
    args = parser.parse_args()

    shard_dir = Path(args.output_dir) / f"worker_{args.worker_id}"
    success_path = shard_dir / "success.jsonl"
    failure_path = shard_dir / "failure.jsonl"
    seen = load_seen_ids(success_path, failure_path)
    successes = sum(1 for _ in success_path.open("r", encoding="utf-8")) if success_path.exists() else 0
    if successes >= args.target_successes:
        print(f"worker={args.worker_id} already has {successes} successes")
        return

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        tensor_parallel_size=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        enforce_eager=True,
    )
    sampling = SamplingParams(
        n=args.max_attempts,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_new_tokens,
    )

    source_rows = read_jsonl(args.input_jsonl)
    if args.max_input_rows is not None:
        source_rows = source_rows[: args.max_input_rows]
    rows = [
        row
        for index, row in enumerate(source_rows)
        if index % args.num_workers == args.worker_id and str(row.get("id")) not in seen
    ]

    for start in range(0, len(rows), args.batch_size):
        if successes >= args.target_successes:
            break
        batch = rows[start : start + args.batch_size]
        prompts = [build_prompt(tokenizer, str(row["problem"])) for row in batch]
        outputs = llm.generate(prompts, sampling)
        for row, output in zip(batch, outputs, strict=True):
            attempts = []
            verified = []
            for idx, candidate in enumerate(output.outputs):
                text = candidate.text
                boxed = extract_boxed_answer(text)
                ok = verify_answer(boxed, row["gold_answer"])
                attempt = {
                    "attempt_index": idx,
                    "text": text,
                    "boxed_answer": boxed,
                    "verified": ok,
                    "finish_reason": candidate.finish_reason,
                    "token_count": len(candidate.token_ids),
                }
                attempts.append(attempt)
                if ok:
                    verified.append(attempt)
            selected = min(verified, key=lambda item: item["token_count"]) if verified else None
            new_row = dict(row)
            new_row["qwen8b_pi_trace"] = None if selected is None else selected["text"]
            new_row["qwen8b_pi_verified"] = selected is not None
            new_row["qwen8b_pi_attempts"] = len(attempts)
            new_row["qwen32b_pi_trace"] = None if selected is None else selected["text"]
            new_row["qwen32b_pi_verified"] = selected is not None
            new_row.setdefault("metadata", {})
            new_row["metadata"]["qwen8b_pi_generation"] = {
                "model": "Qwen3-32B",
                "model_path": args.model,
                "trace_label": "qwen3_32b_pi_thinking_on",
                "enable_thinking": True,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "worker_id": args.worker_id,
                "num_workers": args.num_workers,
                "params": vars(args),
                "attempts": attempts,
            }
            new_row["metadata"]["qwen32b_pi_generation"] = new_row["metadata"]["qwen8b_pi_generation"]
            append_jsonl(success_path if selected is not None else failure_path, new_row)
            if selected is not None:
                successes += 1
        print(f"worker={args.worker_id} processed={start + len(batch)} successes={successes}", flush=True)

    print(f"worker={args.worker_id} finished successes={successes}")


if __name__ == "__main__":
    main()
