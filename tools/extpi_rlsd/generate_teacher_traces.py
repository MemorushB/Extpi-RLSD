#!/usr/bin/env python3
"""Generate and verify Qwen3-8B privileged traces with vLLM."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone

from recipes.extpi_rlsd.rewards.math_verify_reward import extract_boxed_answer, verify_answer
from tools.extpi_rlsd.common import enforce_gpu6, read_jsonl, write_jsonl

PI_GENERATION_PROMPT = """Solve the following mathematics problem. Provide self-contained visible reasoning.
End with the final answer in \\boxed{{...}}.

Do not mention any gold or reference solution.

{problem}"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_jsonl", required=True)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--max_prompts", type=int, default=None)
    parser.add_argument("--max_attempts", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.78)
    parser.add_argument("--trust_remote_code", action="store_true", default=True)
    args = parser.parse_args()

    enforce_gpu6()
    rows = read_jsonl(args.input_jsonl)
    if args.max_prompts is not None:
        rows = rows[: args.max_prompts]

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    llm = LLM(
        model=args.model,
        trust_remote_code=args.trust_remote_code,
        tensor_parallel_size=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        dtype="bfloat16",
        enforce_eager=True,
    )
    sampling = SamplingParams(
        n=args.max_attempts,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_new_tokens,
    )
    prompts = []
    for row in rows:
        messages = [{"role": "user", "content": PI_GENERATION_PROMPT.format(problem=row["problem"])}]
        prompts.append(
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
        )

    outputs = llm.generate(prompts, sampling)
    updated = []
    for row, output in zip(rows, outputs, strict=True):
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
        new_row.setdefault("metadata", {})
        new_row["metadata"]["qwen8b_pi_generation"] = {
            "model": args.model,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "params": vars(args),
            "attempts": attempts,
        }
        updated.append(new_row)
    write_jsonl(args.output_jsonl, updated)


if __name__ == "__main__":
    main()
