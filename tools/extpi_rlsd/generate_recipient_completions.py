#!/usr/bin/env python3
"""Generate fixed-seed plain or PI recipient completions for uplift screening."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from recipes.extpi_rlsd.rewards.math_verify_reward import extract_boxed_answer, verify_answer
from tools.extpi_rlsd.common import enforce_gpu6, read_jsonl, write_jsonl
from verl.trainer.extpi_rlsd.prompt_assembly import PI_TEACHER_USER_TEMPLATE, STUDENT_USER_TEMPLATE


def parse_seeds(value: str) -> list[int]:
    seeds = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not seeds:
        raise ValueError("At least one seed is required")
    return seeds


def build_messages(row: dict[str, Any], mode: str) -> list[dict[str, str]]:
    problem = str(row["problem"])
    if mode == "plain":
        content = STUDENT_USER_TEMPLATE.format(problem=problem)
    elif mode == "pi":
        trace = row.get("qwen8b_pi_trace")
        if not trace:
            raise ValueError(f"PI completion requested for row without qwen8b_pi_trace: {row.get('id')}")
        content = PI_TEACHER_USER_TEMPLATE.format(problem=problem, qwen8b_pi_trace=str(trace))
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return [{"role": "user", "content": content}]


def build_prompts(tokenizer: Any, rows: list[dict[str, Any]], mode: str, enable_thinking: bool) -> list[str]:
    prompts = []
    for row in rows:
        kwargs = {"tokenize": False, "add_generation_prompt": True}
        try:
            prompts.append(
                tokenizer.apply_chat_template(build_messages(row, mode), enable_thinking=enable_thinking, **kwargs)
            )
        except TypeError:
            prompts.append(tokenizer.apply_chat_template(build_messages(row, mode), **kwargs))
    return prompts


def _record(row: dict[str, Any], mode: str, seed: int, text: str, finish_reason: str | None, token_count: int) -> dict:
    boxed = extract_boxed_answer(text)
    return {
        "id": row["id"],
        "mode": mode,
        "seed": seed,
        "completion": text,
        "boxed_answer": boxed,
        "verified": verify_answer(boxed, row["gold_answer"]),
        "finish_reason": finish_reason,
        "token_count": token_count,
        "truncated": finish_reason == "length",
    }


def _generate_vllm(args: argparse.Namespace, rows: list[dict[str, Any]], seeds: list[int]) -> list[dict]:
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    prompts = build_prompts(tokenizer, rows, args.mode, enable_thinking=args.enable_thinking)
    llm = LLM(
        model=args.model,
        trust_remote_code=args.trust_remote_code,
        tensor_parallel_size=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        dtype="bfloat16",
    )
    records = []
    for seed in seeds:
        outputs = llm.generate(
            prompts,
            SamplingParams(
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_new_tokens,
                n=1,
                seed=seed,
            ),
        )
        for row, output in zip(rows, outputs, strict=True):
            candidate = output.outputs[0]
            records.append(
                _record(row, args.mode, seed, candidate.text, candidate.finish_reason, len(candidate.token_ids))
            )
    return records


def _generate_hf(args: argparse.Namespace, rows: list[dict[str, Any]], seeds: list[int]) -> list[dict]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=args.trust_remote_code,
        attn_implementation=args.attn_implementation,
        device_map="auto",
    )
    model.eval()
    prompts = build_prompts(tokenizer, rows, args.mode, enable_thinking=args.enable_thinking)
    device = next(model.parameters()).device
    records = []
    for row, prompt in zip(rows, prompts, strict=True):
        encoded = tokenizer(prompt, return_tensors="pt").to(device)
        prompt_len = int(encoded["input_ids"].shape[-1])
        for seed in seeds:
            generator = torch.Generator(device=device)
            generator.manual_seed(seed)
            output = model.generate(
                **encoded,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                max_new_tokens=args.max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                generator=generator,
            )
            new_tokens = output[0, prompt_len:]
            token_ids = new_tokens.detach().cpu().tolist()
            reached_limit = len(token_ids) >= args.max_new_tokens
            saw_eos = tokenizer.eos_token_id in token_ids
            finish_reason = "length" if reached_limit and not saw_eos else "stop"
            records.append(
                _record(
                    row,
                    args.mode,
                    seed,
                    tokenizer.decode(new_tokens, skip_special_tokens=True),
                    finish_reason,
                    len(token_ids),
                )
            )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_jsonl", required=True)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--mode", choices=["plain", "pi"], required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--backend", choices=["vllm", "hf"], default="vllm")
    parser.add_argument("--seeds", default="0,1,2,3")
    parser.add_argument("--max_rows", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.5)
    parser.add_argument("--enable_thinking", action="store_true")
    parser.add_argument("--trust_remote_code", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--attn_implementation", default="flash_attention_2")
    args = parser.parse_args()

    enforce_gpu6()
    rows = read_jsonl(args.input_jsonl)
    if args.mode == "pi":
        rows = [row for row in rows if bool(row.get("qwen8b_pi_verified", False)) and row.get("qwen8b_pi_trace")]
    if args.max_rows is not None:
        rows = rows[: args.max_rows]
    seeds = parse_seeds(args.seeds)
    records = _generate_vllm(args, rows, seeds) if args.backend == "vllm" else _generate_hf(args, rows, seeds)
    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_path, records)


if __name__ == "__main__":
    main()
