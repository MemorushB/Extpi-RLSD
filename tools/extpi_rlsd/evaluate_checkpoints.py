#!/usr/bin/env python3
"""Evaluate checkpoints and write aggregator-compatible ExtPI-RLSD JSON metrics."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from recipes.extpi_rlsd.rewards.math_verify_reward import compute_score, verify_answer
from tools.extpi_rlsd.common import read_jsonl
from verl.trainer.extpi_rlsd.prompt_assembly import STUDENT_USER_TEMPLATE


@dataclass
class EvalItem:
    sample_id: str
    problem: str
    ground_truth: str
    messages: list[dict[str, str]]
    pi_trace: str | None = None


def _to_python(value: Any) -> Any:
    if hasattr(value, "as_py"):
        return _to_python(value.as_py())
    if hasattr(value, "tolist") and not isinstance(value, str):
        return _to_python(value.tolist())
    if isinstance(value, dict):
        return {str(key): _to_python(inner) for key, inner in value.items()}
    if isinstance(value, list | tuple):
        return [_to_python(inner) for inner in value]
    return value


def _load_raw_rows(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if path.suffix == ".parquet":
        import pandas as pd

        return [_to_python(row) for row in pd.read_parquet(path).to_dict(orient="records")]
    return read_jsonl(path)


def load_eval_items(path: str | Path, max_rows: int | None = None) -> list[EvalItem]:
    rows = _load_raw_rows(path)
    if max_rows is not None:
        rows = rows[:max_rows]
    items = []
    for idx, row in enumerate(rows):
        row = _to_python(row)
        extra_info = row.get("extra_info") if isinstance(row.get("extra_info"), dict) else {}
        reward_model = row.get("reward_model") if isinstance(row.get("reward_model"), dict) else {}
        problem = row.get("problem") or extra_info.get("problem")
        if problem is None and isinstance(row.get("prompt"), list) and row["prompt"]:
            problem = row["prompt"][0].get("content", "")
        if problem is None:
            raise ValueError(f"Cannot find problem text in eval row {idx}")
        ground_truth = row.get("gold_answer") or reward_model.get("ground_truth")
        if ground_truth is None:
            raise ValueError(f"Cannot find ground truth in eval row {idx}")
        messages = row.get("prompt")
        if not isinstance(messages, list):
            messages = [{"role": "user", "content": STUDENT_USER_TEMPLATE.format(problem=str(problem))}]
        items.append(
            EvalItem(
                sample_id=str(row.get("id") or extra_info.get("id") or idx),
                problem=str(problem),
                ground_truth=str(ground_truth),
                messages=[{"role": str(msg["role"]), "content": str(msg["content"])} for msg in messages],
                pi_trace=extra_info.get("qwen8b_pi_trace") or row.get("qwen8b_pi_trace"),
            )
        )
    return items


def build_prompts(tokenizer: Any, items: list[EvalItem], enable_thinking: bool) -> list[str]:
    prompts = []
    for item in items:
        kwargs = {"tokenize": False, "add_generation_prompt": True}
        try:
            prompts.append(tokenizer.apply_chat_template(item.messages, enable_thinking=enable_thinking, **kwargs))
        except TypeError:
            prompts.append(tokenizer.apply_chat_template(item.messages, **kwargs))
    return prompts


def _normal_answer(value: str | None) -> str:
    return "" if value is None else "".join(str(value).lower().split())


def _response_metrics(text: str, ground_truth: str) -> tuple[float, float, str]:
    result = compute_score("extpi_rlsd/eval", text, ground_truth)
    return float(result["accuracy"]), float(result["format"]), str(result.get("boxed_answer", ""))


def _hf_truncated(new_tokens: Any, *, eos_token_id: int | None, max_new_tokens: int) -> bool:
    token_ids = new_tokens.detach().cpu().tolist() if hasattr(new_tokens, "detach") else list(new_tokens)
    return len(token_ids) >= max_new_tokens and (eos_token_id is None or eos_token_id not in token_ids)


def summarize_scored_items(
    *,
    run: str,
    checkpoint: str,
    scored_items: list[dict[str, Any]],
) -> dict[str, Any]:
    per_problem = []
    sample_accuracies = []
    sample_formats = []
    truncations = []
    response_lengths = []
    prompt_tokens = []
    completion_tokens = []
    leakage_hits = 0
    total_samples = 0

    for item in scored_items:
        sample_scores = []
        boxed_answers = []
        for sample in item["samples"]:
            accuracy, fmt, boxed = _response_metrics(sample["text"], item["ground_truth"])
            sample_scores.append(accuracy)
            sample_accuracies.append(accuracy)
            sample_formats.append(fmt)
            boxed_answers.append(boxed)
            truncations.append(float(bool(sample.get("truncated", False))))
            response_lengths.append(float(sample.get("completion_tokens", len(sample["text"]))))
            prompt_tokens.append(float(sample.get("prompt_tokens", 0)))
            completion_tokens.append(float(sample.get("completion_tokens", 0)))
            pi_trace = item.get("pi_trace")
            if "<reference_solution>" in sample["text"] or (pi_trace and pi_trace.strip() in sample["text"]):
                leakage_hits += 1
            total_samples += 1

        answer_counts = Counter(_normal_answer(answer) for answer in boxed_answers if answer)
        majority_answer = answer_counts.most_common(1)[0][0] if answer_counts else ""
        majority_correct = verify_answer(majority_answer, item["ground_truth"]) if majority_answer else False
        greedy_accuracy, greedy_format, greedy_boxed = _response_metrics(item["greedy"]["text"], item["ground_truth"])
        per_problem.append(
            {
                "id": item["id"],
                "avg": sum(sample_scores) / len(sample_scores) if sample_scores else 0.0,
                "pass": float(any(score > 0 for score in sample_scores)),
                "majority": float(majority_correct),
                "greedy_accuracy": greedy_accuracy,
                "greedy_format": greedy_format,
                "greedy_boxed_answer": greedy_boxed,
            }
        )

    problem_avgs = [row["avg"] for row in per_problem]
    mean_problem_avg = sum(problem_avgs) / len(problem_avgs) if problem_avgs else 0.0
    group_variance = (
        sum((value - mean_problem_avg) ** 2 for value in problem_avgs) / len(problem_avgs) if problem_avgs else 0.0
    )
    return {
        "run": run,
        "checkpoint": checkpoint,
        "avg": sum(sample_accuracies) / len(sample_accuracies) if sample_accuracies else 0.0,
        "majority": sum(row["majority"] for row in per_problem) / len(per_problem) if per_problem else 0.0,
        "pass": sum(row["pass"] for row in per_problem) / len(per_problem) if per_problem else 0.0,
        "greedy_accuracy": (
            sum(row["greedy_accuracy"] for row in per_problem) / len(per_problem) if per_problem else 0.0
        ),
        "format_rate": sum(sample_formats) / len(sample_formats) if sample_formats else 0.0,
        "truncation_rate": sum(truncations) / len(truncations) if truncations else 0.0,
        "mean_response_length": sum(response_lengths) / len(response_lengths) if response_lengths else 0.0,
        "group_variance": group_variance,
        "all_correct_rate": sum(1.0 for row in per_problem if row["avg"] == 1.0) / len(per_problem)
        if per_problem
        else 0.0,
        "all_wrong_rate": sum(1.0 for row in per_problem if row["avg"] == 0.0) / len(per_problem)
        if per_problem
        else 0.0,
        "prompt_tokens": sum(prompt_tokens),
        "completion_tokens": sum(completion_tokens),
        "total_samples": total_samples,
        "privacy_leakage_rate": leakage_hits / total_samples if total_samples else 0.0,
        "per_problem": per_problem,
    }


def _evaluate_hf(args: argparse.Namespace, items: list[EvalItem]) -> list[dict[str, Any]]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_path = args.checkpoint or args.model
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=args.trust_remote_code,
        attn_implementation=args.attn_implementation,
        device_map="auto",
    )
    if args.adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter_path)
    model.eval()

    prompts = build_prompts(tokenizer, items, enable_thinking=args.enable_thinking)
    scored = []
    for item, prompt in zip(items, prompts, strict=True):
        encoded = tokenizer(prompt, return_tensors="pt").to(model.device)
        prompt_len = int(encoded["input_ids"].shape[-1])
        greedy = model.generate(
            **encoded,
            do_sample=False,
            max_new_tokens=args.max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
        )
        greedy_new = greedy[0, prompt_len:]
        samples = []
        for _ in range(args.num_samples):
            output = model.generate(
                **encoded,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                max_new_tokens=args.max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
            )
            new_tokens = output[0, prompt_len:]
            samples.append(
                {
                    "text": tokenizer.decode(new_tokens, skip_special_tokens=True),
                    "truncated": _hf_truncated(
                        new_tokens, eos_token_id=tokenizer.eos_token_id, max_new_tokens=args.max_new_tokens
                    ),
                    "prompt_tokens": prompt_len,
                    "completion_tokens": int(len(new_tokens)),
                }
            )
        scored.append(
            {
                "id": item.sample_id,
                "ground_truth": item.ground_truth,
                "pi_trace": item.pi_trace,
                "greedy": {
                    "text": tokenizer.decode(greedy_new, skip_special_tokens=True),
                    "truncated": _hf_truncated(
                        greedy_new, eos_token_id=tokenizer.eos_token_id, max_new_tokens=args.max_new_tokens
                    ),
                    "prompt_tokens": prompt_len,
                    "completion_tokens": int(len(greedy_new)),
                },
                "samples": samples,
            }
        )
    return scored


def _evaluate_vllm(args: argparse.Namespace, items: list[EvalItem]) -> list[dict[str, Any]]:
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    model_path = args.checkpoint or args.model
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    prompts = build_prompts(tokenizer, items, enable_thinking=args.enable_thinking)
    prompt_token_counts = [len(tokenizer.encode(prompt, add_special_tokens=False)) for prompt in prompts]
    llm = LLM(model=model_path, trust_remote_code=args.trust_remote_code, tensor_parallel_size=1)
    greedy_outputs = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=args.max_new_tokens, n=1))
    sample_outputs = llm.generate(
        prompts,
        SamplingParams(
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_new_tokens,
            n=args.num_samples,
        ),
    )
    scored = []
    for item, prompt_tokens, greedy_output, sample_output in zip(
        items, prompt_token_counts, greedy_outputs, sample_outputs, strict=True
    ):
        samples = [
            {
                "text": output.text,
                "truncated": getattr(output, "finish_reason", None) == "length",
                "prompt_tokens": prompt_tokens,
                "completion_tokens": len(output.token_ids),
            }
            for output in sample_output.outputs
        ]
        greedy = greedy_output.outputs[0]
        scored.append(
            {
                "id": item.sample_id,
                "ground_truth": item.ground_truth,
                "pi_trace": item.pi_trace,
                "greedy": {
                    "text": greedy.text,
                    "truncated": getattr(greedy, "finish_reason", None) == "length",
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": len(greedy.token_ids),
                },
                "samples": samples,
            }
        )
    return scored


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval_file", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--adapter_path", default=None)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--run", default=None)
    parser.add_argument("--checkpoint_name", default=None)
    parser.add_argument("--backend", choices=["hf", "vllm"], default="hf")
    parser.add_argument("--num_samples", type=int, default=12)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--max_rows", type=int, default=None)
    parser.add_argument("--enable_thinking", action="store_true")
    parser.add_argument("--trust_remote_code", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--attn_implementation", default="flash_attention_2")
    args = parser.parse_args()

    items = load_eval_items(args.eval_file, max_rows=args.max_rows)
    scored_items = _evaluate_vllm(args, items) if args.backend == "vllm" else _evaluate_hf(args, items)
    payload = summarize_scored_items(
        run=args.run or Path(args.checkpoint or args.model).name,
        checkpoint=args.checkpoint_name or Path(args.checkpoint or args.model).name,
        scored_items=scored_items,
    )
    payload["eval_file"] = args.eval_file
    payload["model"] = args.model
    payload["checkpoint_path"] = args.checkpoint
    payload["adapter_path"] = args.adapter_path
    payload["backend"] = args.backend
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
