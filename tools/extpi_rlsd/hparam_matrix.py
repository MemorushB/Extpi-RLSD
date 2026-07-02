#!/usr/bin/env python3
"""Manage ExtPI-RLSD hyperparameter matrix runs and summaries."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shlex
from pathlib import Path
from typing import Any

DEFAULT_PI_TRACE_FIELD = "qwen32b_pi_trace"
SUMMARY_FIELDS = [
    "run_id",
    "step",
    "avg@4",
    "majority@4",
    "pass@4",
    "greedy",
    "format",
    "truncation",
    "mean_len",
    "ProxyScore",
    "ranking_score",
    "group_nonzero_std_ratio_mean",
    "rlsd_weight_std_mean",
    "rlsd_clip_total_mean",
    "privacy_leak_count",
    "pi_prompt_trunc_count",
    "hard_gate_violation",
    "selection_veto",
    "baseline_missing",
]


def _expand(value: str) -> str:
    return os.path.expandvars(value.strip())


def _bool_env(value: str | bool | None, default: bool = False) -> str:
    if value is None or value == "":
        return "True" if default else "False"
    if isinstance(value, bool):
        return "True" if value else "False"
    return "True" if str(value).strip().lower() in {"1", "true", "yes", "y"} else "False"


def _verified_field(trace_field: str) -> str:
    return trace_field.replace("_trace", "_verified") if trace_field.endswith("_trace") else f"{trace_field}_verified"


def load_matrix(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        rows = [{key: _expand(value or "") for key, value in row.items()} for row in csv.DictReader(handle)]
    if not rows:
        raise ValueError(f"Empty hyperparameter matrix: {path}")
    return rows


def get_run(rows: list[dict[str, str]], run_id: str) -> dict[str, str]:
    matches = [row for row in rows if row.get("run_id") == run_id]
    if not matches:
        known = ", ".join(row.get("run_id", "") for row in rows)
        raise ValueError(f"Unknown RUN_ID={run_id!r}; known run ids: {known}")
    if len(matches) > 1:
        raise ValueError(f"Duplicate RUN_ID={run_id!r} in matrix")
    return matches[0]


def env_for_run(row: dict[str, str]) -> dict[str, str]:
    seed = row.get("seed") or "42"
    run_id = row["run_id"]
    experiment_name = row.get("experiment_name") or f"UPOD-SELF-q4-ot800-hparam-{run_id}-s{seed}"
    pi_trace_field = row.get("pi_trace_field") or DEFAULT_PI_TRACE_FIELD
    env = {
        "HPARAM_RUN_ID": run_id,
        "EXPERIMENT_NAME": experiment_name,
        "MODEL_PATH": row.get("model_path") or "Qwen/Qwen3-4B-Thinking-2507",
        "TRAIN_FILE": row.get("train_file") or "${EXTPI_DATA_ROOT}/datasets/splits/frontier_mvp/mvp_train.parquet",
        "VAL_FILE": row.get("val_file") or "${EXTPI_DATA_ROOT}/datasets/splits/frontier_mvp/matched_dev.parquet",
        "RLSD_LAMBDA": row.get("rlsd_lambda") or "0.2",
        "RLSD_LAMBDA_WARMUP_STEPS": row.get("rlsd_lambda_warmup_steps") or "0",
        "RLSD_LAMBDA_DECAY_STEPS": row.get("rlsd_lambda_decay_steps") or "0",
        "RLSD_CLIP_RANGE": row.get("rlsd_clip_range") or row.get("rlsd_reweight_clip_range") or "0.1",
        "RLSD_NEGATIVE_ONLY": _bool_env(row.get("rlsd_negative_only"), default=True),
        "RLSD_TOKEN_MASK": row.get("rlsd_token_mask") or "4k+ans",
        "TRAIN_BATCH_SIZE": row.get("train_batch_size") or "16",
        "PPO_MINI_BATCH_SIZE": row.get("ppo_mini_batch_size") or row.get("train_batch_size") or "16",
        "ROLLOUT_N": row.get("rollout_n") or "4",
        "ACTOR_LR": row.get("actor_lr") or "5e-7",
        "TEACHER_UPDATE_MODE": row.get("teacher_update_mode") or "periodic_snapshot",
        "TEACHER_SYNC_INTERVAL": row.get("teacher_sync_interval") or "10",
        "MAX_RESPONSE_LENGTH": row.get("max_response_length") or "6144",
        "TEACHER_MAX_PROMPT_LENGTH": row.get("teacher_max_prompt_length") or "12288",
        "TOTAL_TRAINING_STEPS": row.get("total_training_steps") or "25",
        "SAVE_FREQ": row.get("save_freq") or "25",
        "TEST_FREQ": row.get("test_freq") or "25",
        "SEED": seed,
        "PI_TRACE_FIELD": pi_trace_field,
        "PI_VERIFIED_FIELD": row.get("pi_verified_field") or _verified_field(pi_trace_field),
        "HPARAM_EVAL_MAX_ROWS": row.get("eval_max_rows") or "128",
        "HPARAM_EVAL_NUM_SAMPLES": row.get("eval_num_samples") or "4",
        "HPARAM_EVAL_SEEDS": row.get("eval_seeds") or "0,1,2,3",
        "HPARAM_EVAL_MAX_NEW_TOKENS": row.get("eval_max_new_tokens") or "8192",
    }
    return {key: _expand(value) for key, value in env.items()}


def _print_exports(env: dict[str, str]) -> None:
    for key in sorted(env):
        print(f"export {key}={shlex.quote(str(env[key]))}")


def _to_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, int | float):
        return float(value)
    return None


def _metric_value(data: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        if key in data:
            value = _to_float(data[key])
            if value is not None and math.isfinite(value):
                return value
    return None


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def load_metric_records(path: str | Path) -> list[dict[str, Any]]:
    records = []
    metric_path = Path(path)
    if not metric_path.exists():
        return records
    with metric_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            data = row.get("data", {})
            if isinstance(data, dict):
                records.append({"step": row.get("step"), "data": data})
    return records


def _rolling_violation(values: list[float], *, threshold: float, window: int = 5) -> bool:
    if len(values) < window:
        return False
    return any(_mean(values[idx : idx + window]) > threshold for idx in range(0, len(values) - window + 1))


def _zero_grad_violation(values: list[float], *, window: int = 5) -> bool:
    streak = 0
    for value in values:
        if value == 0.0:
            streak += 1
            if streak >= window:
                return True
        else:
            streak = 0
    return False


def summarize_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    group_values = []
    weight_std_values = []
    clip_totals = []
    truncation_values = []
    grad_values = []
    leak_count = 0.0
    prompt_trunc_count = 0.0
    sign_flip_count = 0.0
    non_finite = False

    for record in records:
        data = record["data"]
        for value in data.values():
            scalar = _to_float(value)
            if scalar is not None and not math.isfinite(scalar):
                non_finite = True
        group = _metric_value(data, ("train/group_nonzero_std_ratio",))
        if group is not None:
            group_values.append(group)
        weight_std = _metric_value(data, ("actor/rlsd/weight_std", "rlsd/weight_std"))
        if weight_std is not None:
            weight_std_values.append(weight_std)
        clip_low = _metric_value(data, ("actor/rlsd/clip_low_ratio", "rlsd/clip_low_ratio")) or 0.0
        clip_high = _metric_value(data, ("actor/rlsd/clip_high_ratio", "rlsd/clip_high_ratio")) or 0.0
        if clip_low or clip_high:
            clip_totals.append(clip_low + clip_high)
        truncation = _metric_value(data, ("response_length/clip_ratio", "train/truncation_proxy_last_token_ratio"))
        if truncation is not None:
            truncation_values.append(truncation)
        grad = _metric_value(data, ("actor/grad_norm",))
        if grad is not None:
            grad_values.append(grad)
        leak_count = max(
            leak_count,
            _metric_value(data, ("privacy/student_prompt_pi_leak_count",)) or 0.0,
        )
        prompt_trunc_count = max(
            prompt_trunc_count,
            _metric_value(data, ("extpi/pi_teacher_prompt_truncated_count",)) or 0.0,
        )
        sign_flip_count = max(
            sign_flip_count,
            _metric_value(data, ("actor/rlsd/sign_flip_count", "rlsd/sign_flip_count")) or 0.0,
        )

    violations = []
    if leak_count > 0:
        violations.append("pi_leak")
    if prompt_trunc_count > 0:
        violations.append("pi_prompt_truncation")
    if sign_flip_count > 0:
        violations.append("sign_flip")
    if non_finite:
        violations.append("non_finite_metric")
    if _rolling_violation(clip_totals, threshold=0.25):
        violations.append("rlsd_clip_total")
    if _rolling_violation(truncation_values, threshold=0.20):
        violations.append("response_truncation")
    if _zero_grad_violation(grad_values):
        violations.append("zero_grad_norm")

    return {
        "group_nonzero_std_ratio_mean": _mean(group_values),
        "rlsd_weight_std_mean": _mean(weight_std_values),
        "rlsd_clip_total_mean": _mean(clip_totals),
        "privacy_leak_count": leak_count,
        "pi_prompt_trunc_count": prompt_trunc_count,
        "hard_gate_violation": ";".join(violations),
    }


def proxy_score(eval_payload: dict[str, Any], baseline_payload: dict[str, Any] | None) -> dict[str, Any]:
    avg = float(eval_payload.get("avg", 0.0))
    majority = float(eval_payload.get("majority", 0.0))
    pass_at = float(eval_payload.get("pass", 0.0))
    greedy = float(eval_payload.get("greedy_accuracy", 0.0))
    fmt = float(eval_payload.get("format_rate", 0.0))
    truncation = float(eval_payload.get("truncation_rate", 0.0))
    mean_len = float(eval_payload.get("mean_response_length", 0.0))
    baseline_missing = baseline_payload is None
    length_penalty = 0.0
    selection_veto = []
    if baseline_payload is not None:
        base_len = float(baseline_payload.get("mean_response_length", 0.0))
        if base_len > 0:
            length_penalty = abs(mean_len - base_len) / base_len
        if fmt < float(baseline_payload.get("format_rate", 0.0)) - 0.02:
            selection_veto.append("format_drop")
        if truncation > float(baseline_payload.get("truncation_rate", 0.0)) + 0.05:
            selection_veto.append("truncation_increase")
    score = (
        0.40 * avg
        + 0.25 * majority
        + 0.20 * pass_at
        + 0.10 * greedy
        + 0.05 * fmt
        - 0.10 * truncation
        - 0.03 * length_penalty
    )
    return {
        "ProxyScore": score,
        "ranking_score": score - (1.0 if selection_veto else 0.0),
        "selection_veto": ";".join(selection_veto),
        "baseline_missing": baseline_missing,
    }


def _checkpoint_step(checkpoint: str) -> str:
    return checkpoint.replace("step_", "") if checkpoint.startswith("step_") else checkpoint


def build_summary_row(
    *,
    run_id: str,
    metrics_jsonl: str | Path,
    eval_json: str | Path,
    baseline_eval_json: str | Path | None,
) -> dict[str, Any]:
    eval_payload = json.loads(Path(eval_json).read_text(encoding="utf-8"))
    baseline_payload = (
        json.loads(Path(baseline_eval_json).read_text(encoding="utf-8")) if baseline_eval_json else None
    )
    metric_summary = summarize_metrics(load_metric_records(metrics_jsonl))
    score_summary = proxy_score(eval_payload, baseline_payload)
    return {
        "run_id": run_id,
        "step": _checkpoint_step(str(eval_payload.get("checkpoint", ""))),
        "avg@4": float(eval_payload.get("avg", 0.0)),
        "majority@4": float(eval_payload.get("majority", 0.0)),
        "pass@4": float(eval_payload.get("pass", 0.0)),
        "greedy": float(eval_payload.get("greedy_accuracy", 0.0)),
        "format": float(eval_payload.get("format_rate", 0.0)),
        "truncation": float(eval_payload.get("truncation_rate", 0.0)),
        "mean_len": float(eval_payload.get("mean_response_length", 0.0)),
        **score_summary,
        **metric_summary,
    }


def write_summary_csv(path: str | Path, row: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    if output.exists():
        with output.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        rows = [
            old
            for old in rows
            if not (old.get("run_id") == str(row["run_id"]) and old.get("step") == str(row["step"]))
        ]
    rows.append({field: row.get(field, "") for field in SUMMARY_FIELDS})
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def cmd_export_env(args: argparse.Namespace) -> None:
    _print_exports(env_for_run(get_run(load_matrix(args.matrix), args.run_id)))


def cmd_describe(args: argparse.Namespace) -> None:
    payload = get_run(load_matrix(args.matrix), args.run_id)
    if args.output_json:
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    else:
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


def cmd_summarize(args: argparse.Namespace) -> None:
    row = build_summary_row(
        run_id=args.run_id,
        metrics_jsonl=args.metrics_jsonl,
        eval_json=args.eval_json,
        baseline_eval_json=args.baseline_eval_json,
    )
    if args.output_json:
        output_json = Path(args.output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(row, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    if args.output_csv:
        write_summary_csv(args.output_csv, row)
    if not args.output_json and not args.output_csv:
        print(json.dumps(row, ensure_ascii=False, indent=2, sort_keys=True))


def cmd_select_top(args: argparse.Namespace) -> None:
    with Path(args.summary_csv).open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    eligible = [
        row
        for row in rows
        if not row.get("hard_gate_violation") and not row.get("selection_veto")
    ]
    eligible.sort(key=lambda row: float(row.get("ranking_score") or row.get("ProxyScore") or 0.0), reverse=True)
    for row in eligible[: args.top_k]:
        print(row["run_id"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    export_parser = subparsers.add_parser("export-env")
    export_parser.add_argument("--matrix", required=True)
    export_parser.add_argument("--run_id", required=True)
    export_parser.set_defaults(func=cmd_export_env)

    describe_parser = subparsers.add_parser("describe")
    describe_parser.add_argument("--matrix", required=True)
    describe_parser.add_argument("--run_id", required=True)
    describe_parser.add_argument("--output_json", default=None)
    describe_parser.set_defaults(func=cmd_describe)

    summarize_parser = subparsers.add_parser("summarize")
    summarize_parser.add_argument("--run_id", required=True)
    summarize_parser.add_argument("--metrics_jsonl", required=True)
    summarize_parser.add_argument("--eval_json", required=True)
    summarize_parser.add_argument("--baseline_eval_json", default=None)
    summarize_parser.add_argument("--output_json", default=None)
    summarize_parser.add_argument("--output_csv", default=None)
    summarize_parser.set_defaults(func=cmd_summarize)

    top_parser = subparsers.add_parser("select-top")
    top_parser.add_argument("--summary_csv", required=True)
    top_parser.add_argument("--top_k", type=int, default=6)
    top_parser.set_defaults(func=cmd_select_top)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
