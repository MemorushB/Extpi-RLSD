import json
from pathlib import Path

from tools.extpi_rlsd import hparam_matrix

ROOT = Path(__file__).resolve().parents[2]


def test_hparam_matrix_exports_qwen32b_trace_field_defaults():
    rows = hparam_matrix.load_matrix(ROOT / "experiments" / "extpi_rlsd" / "hparam_matrix.csv")
    env = hparam_matrix.env_for_run(hparam_matrix.get_run(rows, "R1-01"))

    assert env["PI_TRACE_FIELD"] == "qwen32b_pi_trace"
    assert env["RLSD_TOKEN_MASK"] == "4k+ans"
    assert env["RLSD_NEGATIVE_ONLY"] == "True"
    assert env["PPO_MINI_BATCH_SIZE"] == env["TRAIN_BATCH_SIZE"] == "16"


def test_hparam_summary_computes_proxy_score_and_hard_gates(tmp_path):
    metrics_jsonl = tmp_path / "metrics.jsonl"
    records = [
        {
            "step": idx,
            "data": {
                "train/group_nonzero_std_ratio": 0.5,
                "rlsd/weight_std": 0.1,
                "rlsd/clip_low_ratio": 0.1,
                "rlsd/clip_high_ratio": 0.2,
                "response_length/clip_ratio": 0.0,
                "actor/grad_norm": 1.0,
                "privacy/student_prompt_pi_leak_count": 0,
                "extpi/pi_teacher_prompt_truncated_count": 0,
                "rlsd/sign_flip_count": 0,
            },
        }
        for idx in range(5)
    ]
    metrics_jsonl.write_text("\n".join(json.dumps(row) for row in records) + "\n", encoding="utf-8")
    eval_json = tmp_path / "eval.json"
    eval_json.write_text(
        json.dumps(
            {
                "checkpoint": "step_25",
                "avg": 0.4,
                "majority": 0.3,
                "pass": 0.5,
                "greedy_accuracy": 0.25,
                "format_rate": 0.95,
                "truncation_rate": 0.1,
                "mean_response_length": 1000,
            }
        ),
        encoding="utf-8",
    )

    row = hparam_matrix.build_summary_row(
        run_id="R1-01",
        metrics_jsonl=metrics_jsonl,
        eval_json=eval_json,
        baseline_eval_json=None,
    )

    assert row["step"] == "25"
    assert row["baseline_missing"] is True
    assert abs(row["rlsd_clip_total_mean"] - 0.3) < 1e-9
    assert row["hard_gate_violation"] == "rlsd_clip_total"
    assert row["ProxyScore"] > 0


def test_hparam_summary_flags_pi_leak(tmp_path):
    metrics_jsonl = tmp_path / "metrics.jsonl"
    metrics_jsonl.write_text(
        json.dumps({"step": 1, "data": {"privacy/student_prompt_pi_leak_count": 1}}) + "\n",
        encoding="utf-8",
    )

    summary = hparam_matrix.summarize_metrics(hparam_matrix.load_metric_records(metrics_jsonl))

    assert summary["privacy_leak_count"] == 1
    assert "pi_leak" in summary["hard_gate_violation"]
