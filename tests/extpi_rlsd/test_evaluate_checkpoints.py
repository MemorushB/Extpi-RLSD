from tools.extpi_rlsd.common import write_jsonl
from tools.extpi_rlsd.evaluate_checkpoints import load_eval_items, parse_seeds, summarize_scored_items


def test_summarize_scored_items_outputs_aggregator_fields():
    payload = summarize_scored_items(
        run="method",
        checkpoint="step_1",
        scored_items=[
            {
                "id": "a",
                "ground_truth": "2",
                "greedy": {"text": "\\boxed{2}", "completion_tokens": 3},
                "samples": [
                    {"text": "\\boxed{2}", "completion_tokens": 3, "prompt_tokens": 5},
                    {"text": "\\boxed{3}", "completion_tokens": 3, "prompt_tokens": 5},
                ],
            },
            {
                "id": "b",
                "ground_truth": "4",
                "greedy": {"text": "\\boxed{5}", "completion_tokens": 3},
                "samples": [
                    {"text": "\\boxed{4}", "completion_tokens": 3, "prompt_tokens": 5},
                    {"text": "no box", "completion_tokens": 2, "prompt_tokens": 5},
                ],
            },
        ],
    )

    assert payload["run"] == "method"
    assert payload["checkpoint"] == "step_1"
    assert payload["avg"] == 0.5
    assert payload["pass"] == 1.0
    assert payload["greedy_accuracy"] == 0.5
    assert payload["format_rate"] == 0.75
    assert payload["total_samples"] == 4
    assert "group_variance" in payload


def test_parse_seeds_extends_and_truncates_to_num_samples():
    assert parse_seeds("7,8", 4) == [7, 8, 9, 10]
    assert parse_seeds("1,2,3", 2) == [1, 2]


def test_load_eval_items_uses_qwen32b_trace_by_default(tmp_path):
    path = tmp_path / "eval.jsonl"
    write_jsonl(
        path,
        [
            {
                "id": "a",
                "problem": "1+1?",
                "gold_answer": "2",
                "qwen32b_pi_trace": "new trace",
                "qwen8b_pi_trace": "old trace",
            }
        ],
    )

    item = load_eval_items(path)[0]

    assert item.pi_trace == "new trace"


def test_load_eval_items_supports_legacy_trace_override(tmp_path):
    path = tmp_path / "eval.jsonl"
    write_jsonl(
        path,
        [{"id": "a", "problem": "1+1?", "gold_answer": "2", "qwen8b_pi_trace": "old trace"}],
    )

    item = load_eval_items(path, pi_trace_field="qwen8b_pi_trace")[0]

    assert item.pi_trace == "old trace"
