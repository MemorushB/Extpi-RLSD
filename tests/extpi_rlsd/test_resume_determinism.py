import sys
from pathlib import Path

from tools.extpi_rlsd import build_frontier
from tools.extpi_rlsd.common import read_jsonl, write_jsonl


def _frontier_rows():
    rows = []
    for idx in range(24):
        rows.append(
            {
                "id": f"id-{idx:02d}",
                "problem": f"problem {idx}",
                "gold_answer": str(idx),
                "qwen32b_pi_trace": f"trace {idx}",
                "qwen32b_pi_verified": True,
                "p_plain": 0.5,
                "p_PI": 0.75,
                "student_completion_truncated": False,
            }
        )
    return rows


def _run_build_frontier(input_path: Path, output_dir: Path) -> list[str]:
    old_argv = sys.argv
    try:
        sys.argv = [
            "build_frontier.py",
            "--input_jsonl",
            str(input_path),
            "--output_dir",
            str(output_dir),
            "--seed",
            "123",
            "--smoke",
            "4",
            "--mvp_train",
            "8",
            "--matched_dev",
            "4",
            "--confirmation_train",
            "4",
            "--confirmation_dev",
            "4",
        ]
        build_frontier.main()
    finally:
        sys.argv = old_argv
    return [row["id"] for row in read_jsonl(output_dir / "mvp_train.jsonl")]


def test_frontier_split_seed_is_deterministic(tmp_path):
    input_path = tmp_path / "screened.jsonl"
    write_jsonl(input_path, _frontier_rows())

    first = _run_build_frontier(input_path, tmp_path / "first")
    second = _run_build_frontier(input_path, tmp_path / "second")

    assert first == second


def test_run_scripts_keep_training_data_order_stable():
    root = Path(__file__).resolve().parents[2]
    run_grpo = (root / "recipes/extpi_rlsd/scripts/run_grpo.sh").read_text(encoding="utf-8")
    run_extpi = (root / "recipes/extpi_rlsd/scripts/run_extpi_rlsd.sh").read_text(encoding="utf-8")

    assert "data.shuffle=False" in run_grpo
    assert "data.shuffle=False" in run_extpi
    assert "trainer.use_v1=False" in run_extpi
