import json
import sys

from tools.extpi_rlsd import prepare_opsd_data
from tools.extpi_rlsd.common import read_jsonl, write_jsonl


def test_prepare_opsd_skips_rows_without_gold_answer(tmp_path):
    input_path = tmp_path / "source.jsonl"
    output_dir = tmp_path / "out"
    write_jsonl(
        input_path,
        [
            {"problem": "What is 1+1?", "Answer": "2", "solution": "\\boxed{2}"},
            {"problem": "Prove a theorem.", "Answer": "", "solution": "proof"},
        ],
    )

    old_argv = sys.argv
    try:
        sys.argv = [
            "prepare_opsd_data.py",
            "--input_jsonl",
            str(input_path),
            "--output_dir",
            str(output_dir),
        ]
        prepare_opsd_data.main()
    finally:
        sys.argv = old_argv

    clean = read_jsonl(output_dir / "all_clean.jsonl")
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))

    assert [row["gold_answer"] for row in clean] == ["2"]
    assert manifest["stats"]["clean"] == 1
    assert manifest["stats"]["invalid_missing_answer"] == 1
