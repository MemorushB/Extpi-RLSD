import json
import sys

import pandas as pd

from tools.extpi_rlsd import build_closed_sft_parquet, build_teacher_sft_parquet
from tools.extpi_rlsd.common import write_jsonl


def test_build_closed_sft_parquet_keeps_only_verified_rows(tmp_path):
    input_path = tmp_path / "closed.jsonl"
    output_path = tmp_path / "train.parquet"
    manifest_path = tmp_path / "manifest.json"
    write_jsonl(
        input_path,
        [
            {
                "id": "ok",
                "problem": "What is 1+1?",
                "gold_answer": "2",
                "closed_teacher_output": "Reasoning. \\boxed{2}",
                "closed_teacher_verified": True,
            },
            {
                "id": "bad",
                "problem": "What is 2+2?",
                "gold_answer": "4",
                "closed_teacher_output": "Wrong. \\boxed{5}",
                "closed_teacher_verified": False,
            },
        ],
    )

    old_argv = sys.argv
    try:
        sys.argv = [
            "build_closed_sft_parquet.py",
            "--input_jsonl",
            str(input_path),
            "--output_parquet",
            str(output_path),
            "--manifest_json",
            str(manifest_path),
        ]
        build_closed_sft_parquet.main()
    finally:
        sys.argv = old_argv

    dataframe = pd.read_parquet(output_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert len(dataframe) == 1
    assert dataframe.iloc[0]["messages"][0]["role"] == "user"
    assert dataframe.iloc[0]["messages"][1]["role"] == "assistant"
    assert bool(dataframe.iloc[0]["enable_thinking"]) is False
    assert manifest["written_count"] == 1
    assert manifest["verified_source_count"] == 1


def test_build_teacher_sft_parquet_uses_generic_response_field(tmp_path):
    input_path = tmp_path / "teacher.jsonl"
    output_path = tmp_path / "train.parquet"
    manifest_path = tmp_path / "manifest.json"
    write_jsonl(
        input_path,
        [
            {
                "id": "ok",
                "problem": "What is 3+4?",
                "gold_answer": "7",
                "response": "Reasoning. \\boxed{7}",
                "verified": True,
            },
            {
                "id": "skip",
                "problem": "What is 2+2?",
                "gold_answer": "4",
                "response": "Wrong. \\boxed{5}",
                "verified": False,
            },
        ],
    )

    old_argv = sys.argv
    try:
        sys.argv = [
            "build_teacher_sft_parquet.py",
            "--input_jsonl",
            str(input_path),
            "--output_parquet",
            str(output_path),
            "--manifest_json",
            str(manifest_path),
            "--response_field",
            "response",
            "--require_verified",
            "true",
        ]
        build_teacher_sft_parquet.main()
    finally:
        sys.argv = old_argv

    dataframe = pd.read_parquet(output_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert len(dataframe) == 1
    assert dataframe.iloc[0]["messages"][1]["content"] == "Reasoning. \\boxed{7}"
    assert manifest["written_count"] == 1
    assert manifest["dropped_count"] == 1
    assert manifest["response_field_counts"] == {"response": 1}
