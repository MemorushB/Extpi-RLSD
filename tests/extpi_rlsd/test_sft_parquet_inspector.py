import pandas as pd

from tools.extpi_rlsd import inspect_sft_parquet
from tools.extpi_rlsd.inspect_sft_parquet import inspect_dataframe


class FakeTokenizer:
    def apply_chat_template(
        self,
        messages,
        add_generation_prompt=False,
        tokenize=True,
        return_dict=False,
        **kwargs,
    ):
        del add_generation_prompt, tokenize, return_dict, kwargs
        tokens = [101, 102]
        for message in messages:
            if message.get("role") == "assistant":
                tokens.append(201)
            content = str(message.get("content", ""))
            tokens.extend(range(len(content.split())))
        return tokens


def _dataframe(messages):
    return pd.DataFrame([{"messages": messages, "enable_thinking": False, "extra_info": {"id": "row-1"}}])


def _patch_prompt_lengths(monkeypatch):
    monkeypatch.setattr(
        inspect_sft_parquet,
        "extract_system_prompt_and_generation",
        lambda tokenizer: ([101, 102], [201]),
    )


def test_sft_parquet_inspector_accepts_short_valid_rows(monkeypatch):
    _patch_prompt_lengths(monkeypatch)

    summary = inspect_dataframe(
        _dataframe(
            [
                {"role": "user", "content": "What is 1+1?"},
                {"role": "assistant", "content": "Reasoning boxed two"},
            ]
        ),
        tokenizer=FakeTokenizer(),
        max_length=32,
    )

    assert summary["rows"] == 1
    assert summary["invalid_row_count"] == 0
    assert summary["overlong_count"] == 0
    assert summary["pi_leak_count"] == 0
    assert summary["total_tokens"]["max"] == 9.0
    assert summary["assistant_loss_tokens"]["max"] == 3.0


def test_sft_parquet_inspector_reports_overlong_rows(monkeypatch):
    _patch_prompt_lengths(monkeypatch)

    summary = inspect_dataframe(
        _dataframe(
            [
                {"role": "user", "content": " ".join(["prompt"] * 8)},
                {"role": "assistant", "content": " ".join(["answer"] * 8)},
            ]
        ),
        tokenizer=FakeTokenizer(),
        max_length=4,
    )

    assert summary["overlong_count"] == 1
    assert summary["overlong_ids"] == ["row-1"]


def test_sft_parquet_inspector_reports_pi_leak_markers(monkeypatch):
    _patch_prompt_lengths(monkeypatch)

    summary = inspect_dataframe(
        _dataframe(
            [
                {"role": "user", "content": "qwen32b_pi_trace should not be here"},
                {"role": "assistant", "content": "answer"},
            ]
        ),
        tokenizer=FakeTokenizer(),
        max_length=32,
    )

    assert summary["pi_leak_count"] == 1
    assert summary["pi_leak_ids"] == ["row-1"]


def test_sft_parquet_inspector_matches_multiturn_prefix_accounting(monkeypatch):
    _patch_prompt_lengths(monkeypatch)

    summary = inspect_dataframe(
        _dataframe(
            [
                {"role": "user", "content": "one two three"},
                {"role": "assistant", "content": "four five"},
            ]
        ),
        tokenizer=FakeTokenizer(),
        max_length=32,
    )

    assert summary["total_tokens"]["max"] == 8.0
    assert summary["assistant_loss_tokens"]["max"] == 2.0
