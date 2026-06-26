from tools.extpi_rlsd.prepare_official_math_eval import DATASET_SPECS, build_prompt, normalize_example


def test_normalize_aime24_example_uses_opsd_prompt_and_eval_schema():
    row = normalize_example(
        dataset_name="aime24",
        spec=DATASET_SPECS["aime24"],
        example={"id": "2024-I-1", "problem": "What is 1+1?", "answer": "2", "solution": "Add."},
        index=0,
        prompt_style="opsd",
    )

    assert row["id"] == "aime24:2024-I-1"
    assert row["gold_answer"] == "2"
    assert row["reward_model"]["ground_truth"] == "2"
    assert row["prompt"][0]["role"] == "user"
    assert "Please reason step by step" in row["prompt"][0]["content"]
    assert row["extra_info"]["dataset"] == "aime24"
    assert row["extra_info"]["source_dataset"] == "HuggingFaceH4/aime_2024"


def test_build_prompt_supports_extpi_style():
    prompt = build_prompt("What is 2+2?", "extpi")

    assert "Solve the following mathematics problem" in prompt
    assert "What is 2+2?" in prompt
