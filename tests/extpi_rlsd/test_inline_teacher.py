import numpy as np
import torch
from omegaconf import OmegaConf

from verl import DataProto
from verl.trainer.extpi_rlsd.trainer import (
    ExtPIRLSDRayPPOTrainer,
    build_direct_opd_teacher_score_batch,
    build_pi_teacher_score_batch,
)


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 2
    padding_side = "left"
    truncation_side = "right"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, enable_thinking=False):
        assert tokenize is False
        suffix = "<think>" if enable_thinking else ""
        return messages[0]["content"] + suffix + ("<assistant>" if add_generation_prompt else "")

    def __call__(self, texts, add_special_tokens=False, padding=False, truncation=False):
        del add_special_tokens, padding, truncation
        return {"input_ids": [[(ord(char) % 50) + 3 for char in text] for text in texts]}


def _source_batch():
    return DataProto.from_dict(
        tensors={
            "responses": torch.tensor([[11, 12, 0], [21, 22, 23]]),
            "response_mask": torch.tensor([[1, 1, 0], [1, 1, 1]]),
            "old_log_probs": torch.zeros(2, 3),
        },
        non_tensors={
            "extra_info": np.array(
                [
                    {"id": "a", "problem": "1+1?", "qwen8b_pi_trace": "trace a"},
                    {"id": "b", "problem": "2+2?", "qwen8b_pi_trace": "trace b"},
                ],
                dtype=object,
            ),
            "raw_prompt": np.array(
                [
                    [{"role": "user", "content": "1+1?"}],
                    [{"role": "user", "content": "2+2?"}],
                ],
                dtype=object,
            ),
        },
    )


def test_pi_teacher_score_batch_preserves_original_response_ids():
    batch = _source_batch()
    built = build_pi_teacher_score_batch(
        source_batch=batch,
        tokenizer=FakeTokenizer(),
        max_prompt_length=256,
    )

    score_tensors = built.batch.batch
    assert score_tensors["responses"].tolist() == batch.batch["responses"].tolist()
    assert score_tensors["response_mask"].tolist() == batch.batch["response_mask"].tolist()
    prefix_width = score_tensors["prompts"].shape[1]
    expected_response_inputs = torch.tensor([[11, 12, 0], [21, 22, 23]])
    assert torch.equal(score_tensors["input_ids"][:, prefix_width:], expected_response_inputs)
    assert all("trace" in prompt for prompt in built.prompts)


def test_direct_opd_teacher_score_batch_uses_problem_only_prompt():
    batch = _source_batch()
    built = build_direct_opd_teacher_score_batch(
        source_batch=batch,
        tokenizer=FakeTokenizer(),
        max_prompt_length=256,
    )

    assert all("trace" not in prompt for prompt in built.prompts)
    assert built.batch.batch["responses"].tolist() == batch.batch["responses"].tolist()
    prefix_width = built.batch.batch["prompts"].shape[1]
    assert torch.equal(built.batch.batch["input_ids"][:, prefix_width:], built.batch.batch["responses"])


def test_extpi_hook_injects_teacher_pi_log_probs_from_ref_path():
    trainer = ExtPIRLSDRayPPOTrainer.__new__(ExtPIRLSDRayPPOTrainer)
    trainer.tokenizer = FakeTokenizer()
    trainer.ref_in_actor = True
    trainer.config = OmegaConf.create(
        {
            "data": {"max_prompt_length": 256},
            "actor_rollout_ref": {
                "actor": {
                    "policy_loss": {
                        "rlsd_enabled": True,
                    }
                }
            },
            "extpi_rlsd": {
                "teacher_update_mode": "base_no_adapter",
                "teacher_max_prompt_length": 256,
                "allow_teacher_prompt_truncation": True,
                "online_teacher_thinking": False,
            },
        }
    )

    def fake_compute_ref_log_prob(score_batch):
        score_tensors = score_batch.batch
        prefix_width = score_tensors["prompts"].shape[1]
        assert score_tensors["responses"].tolist() == [[11, 12, 0], [21, 22, 23]]
        assert torch.equal(score_tensors["input_ids"][:, prefix_width:], score_tensors["responses"])
        ref_log_prob = score_tensors["input_ids"][:, prefix_width:].float() / 100
        return DataProto.from_dict(tensors={"ref_log_prob": ref_log_prob})

    trainer._compute_ref_log_prob = fake_compute_ref_log_prob
    metrics = {}
    out = trainer._before_actor_update(_source_batch(), metrics=metrics, timing_raw={})

    expected = torch.tensor([[0.11, 0.12, 0.0], [0.21, 0.22, 0.23]])
    assert torch.allclose(out.batch["teacher_pi_log_probs"], expected)
    assert metrics["extpi/pi_teacher_response_tokens"] == 5.0
    assert "extpi/teacher_pi_delta_mean" in metrics


def test_direct_opd_hook_injects_teacher_log_probs_from_inline_scorer():
    trainer = ExtPIRLSDRayPPOTrainer.__new__(ExtPIRLSDRayPPOTrainer)
    trainer.tokenizer = FakeTokenizer()
    trainer.config = OmegaConf.create(
        {
            "data": {"max_prompt_length": 256},
            "actor_rollout_ref": {
                "rollout": {"n": 2},
                "actor": {
                    "policy_loss": {
                        "opd_pg_enabled": True,
                    }
                },
            },
            "extpi_rlsd": {
                "direct_opd_teacher_backend": "inline_external_hf",
                "direct_opd_teacher_model_path": "fake-teacher",
                "direct_opd_teacher_max_prompt_length": 256,
            },
        }
    )

    def fake_score_inline(build, extpi_config):
        assert extpi_config["direct_opd_teacher_model_path"] == "fake-teacher"
        prefix_width = build.batch.batch["prompts"].shape[1]
        assert torch.equal(build.batch.batch["input_ids"][:, prefix_width:], build.batch.batch["responses"])
        return build.batch.batch["responses"].float() / 10

    trainer._score_direct_opd_inline = fake_score_inline
    metrics = {}
    out = trainer._before_actor_update(_source_batch(), metrics=metrics, timing_raw={})

    expected = torch.tensor([[1.1, 1.2, 0.0], [2.1, 2.2, 2.3]])
    assert torch.allclose(out.batch["teacher_opd_log_probs"], expected)
    assert metrics["opd/teacher_response_tokens"] == 5.0
    assert metrics["train/step_teacher_tokens"] == 5.0


def test_extpi_score_batch_requires_pi_trace():
    batch = _source_batch()
    batch.non_tensor_batch["extra_info"][0] = {"id": "a", "problem": "1+1?", "qwen8b_pi_trace": None}
    try:
        build_pi_teacher_score_batch(source_batch=batch, tokenizer=FakeTokenizer(), max_prompt_length=256)
    except ValueError as exc:
        assert "qwen8b_pi_trace" in str(exc)
        return
    raise AssertionError("Expected missing PI trace to fail")
