import numpy as np
import torch
from omegaconf import OmegaConf

from verl import DataProto
from verl.trainer.extpi_rlsd.trainer import (
    ExtPIRLSDRayPPOTrainer,
    _build_rlsd_token_mask,
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
            "prompts": torch.tensor([[0, 0, 101, 102], [0, 201, 202, 203]]),
            "responses": torch.tensor([[11, 12, 0], [21, 22, 23]]),
            "attention_mask": torch.tensor([[0, 0, 1, 1, 1, 1, 0], [0, 1, 1, 1, 1, 1, 1]]),
            "response_mask": torch.tensor([[1, 1, 0], [1, 1, 1]]),
            "old_log_probs": torch.zeros(2, 3),
        },
        non_tensors={
            "extra_info": np.array(
                [
                    {"id": "a", "problem": "1+1?", "qwen32b_pi_trace": "trace a"},
                    {"id": "b", "problem": "2+2?", "qwen32b_pi_trace": "trace b"},
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


def test_pi_teacher_score_batch_supports_explicit_legacy_trace_field():
    batch = _source_batch()
    batch.non_tensor_batch["extra_info"] = np.array(
        [
            {"id": "a", "problem": "1+1?", "qwen8b_pi_trace": "legacy trace a"},
            {"id": "b", "problem": "2+2?", "qwen8b_pi_trace": "legacy trace b"},
        ],
        dtype=object,
    )

    built = build_pi_teacher_score_batch(
        source_batch=batch,
        tokenizer=FakeTokenizer(),
        max_prompt_length=256,
        pi_trace_field="qwen8b_pi_trace",
    )

    assert all("legacy trace" in prompt for prompt in built.prompts)


def test_pi_teacher_prefixes_are_left_padded_even_if_tokenizer_default_is_right():
    class RightPadTokenizer(FakeTokenizer):
        padding_side = "right"

    built = build_pi_teacher_score_batch(
        source_batch=_source_batch(),
        tokenizer=RightPadTokenizer(),
        max_prompt_length=128,
    )

    prefix_mask = built.batch.batch["attention_mask"][:, :128].bool()
    assert prefix_mask[:, -1].all()
    for row in prefix_mask:
        first_real = int(row.nonzero()[0].item())
        assert not row[:first_real].any()
        assert row[first_real:].all()


def test_rlsd_token_mask_4k_plus_answer_uses_prefix_and_tail_without_padding():
    response_mask = torch.tensor([[1] * 5000 + [0] * 2])

    token_mask = _build_rlsd_token_mask(response_mask, "4k+ans")

    assert token_mask[0, 0].item() == 1
    assert token_mask[0, 4095].item() == 1
    assert token_mask[0, 4096].item() == 0
    assert token_mask[0, 4743].item() == 0
    assert token_mask[0, 4744].item() == 1
    assert token_mask[0, 4999].item() == 1
    assert token_mask[0, 5000].item() == 0


def test_direct_opd_teacher_score_batch_reuses_student_prompt_tokens():
    batch = _source_batch()
    batch.non_tensor_batch["extra_info"][0]["problem"] = "template drift should not be retokenized"
    built = build_direct_opd_teacher_score_batch(
        source_batch=batch,
        tokenizer=FakeTokenizer(),
        max_prompt_length=256,
    )

    assert built.prompts == []
    assert torch.equal(built.batch.batch["prompts"], batch.batch["prompts"])
    assert torch.equal(built.batch.batch["attention_mask"][:, :4], batch.batch["attention_mask"][:, :4])
    assert built.batch.batch["responses"].tolist() == batch.batch["responses"].tolist()
    prefix_width = built.batch.batch["prompts"].shape[1]
    assert torch.equal(built.batch.batch["input_ids"][:, prefix_width:], built.batch.batch["responses"])


def test_direct_opd_rejects_right_padded_prefix():
    batch = _source_batch()
    batch.batch["prompts"] = torch.tensor([[101, 102, 0, 0], [201, 202, 203, 0]])
    batch.batch["attention_mask"][:, :4] = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]])

    try:
        build_direct_opd_teacher_score_batch(
            source_batch=batch,
            tokenizer=FakeTokenizer(),
            max_prompt_length=256,
        )
    except ValueError as exc:
        assert "left-padded" in str(exc)
        return
    raise AssertionError("Expected right-padded direct OPD prefix to fail")


def test_direct_opd_rejects_truncated_prefix():
    batch = _source_batch()

    try:
        build_direct_opd_teacher_score_batch(
            source_batch=batch,
            tokenizer=FakeTokenizer(),
            max_prompt_length=3,
            allow_prompt_truncation=True,
        )
    except ValueError as exc:
        assert "cannot truncate" in str(exc)
        return
    raise AssertionError("Expected direct OPD prompt truncation to fail")


def test_direct_opd_rejects_thinking_mode_with_exact_prompt_reuse():
    batch = _source_batch()

    try:
        build_direct_opd_teacher_score_batch(
            source_batch=batch,
            tokenizer=FakeTokenizer(),
            max_prompt_length=256,
            teacher_thinking=True,
        )
    except ValueError as exc:
        assert "direct_opd_teacher_thinking=True" in str(exc)
        return
    raise AssertionError("Expected direct OPD thinking mode to fail")


def test_direct_opd_teacher_score_batch_requires_student_prompt_tokens():
    batch = _source_batch()
    del batch.batch["prompts"]

    try:
        build_direct_opd_teacher_score_batch(
            source_batch=batch,
            tokenizer=FakeTokenizer(),
            max_prompt_length=256,
        )
    except ValueError as exc:
        assert "prompts" in str(exc)
        return
    raise AssertionError("Expected missing student prompt tokens to fail")


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
    assert metrics["privacy/student_prompt_pi_leak_checked_count"] == 2.0
    assert metrics["privacy/student_prompt_pi_leak_count"] == 0.0


def test_extpi_current_no_grad_uses_current_actor_adapter():
    trainer = ExtPIRLSDRayPPOTrainer.__new__(ExtPIRLSDRayPPOTrainer)
    trainer.tokenizer = FakeTokenizer()
    trainer.ref_in_actor = True
    trainer.global_steps = 3
    trainer.config = OmegaConf.create(
        {
            "data": {"max_prompt_length": 256},
            "actor_rollout_ref": {
                "model": {"extpi_teacher_adapter": False},
                "rollout": {"temperature": 1.0, "n": 2},
                "actor": {
                    "policy_loss": {
                        "rlsd_enabled": True,
                        "rlsd_lambda": 0.5,
                        "rlsd_lambda_warmup_steps": 0,
                        "rlsd_lambda_decay_steps": 0,
                    }
                },
            },
            "extpi_rlsd": {
                "teacher_update_mode": "current_no_grad",
                "teacher_max_prompt_length": 256,
                "allow_teacher_prompt_truncation": True,
                "online_teacher_thinking": False,
            },
        }
    )
    captured = {}

    def fake_compute_actor_log_prob(score_batch, *, no_lora_adapter=False, lora_adapter_name=None):
        captured["no_lora_adapter"] = no_lora_adapter
        captured["lora_adapter_name"] = lora_adapter_name
        return DataProto.from_dict(tensors={"ref_log_prob": score_batch.batch["responses"].float() / 100})

    trainer._compute_actor_log_prob = fake_compute_actor_log_prob
    metrics = {}
    out = trainer._before_actor_update(_source_batch(), metrics=metrics, timing_raw={})

    assert captured == {"no_lora_adapter": False, "lora_adapter_name": None}
    assert torch.allclose(out.batch["teacher_pi_log_probs"], torch.tensor([[0.11, 0.12, 0.0], [0.21, 0.22, 0.23]]))
    assert out.meta_info["rlsd_effective_lambda"] == 0.5


def test_extpi_periodic_snapshot_syncs_and_uses_teacher_adapter():
    class FakeActorRolloutWG:
        def __init__(self):
            self.calls = []

        def sync_extpi_teacher_snapshot(self, **kwargs):
            self.calls.append(kwargs)

    trainer = ExtPIRLSDRayPPOTrainer.__new__(ExtPIRLSDRayPPOTrainer)
    trainer.tokenizer = FakeTokenizer()
    trainer.ref_in_actor = True
    trainer.global_steps = 1
    trainer._extpi_teacher_last_sync_step = None
    trainer.actor_rollout_wg = FakeActorRolloutWG()
    trainer.config = OmegaConf.create(
        {
            "data": {"max_prompt_length": 256},
            "actor_rollout_ref": {
                "model": {"extpi_teacher_adapter": True},
                "rollout": {"temperature": 1.0, "n": 2},
                "actor": {
                    "policy_loss": {
                        "rlsd_enabled": True,
                        "rlsd_lambda": 0.5,
                        "rlsd_lambda_warmup_steps": 0,
                        "rlsd_lambda_decay_steps": 0,
                    }
                },
            },
            "extpi_rlsd": {
                "teacher_update_mode": "periodic_snapshot",
                "teacher_sync_interval": 20,
                "teacher_max_prompt_length": 256,
                "allow_teacher_prompt_truncation": True,
                "online_teacher_thinking": False,
            },
        }
    )
    captured = {}

    def fake_compute_actor_log_prob(score_batch, *, no_lora_adapter=False, lora_adapter_name=None):
        del no_lora_adapter
        captured["lora_adapter_name"] = lora_adapter_name
        return DataProto.from_dict(tensors={"ref_log_prob": score_batch.batch["responses"].float() / 100})

    trainer._compute_actor_log_prob = fake_compute_actor_log_prob
    metrics = {}
    trainer._before_actor_update(_source_batch(), metrics=metrics, timing_raw={})

    assert captured["lora_adapter_name"] == "extpi_teacher"
    assert trainer.actor_rollout_wg.calls == [{"source_adapter": "default", "target_adapter": "extpi_teacher"}]
    assert metrics["extpi/teacher_snapshot_synced"] == 1.0
    assert metrics["extpi/teacher_snapshot_step"] == 1.0
    assert metrics["extpi/teacher_sync_interval"] == 20.0


def test_extpi_periodic_snapshot_rejects_non_positive_sync_interval():
    trainer = ExtPIRLSDRayPPOTrainer.__new__(ExtPIRLSDRayPPOTrainer)
    trainer.tokenizer = FakeTokenizer()
    trainer.ref_in_actor = True
    trainer.global_steps = 1
    trainer._extpi_teacher_last_sync_step = None
    trainer.config = OmegaConf.create(
        {
            "data": {"max_prompt_length": 256},
            "actor_rollout_ref": {
                "model": {"extpi_teacher_adapter": True},
                "rollout": {"temperature": 1.0, "n": 2},
                "actor": {"policy_loss": {"rlsd_enabled": True}},
            },
            "extpi_rlsd": {
                "teacher_update_mode": "periodic_snapshot",
                "teacher_sync_interval": 0,
                "teacher_max_prompt_length": 256,
            },
        }
    )

    try:
        trainer._before_actor_update(_source_batch(), metrics={}, timing_raw={})
    except ValueError as exc:
        assert "teacher_sync_interval" in str(exc)
        return
    raise AssertionError("Expected periodic_snapshot with sync interval 0 to fail")


def test_extpi_effective_lambda_schedule_matches_warmup_and_decay():
    trainer = ExtPIRLSDRayPPOTrainer.__new__(ExtPIRLSDRayPPOTrainer)
    policy_loss = {
        "rlsd_lambda": 0.5,
        "rlsd_lambda_warmup_steps": 10,
        "rlsd_lambda_decay_steps": 100,
    }

    trainer.global_steps = 5
    assert trainer._compute_effective_rlsd_lambda(policy_loss) == 0.25
    trainer.global_steps = 10
    assert trainer._compute_effective_rlsd_lambda(policy_loss) == 0.5
    trainer.global_steps = 60
    assert trainer._compute_effective_rlsd_lambda(policy_loss) == 0.25
    trainer.global_steps = 110
    assert trainer._compute_effective_rlsd_lambda(policy_loss) == 0.0


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
        assert torch.equal(build.batch.batch["prompts"], torch.tensor([[0, 0, 101, 102], [0, 201, 202, 203]]))
        assert torch.equal(build.batch.batch["input_ids"][:, prefix_width:], build.batch.batch["responses"])
        return build.batch.batch["responses"].float() / 10

    trainer._score_direct_opd_inline = fake_score_inline
    metrics = {}
    out = trainer._before_actor_update(_source_batch(), metrics=metrics, timing_raw={})

    expected = torch.tensor([[1.1, 1.2, 0.0], [2.1, 2.2, 2.3]])
    assert torch.allclose(out.batch["teacher_opd_log_probs"], expected)
    assert metrics["opd/teacher_response_tokens"] == 5.0
    assert metrics["train/step_teacher_tokens"] == 5.0
    assert "opd/inline_teacher_score_time" in metrics
    assert "opd/inline_teacher_memory_allocated_gb" in metrics
    assert "opd/inline_teacher_memory_reserved_gb" in metrics


def test_extpi_score_batch_requires_pi_trace():
    batch = _source_batch()
    batch.non_tensor_batch["extra_info"][0] = {"id": "a", "problem": "1+1?", "qwen32b_pi_trace": None}
    try:
        build_pi_teacher_score_batch(source_batch=batch, tokenizer=FakeTokenizer(), max_prompt_length=256)
    except ValueError as exc:
        assert "qwen32b_pi_trace" in str(exc)
        return
    raise AssertionError("Expected missing PI trace to fail")


def test_training_group_diagnostics_use_uid_after_batch_reorder():
    trainer = ExtPIRLSDRayPPOTrainer.__new__(ExtPIRLSDRayPPOTrainer)
    trainer.config = OmegaConf.create({"actor_rollout_ref": {"rollout": {"n": 4}}})
    batch = DataProto.from_dict(
        tensors={
            "response_mask": torch.ones(8, 1),
            "token_level_scores": torch.tensor([[1.0], [1.0], [0.0], [1.0], [1.0], [1.0], [0.0], [1.0]]),
        },
        non_tensors={"uid": np.array(["a", "b", "a", "b", "a", "b", "a", "b"], dtype=object)},
    )

    metrics = {}
    trainer._add_training_diagnostics(batch, metrics)

    assert metrics["train/group_metrics_uid_based"] == 1.0
    assert metrics["train/group_nonzero_std_ratio"] == 0.5
    assert metrics["train/group_all_correct_ratio"] == 0.5
    assert metrics["train/group_all_wrong_ratio"] == 0.0
