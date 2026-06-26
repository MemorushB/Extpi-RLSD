"""Legacy PPO trainer extensions for ExtPI-RLSD."""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import torch
from omegaconf import OmegaConf

from verl import DataProto
from verl.trainer.extpi_rlsd.prompt_assembly import (
    assert_no_pi_in_student_prompt,
    build_pi_teacher_prompt,
)
from verl.trainer.extpi_rlsd.scorer import InlineHFTeacherScorer, TeacherScoreRequest, assert_tokenizer_compatible
from verl.utils.torch_functional import masked_mean

try:
    from verl.trainer.ppo.ray_trainer import RayPPOTrainer
except ModuleNotFoundError as exc:
    RayPPOTrainer = object
    _RAY_TRAINER_IMPORT_ERROR: ModuleNotFoundError | None = exc
else:
    _RAY_TRAINER_IMPORT_ERROR = None


@dataclass(frozen=True)
class PiTeacherBatchBuild:
    """PI teacher scoring batch plus construction diagnostics."""

    batch: DataProto
    metrics: dict[str, float]
    prompts: list[str]


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if hasattr(value, "as_py"):
        value = value.as_py()
    if isinstance(value, dict):
        return value
    raise TypeError(f"Expected extra_info dict, got {type(value)!r}")


def _raw_prompt_to_text(raw_prompt: Any) -> str | None:
    if raw_prompt is None:
        return None
    if isinstance(raw_prompt, str):
        return raw_prompt
    if isinstance(raw_prompt, list):
        parts = []
        for item in raw_prompt:
            if isinstance(item, dict):
                content = item.get("content", "")
                parts.append(content if isinstance(content, str) else str(content))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(raw_prompt)


def _tokenize_prefixes(
    tokenizer: Any,
    prompts: list[str],
    *,
    max_length: int,
    pad_token_id: int,
    allow_truncation: bool,
) -> tuple[torch.LongTensor, torch.Tensor, int]:
    encoded = tokenizer(prompts, add_special_tokens=False, padding=False, truncation=False)
    input_ids_list = encoded["input_ids"]
    truncation_side = getattr(tokenizer, "truncation_side", "right")
    padding_side = getattr(tokenizer, "padding_side", "right")
    normalized: list[list[int]] = []
    truncated = 0
    for input_ids in input_ids_list:
        ids = [int(token_id) for token_id in input_ids]
        if len(ids) > max_length:
            if not allow_truncation:
                raise ValueError(
                    f"PI teacher prompt has {len(ids)} tokens, exceeding teacher_max_prompt_length={max_length}"
                )
            truncated += 1
            ids = ids[-max_length:] if truncation_side == "left" else ids[:max_length]
        normalized.append(ids)

    batch_size = len(normalized)
    input_ids = torch.full((batch_size, max_length), fill_value=pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_length), dtype=torch.long)
    for row_idx, ids in enumerate(normalized):
        if not ids:
            raise ValueError("PI teacher prompt tokenized to an empty prefix")
        width = len(ids)
        if padding_side == "left":
            start = max_length - width
        else:
            start = 0
        input_ids[row_idx, start : start + width] = torch.tensor(ids, dtype=torch.long)
        attention_mask[row_idx, start : start + width] = 1
    return input_ids, attention_mask, truncated


def build_pi_teacher_score_batch(
    *,
    source_batch: DataProto,
    tokenizer: Any,
    max_prompt_length: int,
    allow_prompt_truncation: bool = True,
    teacher_thinking: bool = False,
) -> PiTeacherBatchBuild:
    """Build a temporary batch that scores original responses under PI prefixes."""

    for key in ("responses", "response_mask"):
        if key not in source_batch.batch:
            raise ValueError(f"source batch is missing required tensor key {key!r}")
    if "extra_info" not in source_batch.non_tensor_batch:
        raise ValueError("ExtPI-RLSD requires extra_info with problem and qwen8b_pi_trace")

    extra_infos = [_as_dict(item) for item in source_batch.non_tensor_batch["extra_info"]]
    raw_prompts = source_batch.non_tensor_batch.get("raw_prompt")
    prompts: list[str] = []
    missing_trace_ids: list[str] = []
    leak_checked_count = 0
    leak_count = 0
    for row_idx, extra in enumerate(extra_infos):
        problem = extra.get("problem")
        trace = extra.get("qwen8b_pi_trace")
        sample_id = str(extra.get("id", row_idx))
        if not problem:
            raise ValueError(f"ExtPI-RLSD sample {sample_id} is missing extra_info.problem")
        if not trace:
            missing_trace_ids.append(sample_id)
            continue
        if raw_prompts is not None:
            leak_checked_count += 1
            try:
                assert_no_pi_in_student_prompt(_raw_prompt_to_text(raw_prompts[row_idx]) or "", str(trace))
            except ValueError:
                leak_count += 1
                raise
        prompts.append(
            build_pi_teacher_prompt(
                tokenizer,
                str(problem),
                str(trace),
                enable_thinking=teacher_thinking,
            )
        )
    if missing_trace_ids:
        preview = ", ".join(missing_trace_ids[:5])
        raise ValueError(f"ExtPI-RLSD requires qwen8b_pi_trace for every sample; missing ids: {preview}")

    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = getattr(tokenizer, "eos_token_id", 0) or 0
    prefix_ids, prefix_mask, truncated = _tokenize_prefixes(
        tokenizer,
        prompts,
        max_length=max_prompt_length,
        pad_token_id=int(pad_token_id),
        allow_truncation=allow_prompt_truncation,
    )

    responses = source_batch.batch["responses"].detach().cpu().long()
    response_mask = source_batch.batch["response_mask"].detach().cpu().long()
    if prefix_ids.shape[0] != responses.shape[0]:
        raise ValueError("PI prompt batch size does not match response batch size")
    response_mask_bool = response_mask.bool()
    response_input_ids = responses.masked_fill(~response_mask_bool, int(pad_token_id))
    input_ids = torch.cat([prefix_ids, response_input_ids], dim=1)
    attention_mask = torch.cat([prefix_mask, response_mask_bool.long()], dim=1)
    position_ids = attention_mask.long().cumsum(dim=1).sub(1).clamp_min(0)

    score_batch = DataProto.from_dict(
        tensors={
            "prompts": prefix_ids,
            "responses": responses,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "response_mask": response_mask,
        }
    )
    metrics = {
        "extpi/pi_teacher_prompt_truncated_count": float(truncated),
        "extpi/pi_teacher_prompt_tokens_mean": float(prefix_mask.sum(dim=-1).float().mean().item()),
        "extpi/pi_teacher_response_tokens": float(response_mask.sum().item()),
        "privacy/student_prompt_pi_leak_checked_count": float(leak_checked_count),
        "privacy/student_prompt_pi_leak_count": float(leak_count),
    }
    return PiTeacherBatchBuild(batch=score_batch, metrics=metrics, prompts=prompts)


def build_direct_opd_teacher_score_batch(
    *,
    source_batch: DataProto,
    tokenizer: Any,
    max_prompt_length: int,
    allow_prompt_truncation: bool = True,
    teacher_thinking: bool = False,
) -> PiTeacherBatchBuild:
    """Build a temporary batch that scores original responses under direct teacher prompts."""

    for key in ("prompts", "responses", "attention_mask", "response_mask"):
        if key not in source_batch.batch:
            raise ValueError(f"source batch is missing required tensor key {key!r}")
    del teacher_thinking

    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = getattr(tokenizer, "eos_token_id", 0) or 0

    prefix_ids = source_batch.batch["prompts"].detach().cpu().long()
    prefix_width = prefix_ids.shape[1]
    full_attention_mask = source_batch.batch["attention_mask"].detach().cpu().long()
    if full_attention_mask.shape[1] < prefix_width:
        raise ValueError(
            "Direct OPD-PG requires attention_mask to include the prompt segment: "
            f"attention width {full_attention_mask.shape[1]} < prompt width {prefix_width}"
        )
    prefix_mask = full_attention_mask[:, :prefix_width]
    if prefix_mask.sum(dim=-1).eq(0).any():
        raise ValueError("Direct OPD-PG found an empty student prompt prefix")

    truncated = 0
    if prefix_width > max_prompt_length:
        if not allow_prompt_truncation:
            raise ValueError(
                f"Direct OPD-PG prompt width {prefix_width} exceeds teacher_max_prompt_length={max_prompt_length}"
            )
        truncated = prefix_ids.shape[0]
        truncation_side = getattr(tokenizer, "truncation_side", "right")
        if truncation_side == "left":
            prefix_ids = prefix_ids[:, -max_prompt_length:]
            prefix_mask = prefix_mask[:, -max_prompt_length:]
        else:
            prefix_ids = prefix_ids[:, :max_prompt_length]
            prefix_mask = prefix_mask[:, :max_prompt_length]

    responses = source_batch.batch["responses"].detach().cpu().long()
    response_mask = source_batch.batch["response_mask"].detach().cpu().long()
    response_mask_bool = response_mask.bool()
    response_input_ids = responses.masked_fill(~response_mask_bool, int(pad_token_id))
    input_ids = torch.cat([prefix_ids, response_input_ids], dim=1)
    attention_mask = torch.cat([prefix_mask, response_mask_bool.long()], dim=1)
    position_ids = attention_mask.long().cumsum(dim=1).sub(1).clamp_min(0)

    score_batch = DataProto.from_dict(
        tensors={
            "prompts": prefix_ids,
            "responses": responses,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "response_mask": response_mask,
        }
    )
    metrics = {
        "opd/teacher_prompt_truncated_count": float(truncated),
        "opd/teacher_prompt_tokens_mean": float(prefix_mask.sum(dim=-1).float().mean().item()),
        "opd/teacher_response_tokens": float(response_mask.sum().item()),
    }
    return PiTeacherBatchBuild(batch=score_batch, metrics=metrics, prompts=[])


def _group_diagnostic_metrics(
    scores: torch.Tensor,
    *,
    uids: Any | None,
    rollout_n: int,
) -> tuple[dict[str, float], bool, bool]:
    if uids is not None:
        if len(uids) != scores.shape[0]:
            return {"train/group_metric_uid_mismatch": 1.0}, True, False
        uid_to_scores: dict[str, list[float]] = defaultdict(list)
        for uid, score in zip(uids, scores.detach().cpu().tolist(), strict=True):
            uid_to_scores[str(uid)].append(float(score))
        groups = [torch.tensor(values, dtype=torch.float32) for values in uid_to_scores.values() if len(values) >= 2]
        if not groups:
            return {}, True, True
        group_std = torch.tensor([group.std(unbiased=False).item() for group in groups], dtype=torch.float32)
        all_correct = torch.tensor([(group >= 1.0).all().item() for group in groups], dtype=torch.float32)
        all_wrong = torch.tensor([(group < 1.0).all().item() for group in groups], dtype=torch.float32)
        return {
            "train/group_nonzero_std_ratio": float((group_std > 0).float().mean().item()),
            "train/group_all_correct_ratio": float(all_correct.mean().item()),
            "train/group_all_wrong_ratio": float(all_wrong.mean().item()),
        }, True, True

    if rollout_n <= 0:
        return {}, False, False
    usable = (scores.shape[0] // rollout_n) * rollout_n
    if usable == 0:
        return {}, False, False
    grouped = scores[:usable].view(-1, rollout_n)
    group_std = grouped.std(dim=-1, unbiased=False)
    correct = grouped >= 1.0
    return {
        "train/group_nonzero_std_ratio": float((group_std > 0).float().mean().detach().item()),
        "train/group_all_correct_ratio": float(correct.all(dim=-1).float().mean().detach().item()),
        "train/group_all_wrong_ratio": float((~correct).all(dim=-1).float().mean().detach().item()),
    }, False, True


class ExtPIRLSDRayPPOTrainer(RayPPOTrainer):
    """Legacy PPO trainer that injects PI teacher logprobs for RLSD updates."""

    def __init__(self, *args, **kwargs) -> None:
        if _RAY_TRAINER_IMPORT_ERROR is not None:
            raise ModuleNotFoundError(
                "ExtPIRLSDRayPPOTrainer requires the full verl PPO trainer dependencies"
            ) from _RAY_TRAINER_IMPORT_ERROR
        super().__init__(*args, **kwargs)

    def _add_training_diagnostics(self, batch: DataProto, metrics: dict[str, Any]) -> None:
        if "response_mask" in batch.batch:
            response_mask = batch.batch["response_mask"].bool()
            response_lengths = response_mask.sum(dim=-1).float()
            metrics["train/step_student_tokens"] = float(response_mask.sum().detach().item())
            metrics["train/response_length_mean"] = response_lengths.mean().detach().item()
            metrics["train/truncation_proxy_last_token_ratio"] = response_mask[:, -1].float().mean().detach().item()
        if "token_level_scores" not in batch.batch or "response_mask" not in batch.batch:
            return
        rollout_n = int(self.config.actor_rollout_ref.rollout.n)
        scores = (batch.batch["token_level_scores"] * batch.batch["response_mask"]).sum(dim=-1).float()
        uids = batch.non_tensor_batch.get("uid") if batch.non_tensor_batch is not None else None
        group_metrics, uid_seen, metric_ready = _group_diagnostic_metrics(scores, uids=uids, rollout_n=rollout_n)
        metrics.update(group_metrics)
        if metric_ready:
            metrics["train/group_metrics_uid_based"] = 1.0 if uid_seen else 0.0

    def _get_direct_opd_scorer(self, extpi_config: dict[str, Any]) -> InlineHFTeacherScorer:
        scorer = getattr(self, "_direct_opd_scorer", None)
        if scorer is not None:
            return scorer
        teacher_model = extpi_config.get("direct_opd_teacher_model_path")
        if not teacher_model:
            raise ValueError("direct_opd_teacher_model_path is required when direct_opd_enabled=True")
        device = extpi_config.get("direct_opd_teacher_device", "cuda")
        if str(device).startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                "inline_external_hf requires CUDA visible in the trainer actor. "
                "Set RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1 or use the multi-GPU teacher-pool entrypoint."
            )
        dtype_name = str(extpi_config.get("direct_opd_teacher_dtype", "bfloat16"))
        dtype = getattr(torch, dtype_name)
        load_before = (
            torch.cuda.memory_allocated() / (1024**3)
            if str(device).startswith("cuda") and torch.cuda.is_available()
            else 0.0
        )
        scorer = InlineHFTeacherScorer(
            model_path=str(teacher_model),
            dtype=dtype,
            device=device,
            trust_remote_code=bool(extpi_config.get("trust_remote_code", True)),
            attn_implementation=str(extpi_config.get("direct_opd_attn_implementation", "flash_attention_2")),
        )
        assert_tokenizer_compatible(self.tokenizer, scorer.tokenizer)
        load_after = (
            torch.cuda.memory_allocated() / (1024**3)
            if str(device).startswith("cuda") and torch.cuda.is_available()
            else 0.0
        )
        self._direct_opd_inline_teacher_load_gb = max(0.0, load_after - load_before)
        self._direct_opd_scorer = scorer
        return scorer

    def _score_direct_opd_inline(self, build: PiTeacherBatchBuild, extpi_config: dict[str, Any]) -> torch.Tensor:
        scorer = self._get_direct_opd_scorer(extpi_config)
        tensors = build.batch.batch
        prefix_width = tensors["prompts"].shape[1]
        micro_batch_size = int(extpi_config.get("direct_opd_teacher_micro_batch_size", 1))
        outputs = []
        device = str(extpi_config.get("direct_opd_teacher_device", "cuda"))
        track_cuda = device.startswith("cuda") and torch.cuda.is_available()
        if track_cuda:
            torch.cuda.reset_peak_memory_stats()
        score_start = time.perf_counter()
        for start in range(0, tensors["responses"].shape[0], micro_batch_size):
            end = start + micro_batch_size
            scored = scorer.score(
                TeacherScoreRequest(
                    prompt_input_ids=tensors["prompts"][start:end],
                    prompt_attention_mask=tensors["attention_mask"][start:end, :prefix_width],
                    response_ids=tensors["responses"][start:end],
                    response_mask=tensors["response_mask"][start:end],
                    temperature=float(extpi_config.get("direct_opd_teacher_temperature", 1.0)),
                )
            )
            outputs.append(scored.response_log_probs.detach().cpu())
        self._direct_opd_inline_teacher_score_time = time.perf_counter() - score_start
        self._direct_opd_inline_teacher_max_memory_allocated_gb = (
            torch.cuda.max_memory_allocated() / (1024**3) if track_cuda else 0.0
        )
        return torch.cat(outputs, dim=0)

    def _before_actor_update(
        self,
        batch: DataProto,
        metrics: dict[str, Any],
        timing_raw: dict[str, Any],
    ) -> DataProto:
        del timing_raw
        policy_loss = self.config.actor_rollout_ref.actor.policy_loss
        rlsd_enabled = bool(policy_loss.get("rlsd_enabled", False))
        opd_pg_enabled = bool(policy_loss.get("opd_pg_enabled", False))
        if rlsd_enabled and opd_pg_enabled:
            raise ValueError("RLSD and direct OPD-PG modes are mutually exclusive")
        self._add_training_diagnostics(batch, metrics)
        if not rlsd_enabled and not opd_pg_enabled:
            return batch

        extpi_config = self.config.get("extpi_rlsd", {})
        if OmegaConf.is_config(extpi_config):
            extpi_config = OmegaConf.to_container(extpi_config, resolve=True)
        if opd_pg_enabled:
            backend = extpi_config.get("direct_opd_teacher_backend", "inline_external_hf")
            if backend != "inline_external_hf":
                raise NotImplementedError(
                    "Direct OPD-PG MVP only supports direct_opd_teacher_backend=inline_external_hf"
                )
            max_prompt_length = int(
                extpi_config.get("direct_opd_teacher_max_prompt_length", self.config.data.max_prompt_length)
            )
            allow_truncation = bool(extpi_config.get("allow_teacher_prompt_truncation", True))
            teacher_thinking = bool(extpi_config.get("direct_opd_teacher_thinking", False))
            build = build_direct_opd_teacher_score_batch(
                source_batch=batch,
                tokenizer=self.tokenizer,
                max_prompt_length=max_prompt_length,
                allow_prompt_truncation=allow_truncation,
                teacher_thinking=teacher_thinking,
            )
            teacher_log_probs = self._score_direct_opd_inline(build, extpi_config).to(
                device=batch.batch["responses"].device,
                dtype=batch.batch["old_log_probs"].dtype,
            )
            if teacher_log_probs.shape != batch.batch["response_mask"].shape:
                raise ValueError(
                    "teacher_opd_log_probs shape mismatch: "
                    f"{tuple(teacher_log_probs.shape)} vs response_mask {tuple(batch.batch['response_mask'].shape)}"
                )
            batch.batch["teacher_opd_log_probs"] = teacher_log_probs * batch.batch["response_mask"].to(
                teacher_log_probs
            )
            mask = batch.batch["response_mask"].bool()
            metrics.update(build.metrics)
            metrics["train/step_teacher_tokens"] = metrics["opd/teacher_response_tokens"]
            metrics["opd/teacher_logp_mean"] = masked_mean(teacher_log_probs, mask).detach().item()
            metrics["opd/teacher_delta_mean"] = masked_mean(
                teacher_log_probs - batch.batch["old_log_probs"], mask
            ).detach().item()
            metrics["opd/inline_teacher_load_gb"] = float(getattr(self, "_direct_opd_inline_teacher_load_gb", 0.0))
            metrics["opd/inline_teacher_score_time"] = float(
                getattr(self, "_direct_opd_inline_teacher_score_time", 0.0)
            )
            metrics["opd/inline_teacher_max_memory_allocated_gb"] = float(
                getattr(self, "_direct_opd_inline_teacher_max_memory_allocated_gb", 0.0)
            )
            return batch

        teacher_update_mode = extpi_config.get("teacher_update_mode", "base_no_adapter")
        if teacher_update_mode != "base_no_adapter":
            raise NotImplementedError("ExtPI-RLSD MVP only supports teacher_update_mode=base_no_adapter")
        if not self.ref_in_actor:
            raise ValueError("teacher_update_mode=base_no_adapter requires a LoRA actor so the base model is in actor")

        max_prompt_length = int(extpi_config.get("teacher_max_prompt_length", self.config.data.max_prompt_length))
        allow_truncation = bool(extpi_config.get("allow_teacher_prompt_truncation", True))
        teacher_thinking = bool(extpi_config.get("online_teacher_thinking", False))
        build = build_pi_teacher_score_batch(
            source_batch=batch,
            tokenizer=self.tokenizer,
            max_prompt_length=max_prompt_length,
            allow_prompt_truncation=allow_truncation,
            teacher_thinking=teacher_thinking,
        )
        teacher_ref = self._compute_ref_log_prob(build.batch)
        teacher_log_probs = teacher_ref.batch["ref_log_prob"].detach().to(
            device=batch.batch["responses"].device,
            dtype=batch.batch["old_log_probs"].dtype,
        )
        if teacher_log_probs.shape != batch.batch["response_mask"].shape:
            raise ValueError(
                "teacher_pi_log_probs shape mismatch: "
                f"{tuple(teacher_log_probs.shape)} vs response_mask {tuple(batch.batch['response_mask'].shape)}"
            )
        batch.batch["teacher_pi_log_probs"] = teacher_log_probs * batch.batch["response_mask"].to(teacher_log_probs)

        mask = batch.batch["response_mask"].bool()
        metrics.update(build.metrics)
        metrics["train/step_teacher_tokens"] = metrics["extpi/pi_teacher_response_tokens"]
        metrics["extpi/teacher_pi_logp_mean"] = masked_mean(teacher_log_probs, mask).detach().item()
        metrics["extpi/teacher_pi_delta_mean"] = masked_mean(
            teacher_log_probs - batch.batch["old_log_probs"], mask
        ).detach().item()
        return batch
