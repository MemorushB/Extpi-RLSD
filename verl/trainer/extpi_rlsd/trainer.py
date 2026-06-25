"""Legacy PPO trainer extensions for ExtPI-RLSD."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from omegaconf import OmegaConf

from verl import DataProto
from verl.trainer.extpi_rlsd.prompt_assembly import assert_no_pi_in_student_prompt, build_pi_teacher_prompt
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
            assert_no_pi_in_student_prompt(_raw_prompt_to_text(raw_prompts[row_idx]) or "", str(trace))
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
    }
    return PiTeacherBatchBuild(batch=score_batch, metrics=metrics, prompts=prompts)


class ExtPIRLSDRayPPOTrainer(RayPPOTrainer):
    """Legacy PPO trainer that injects PI teacher logprobs for RLSD updates."""

    def __init__(self, *args, **kwargs) -> None:
        if _RAY_TRAINER_IMPORT_ERROR is not None:
            raise ModuleNotFoundError(
                "ExtPIRLSDRayPPOTrainer requires the full verl PPO trainer dependencies"
            ) from _RAY_TRAINER_IMPORT_ERROR
        super().__init__(*args, **kwargs)

    def _before_actor_update(
        self,
        batch: DataProto,
        metrics: dict[str, Any],
        timing_raw: dict[str, Any],
    ) -> DataProto:
        del timing_raw
        policy_loss = self.config.actor_rollout_ref.actor.policy_loss
        if not bool(policy_loss.get("rlsd_enabled", False)):
            return batch

        extpi_config = self.config.get("extpi_rlsd", {})
        if OmegaConf.is_config(extpi_config):
            extpi_config = OmegaConf.to_container(extpi_config, resolve=True)
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
        metrics["extpi/teacher_pi_logp_mean"] = masked_mean(teacher_log_probs, mask).detach().item()
        metrics["extpi/teacher_pi_delta_mean"] = masked_mean(
            teacher_log_probs - batch.batch["old_log_probs"], mask
        ).detach().item()
        return batch
