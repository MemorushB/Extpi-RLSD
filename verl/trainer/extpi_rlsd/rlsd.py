"""Token-level ExtPI-RLSD and sampled-token OPD-PG math."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class RLSDConfig:
    """Configuration for RLSD advantage reweighting."""

    lam: float = 0.5
    clip_range: float = 0.2
    negative_only: bool = False

    def validate(self) -> None:
        if not 0.0 <= self.lam <= 1.0:
            raise ValueError(f"lam must be in [0, 1], got {self.lam}")
        if not 0.0 <= self.clip_range < 1.0:
            raise ValueError(f"clip_range must be in [0, 1), got {self.clip_range}")


def _as_mask(response_mask: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    return response_mask.to(device=like.device, dtype=like.dtype)


def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(dtype=value.dtype, device=value.device)
    denom = mask.sum().clamp_min(1.0)
    masked_value = torch.where(mask.bool(), value, torch.zeros_like(value))
    return masked_value.sum() / denom


def _masked_std(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mean = _masked_mean(value, mask)
    return torch.sqrt(_masked_mean((value - mean) ** 2, mask).clamp_min(0.0))


def compute_rlsd_advantages(
    *,
    teacher_log_probs: torch.Tensor,
    student_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    config: RLSDConfig,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Reweight GRPO advantages using privileged teacher evidence.

    Formula:
        delta_t = stopgrad(teacher_pi_logp_t - student_logp_t)
        weight_t = exp(sign(A_t) * delta_t)
        weight_t = clamp(weight_t, 1 - eps, 1 + eps)
        A'_t = A_t * ((1 - lambda) + lambda * weight_t)

    The output preserves the sign of every non-zero valid advantage by
    construction because the multiplicative factor is strictly positive.
    """

    config.validate()
    if not (teacher_log_probs.shape == student_log_probs.shape == advantages.shape == response_mask.shape):
        raise ValueError(
            "teacher_log_probs, student_log_probs, advantages, and response_mask must share the same shape"
        )

    mask = _as_mask(response_mask, advantages)
    delta = teacher_log_probs.detach() - student_log_probs.detach()
    sign_adv = torch.sign(advantages.detach())
    signed_delta = (sign_adv * delta).detach()
    valid = mask.bool()
    if valid.any():
        for name, tensor in {
            "teacher_log_probs": teacher_log_probs.detach(),
            "student_log_probs": student_log_probs.detach(),
            "advantages": advantages.detach(),
            "signed_delta": signed_delta,
        }.items():
            if not torch.isfinite(tensor[valid]).all():
                bad = (~torch.isfinite(tensor[valid])).sum().item()
                raise ValueError(f"RLSD found non-finite {name} on {bad} valid tokens")

    log_low = math.log(1.0 - config.clip_range)
    log_high = math.log(1.0 + config.clip_range)
    clipped_log_weight = signed_delta.clamp(min=log_low, max=log_high)
    clipped_weights = torch.exp(clipped_log_weight) * mask
    reweight = ((1.0 - config.lam) + config.lam * clipped_weights) * mask

    if config.negative_only:
        seq_adv = (advantages.detach() * mask).sum(dim=-1, keepdim=True)
        negative_seq = (seq_adv < 0).to(dtype=advantages.dtype, device=advantages.device)
        reweight = negative_seq * reweight + (1.0 - negative_seq) * mask

    reweighted_advantages = advantages * reweight.detach() * mask
    valid_nonzero = (mask > 0) & (advantages != 0)
    sign_flips = ((torch.sign(reweighted_advantages) != torch.sign(advantages)) & valid_nonzero).sum()

    metrics = {
        "rlsd/delta_mean": _masked_mean(delta, mask).detach().item(),
        "rlsd/delta_std": _masked_std(delta, mask).detach().item(),
        "rlsd/log_weight_mean": _masked_mean(signed_delta, mask).detach().item(),
        "rlsd/log_weight_std": _masked_std(signed_delta, mask).detach().item(),
        "rlsd/weight_mean": _masked_mean(clipped_weights, mask).detach().item(),
        "rlsd/weight_std": _masked_std(clipped_weights, mask).detach().item(),
        "rlsd/clip_low_ratio": _masked_mean((signed_delta < log_low).to(dtype=advantages.dtype), mask).detach().item(),
        "rlsd/clip_high_ratio": _masked_mean(
            (signed_delta > log_high).to(dtype=advantages.dtype), mask
        ).detach().item(),
        "rlsd/effective_lambda": config.lam,
        "rlsd/adv_abs_mean_before": _masked_mean(advantages.detach().abs(), mask).detach().item(),
        "rlsd/adv_abs_mean_after": _masked_mean(reweighted_advantages.detach().abs(), mask).detach().item(),
        "rlsd/sign_flip_count": int(sign_flips.detach().item()),
    }

    pos_mask = mask * (advantages > 0).to(dtype=advantages.dtype)
    neg_mask = mask * (advantages < 0).to(dtype=advantages.dtype)
    metrics["rlsd/delta_pos_adv_mean"] = _masked_mean(delta, pos_mask).detach().item()
    metrics["rlsd/delta_neg_adv_mean"] = _masked_mean(delta, neg_mask).detach().item()
    return reweighted_advantages, metrics


def compute_opd_pg_advantages(
    *,
    teacher_log_probs: torch.Tensor,
    student_old_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    logratio_clip_abs: float = 5.0,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Build sampled-token OPD-PG/K1 advantages from teacher and old-policy logprobs."""

    if teacher_log_probs.shape != student_old_log_probs.shape or teacher_log_probs.shape != response_mask.shape:
        raise ValueError("teacher_log_probs, student_old_log_probs, and response_mask must share the same shape")
    if logratio_clip_abs <= 0:
        raise ValueError("logratio_clip_abs must be positive")

    mask = _as_mask(response_mask, teacher_log_probs)
    raw_logratio = (teacher_log_probs.detach() - student_old_log_probs.detach()) * mask
    token_advantages = raw_logratio.clamp(min=-logratio_clip_abs, max=logratio_clip_abs) * mask
    metrics = {
        "opd/logratio_mean": _masked_mean(raw_logratio, mask).detach().item(),
        "opd/logratio_abs_mean": _masked_mean(raw_logratio.abs(), mask).detach().item(),
        "opd/logratio_p99": torch.quantile(raw_logratio[mask.bool()].abs(), 0.99).detach().item()
        if mask.bool().any()
        else 0.0,
        "opd/logratio_clamp_ratio": _masked_mean(
            (raw_logratio.abs() > logratio_clip_abs).to(dtype=teacher_log_probs.dtype), mask
        ).detach().item(),
        "opd/teacher_logp_mean": _masked_mean(teacher_log_probs.detach(), mask).detach().item(),
        "opd/student_old_logp_mean": _masked_mean(student_old_log_probs.detach(), mask).detach().item(),
    }
    return token_advantages, metrics
