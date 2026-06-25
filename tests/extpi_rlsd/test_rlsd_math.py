import torch

from verl.trainer.extpi_rlsd.rlsd import RLSDConfig, compute_opd_pg_advantages, compute_rlsd_advantages


def test_lambda_zero_reproduces_grpo_advantages():
    advantages = torch.tensor([[1.0, -2.0, 0.5]])
    teacher = torch.tensor([[0.4, -0.1, 0.2]])
    old = torch.tensor([[0.1, -0.2, 0.0]])
    mask = torch.tensor([[1, 1, 1]])

    out, metrics = compute_rlsd_advantages(
        teacher_log_probs=teacher,
        student_old_log_probs=old,
        advantages=advantages,
        response_mask=mask,
        config=RLSDConfig(lam=0.0, clip_range=0.2),
    )

    assert torch.allclose(out, advantages)
    assert metrics["rlsd/sign_flip_count"] == 0


def test_equal_teacher_student_logprobs_keep_weights_at_one():
    advantages = torch.tensor([[1.0, -2.0]])
    old = torch.tensor([[0.1, -0.2]])
    mask = torch.tensor([[1, 1]])

    out, metrics = compute_rlsd_advantages(
        teacher_log_probs=old,
        student_old_log_probs=old,
        advantages=advantages,
        response_mask=mask,
        config=RLSDConfig(lam=0.5, clip_range=0.2),
    )

    assert torch.allclose(out, advantages)
    assert metrics["rlsd/weight_mean"] == 1.0


def test_rlsd_preserves_sign_and_masks_padding():
    advantages = torch.tensor([[1.0, -2.0, 3.0]])
    teacher = torch.tensor([[10.0, -10.0, 3.0]])
    old = torch.zeros_like(teacher)
    mask = torch.tensor([[1, 1, 0]])

    out, metrics = compute_rlsd_advantages(
        teacher_log_probs=teacher,
        student_old_log_probs=old,
        advantages=advantages,
        response_mask=mask,
        config=RLSDConfig(lam=0.5, clip_range=0.2),
    )

    assert out[0, 2].item() == 0.0
    assert torch.sign(out[0, :2]).tolist() == torch.sign(advantages[0, :2]).tolist()
    assert metrics["rlsd/sign_flip_count"] == 0
    assert out[0, 0] <= advantages[0, 0] * 1.1 + 1e-6


def test_negative_only_changes_only_negative_sequences():
    advantages = torch.tensor([[1.0, 1.0], [-1.0, -1.0]])
    teacher = torch.tensor([[2.0, 2.0], [2.0, 2.0]])
    old = torch.zeros_like(teacher)
    mask = torch.ones_like(teacher)

    out, _ = compute_rlsd_advantages(
        teacher_log_probs=teacher,
        student_old_log_probs=old,
        advantages=advantages,
        response_mask=mask,
        config=RLSDConfig(lam=0.5, clip_range=0.2, negative_only=True),
    )

    assert torch.allclose(out[0], advantages[0])
    assert not torch.allclose(out[1], advantages[1])


def test_opd_pg_advantages_are_clipped_and_detached():
    teacher = torch.tensor([[10.0, -10.0]], requires_grad=True)
    old = torch.zeros_like(teacher, requires_grad=True)
    mask = torch.tensor([[1, 1]])

    out, metrics = compute_opd_pg_advantages(
        teacher_log_probs=teacher,
        student_old_log_probs=old,
        response_mask=mask,
        logratio_clip_abs=5.0,
    )

    assert torch.allclose(out, torch.tensor([[5.0, -5.0]]))
    assert out.requires_grad is False
    assert metrics["opd/logratio_clamp_ratio"] == 1.0
