import torch

from verl.trainer.extpi_rlsd.rlsd import RLSDConfig, compute_opd_pg_advantages, compute_rlsd_advantages


def _toy_pg_gradient(advantages):
    log_prob = torch.zeros_like(advantages, requires_grad=True)
    loss = -(log_prob * advantages).sum()
    loss.backward()
    return log_prob.grad


def test_rlsd_teacher_logprob_changes_gradient():
    advantages = torch.tensor([[1.0, -1.0]])
    old = torch.zeros_like(advantages)
    mask = torch.ones_like(advantages)

    adv_a, _ = compute_rlsd_advantages(
        teacher_log_probs=torch.zeros_like(advantages),
        student_old_log_probs=old,
        advantages=advantages,
        response_mask=mask,
        config=RLSDConfig(lam=0.5, clip_range=0.2),
    )
    adv_b, _ = compute_rlsd_advantages(
        teacher_log_probs=torch.tensor([[2.0, 2.0]]),
        student_old_log_probs=old,
        advantages=advantages,
        response_mask=mask,
        config=RLSDConfig(lam=0.5, clip_range=0.2),
    )

    assert not torch.allclose(_toy_pg_gradient(adv_a), _toy_pg_gradient(adv_b))


def test_opd_teacher_logprob_changes_gradient():
    old = torch.zeros(1, 2)
    mask = torch.ones_like(old)
    adv_a, _ = compute_opd_pg_advantages(
        teacher_log_probs=torch.zeros_like(old), student_old_log_probs=old, response_mask=mask
    )
    adv_b, _ = compute_opd_pg_advantages(
        teacher_log_probs=torch.tensor([[1.0, -1.0]]), student_old_log_probs=old, response_mask=mask
    )

    assert not torch.allclose(_toy_pg_gradient(adv_a), _toy_pg_gradient(adv_b))
