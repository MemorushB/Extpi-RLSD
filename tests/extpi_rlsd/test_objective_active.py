from types import SimpleNamespace

import torch
from tensordict import TensorDict

from verl.trainer.extpi_rlsd.rlsd import RLSDConfig, compute_opd_pg_advantages, compute_rlsd_advantages
from verl.utils import tensordict_utils as tu
from verl.workers.utils import losses as worker_losses


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
        student_log_probs=old,
        advantages=advantages,
        response_mask=mask,
        config=RLSDConfig(lam=0.5, clip_range=0.2),
    )
    adv_b, _ = compute_rlsd_advantages(
        teacher_log_probs=torch.tensor([[2.0, 2.0]]),
        student_log_probs=old,
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


def _ppo_config(delta_source="current"):
    return SimpleNamespace(
        global_batch_info={},
        loss_scale_factor=None,
        policy_loss={
            "loss_mode": "vanilla",
            "rlsd_enabled": True,
            "rlsd_delta_student_logp_source": delta_source,
        },
        loss_agg_mode="token-mean",
        use_kl_loss=False,
        entropy_coeff=0,
    )


def _ppo_data():
    return TensorDict(
        {
            "response_mask": torch.ones(1, 2, dtype=torch.bool),
            "old_log_probs": torch.tensor([[0.1, 0.2]]),
            "advantages": torch.ones(1, 2),
            "teacher_pi_log_probs": torch.zeros(1, 2),
            "dp_size": torch.tensor(1),
            "batch_num_tokens": torch.tensor(2),
            "global_batch_size": torch.tensor(1),
        },
        batch_size=[],
    )


def test_rlsd_actor_loss_uses_current_logprob_by_default(monkeypatch):
    captured = {}

    def fake_compute_rlsd_advantages(*, teacher_log_probs, student_log_probs, advantages, response_mask, config):
        del teacher_log_probs, response_mask, config
        captured["student_log_probs"] = student_log_probs.clone()
        return advantages, {}

    monkeypatch.setattr(worker_losses, "no_padding_2_padding", lambda tensor, data: tensor)
    monkeypatch.setattr(worker_losses, "compute_rlsd_advantages", fake_compute_rlsd_advantages)
    monkeypatch.setattr(worker_losses, "get_policy_loss_fn", lambda loss_mode: lambda **kwargs: (torch.tensor(0.0), {}))

    current_log_probs = torch.tensor([[3.0, 4.0]], requires_grad=True)
    worker_losses.ppo_loss(_ppo_config(), {"log_probs": current_log_probs}, _ppo_data())

    assert torch.allclose(captured["student_log_probs"], current_log_probs.detach())


def test_rlsd_actor_loss_can_use_old_logprob_when_configured(monkeypatch):
    captured = {}

    def fake_compute_rlsd_advantages(*, teacher_log_probs, student_log_probs, advantages, response_mask, config):
        del teacher_log_probs, response_mask, config
        captured["student_log_probs"] = student_log_probs.clone()
        return advantages, {}

    monkeypatch.setattr(worker_losses, "no_padding_2_padding", lambda tensor, data: tensor)
    monkeypatch.setattr(worker_losses, "compute_rlsd_advantages", fake_compute_rlsd_advantages)
    monkeypatch.setattr(worker_losses, "get_policy_loss_fn", lambda loss_mode: lambda **kwargs: (torch.tensor(0.0), {}))

    worker_losses.ppo_loss(_ppo_config(delta_source="old"), {"log_probs": torch.tensor([[3.0, 4.0]])}, _ppo_data())

    assert torch.allclose(captured["student_log_probs"], torch.tensor([[0.1, 0.2]]))


def test_rlsd_actor_loss_uses_effective_lambda_from_batch_meta(monkeypatch):
    captured = {}

    def fake_compute_rlsd_advantages(*, teacher_log_probs, student_log_probs, advantages, response_mask, config):
        del teacher_log_probs, student_log_probs, response_mask
        captured["lambda"] = config.lam
        return advantages, {"rlsd/effective_lambda": config.lam}

    monkeypatch.setattr(worker_losses, "no_padding_2_padding", lambda tensor, data: tensor)
    monkeypatch.setattr(worker_losses, "compute_rlsd_advantages", fake_compute_rlsd_advantages)
    monkeypatch.setattr(worker_losses, "get_policy_loss_fn", lambda loss_mode: lambda **kwargs: (torch.tensor(0.0), {}))

    data = _ppo_data()
    tu.assign_non_tensor(data, rlsd_effective_lambda=0.125)
    _, metrics = worker_losses.ppo_loss(_ppo_config(), {"log_probs": torch.tensor([[3.0, 4.0]])}, data)

    assert captured["lambda"] == 0.125
    assert metrics["rlsd/effective_lambda"].aggregate() == 0.125
