import torch

from verl.trainer.extpi_rlsd.scorer import InlineHFTeacherScorer, peft_adapter_disabled


class FakePeftModel:
    def __init__(self):
        self.enabled = True

    def disable_adapters(self):
        self.enabled = False

    def enable_adapters(self):
        self.enabled = True


def test_adapter_state_restored_after_context():
    model = FakePeftModel()
    assert model.enabled is True
    with peft_adapter_disabled(model):
        assert model.enabled is False
    assert model.enabled is True


def test_inline_hf_teacher_scorer_does_not_freeze_shared_model_parameters():
    model = torch.nn.Linear(2, 2)
    assert all(parameter.requires_grad for parameter in model.parameters())

    InlineHFTeacherScorer(model=model, tokenizer=object(), device="cpu")

    assert all(parameter.requires_grad for parameter in model.parameters())
