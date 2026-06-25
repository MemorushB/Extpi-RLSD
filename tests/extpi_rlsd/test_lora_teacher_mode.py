from verl.trainer.extpi_rlsd.scorer import peft_adapter_disabled


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
