import torch

from verl.trainer.extpi_rlsd.scorer import build_causal_score_inputs, gather_response_log_probs


def test_response_tokens_are_appended_without_retokenization_and_offsets_are_correct():
    prefix = torch.tensor([[0, 11, 12], [21, 22, 23]])
    prefix_mask = torch.tensor([[0, 1, 1], [1, 1, 1]])
    responses = torch.tensor([[5, 6], [7, 8]])
    response_mask = torch.tensor([[1, 1], [1, 0]])

    built = build_causal_score_inputs(
        prefix_input_ids=prefix,
        prefix_attention_mask=prefix_mask,
        response_ids=responses,
        response_mask=response_mask,
        pad_token_id=0,
    )

    assert built["input_ids"].tolist() == [[0, 11, 12, 5, 6], [21, 22, 23, 7, 0]]
    assert built["response_ids"].tolist() == responses.tolist()
    assert built["response_logit_positions"].tolist() == [[2, 3], [2, 3]]

    logits = torch.zeros(2, 5, 10)
    logits[0, 2, 5] = 10.0
    logits[0, 3, 6] = 10.0
    logits[1, 2, 7] = 10.0
    logits[1, 3, 8] = 10.0

    log_probs = gather_response_log_probs(
        logits=logits,
        response_ids=responses,
        response_logit_positions=built["response_logit_positions"],
        temperature=1.0,
    )

    assert log_probs.shape == responses.shape
    assert torch.all(log_probs > -0.01)


def test_temperature_changes_gathered_logprob_distribution():
    logits = torch.tensor([[[0.0, 2.0]]])
    responses = torch.tensor([[1]])
    positions = torch.tensor([[0]])

    cold = gather_response_log_probs(
        logits=logits,
        response_ids=responses,
        response_logit_positions=positions,
        temperature=0.5,
    )
    hot = gather_response_log_probs(
        logits=logits,
        response_ids=responses,
        response_logit_positions=positions,
        temperature=2.0,
    )

    assert cold.item() > hot.item()
