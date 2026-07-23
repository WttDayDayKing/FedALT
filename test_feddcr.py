import torch

from feddcr import consensus_and_residuals, routing_probabilities, state_update_sketch


def test_state_sketch_is_linear_and_branch_specific():
    before = {"x.lora_A0.weight": torch.zeros(4), "x.lora_A1.weight": torch.zeros(4)}
    after = {"x.lora_A0.weight": torch.ones(4), "x.lora_A1.weight": torch.full((4,), 9.0)}
    sketch = state_update_sketch(before, after, 8, "global")
    assert torch.count_nonzero(sketch) == 4
    assert torch.isclose(sketch.abs().sum(), torch.tensor(4.0))


def test_consensus_is_converted_to_gradient_direction():
    global_proto, residuals = consensus_and_residuals({0: torch.tensor([1.0, 0.0]), 1: torch.tensor([1.0, 0.0])})
    assert torch.allclose(global_proto, torch.tensor([-1.0, 0.0]))
    assert set(residuals) == {0, 1}


def test_router_prefers_aligned_global_gradient():
    probs = routing_probabilities(
        torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0]),
        torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0]), 0.2, 0.5,
    )
    assert probs[0] > probs[2]
    assert probs[1] > probs[2]