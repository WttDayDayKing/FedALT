import torch

from feddcr import consensus_and_residuals, project_named_tensors, routing_probabilities, state_update_sketch
from infer import load_client_checkpoints
from peft.tuners.lora import Linear


def test_state_sketch_is_branch_specific_and_normalized():
    before = {"x.lora_A0.weight": torch.zeros(4), "x.lora_A1.weight": torch.zeros(4)}
    after = {"x.lora_A0.weight": torch.ones(4), "x.lora_A1.weight": torch.full((4,), 9.0)}
    sketch = state_update_sketch(before, after, 8, "global")
    assert torch.count_nonzero(sketch) == 4
    assert torch.isclose(sketch.norm(), torch.tensor(1.0))


def test_consensus_is_converted_to_gradient_direction():
    global_proto, residuals = consensus_and_residuals({0: torch.tensor([1.0, 0.0]), 1: torch.tensor([1.0, 0.0])})
    assert torch.allclose(global_proto, torch.tensor([-1.0, 0.0]))
    assert set(residuals) == {0, 1}


def test_weighted_consensus_prefers_larger_client():
    global_proto, _ = consensus_and_residuals(
        {0: torch.tensor([1.0, 0.0]), 1: torch.tensor([0.0, 1.0])},
        {0: 9, 1: 1},
    )
    assert global_proto[0] < global_proto[1] < 0


def test_clipped_sketch_is_normalized():
    before = {"x.lora_A0.weight": torch.zeros(4)}
    after = {"x.lora_A0.weight": torch.full((4,), 100.0)}
    sketch = state_update_sketch(before, after, 8, "global", clip_norm=1.0)
    assert torch.isclose(sketch.norm(), torch.tensor(1.0))


def test_equivalent_global_and_private_tensors_share_sketch_coordinates():
    value = torch.tensor([1.0, -2.0, 3.0, -4.0])
    global_sketch = project_named_tensors([("x.lora_A0.weight", value)], 17)
    private_sketch = project_named_tensors([("x.lora_A1.weight", value)], 17)
    assert torch.equal(global_sketch, private_sketch)


def test_router_prefers_aligned_global_gradient():
    probs = routing_probabilities(
        torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0]),
        torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0]), 0.2, 0.5,
    )
    assert probs[0] > probs[2]
    assert probs[1] > probs[2]


def test_router_falls_back_when_a_gradient_is_nonfinite():
    probs = routing_probabilities(
        torch.tensor([float("nan"), 0.0]), torch.tensor([0.0, 1.0]),
        torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0]), 0.2, 0.5,
    )
    assert torch.equal(probs, torch.tensor([0.5, 0.5, 0.0]))


def test_feddcr_dual_lora_can_disable_the_fedalt_router():
    layer = Linear(3, 4, r=2, lora_nums=2, asymmetric=False, use_router=False)
    assert not hasattr(layer, "lora_route")
    assert layer(torch.randn(2, 5, 3)).shape == (2, 5, 4)


def test_feddcr_inference_overlays_shared_and_private_states(tmp_path):
    checkpoint_dir = tmp_path / "round_0"
    checkpoint_dir.mkdir()
    torch.save([{"x.lora_A0.weight": torch.tensor([1.0])}], checkpoint_dir / "global.pt")
    torch.save({"x.lora_A1.weight": torch.tensor([2.0])}, checkpoint_dir / "client_0.pt")
    states, _ = load_client_checkpoints(checkpoint_dir, [0])
    assert set(states[0]) == {"x.lora_A0.weight", "x.lora_A1.weight"}


def test_latest_historical_round_uses_root_shared_state(tmp_path):
    checkpoint_dir = tmp_path / "round_1"
    checkpoint_dir.mkdir()
    torch.save([{"x.lora_A0.weight": torch.tensor([1.0])}], tmp_path / "global.pt")
    torch.save({"x.lora_A1.weight": torch.tensor([2.0])}, checkpoint_dir / "client_0.pt")
    states, _ = load_client_checkpoints(checkpoint_dir, [0])
    assert set(states[0]) == {"x.lora_A0.weight", "x.lora_A1.weight"}
