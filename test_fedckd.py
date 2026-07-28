import torch

from fedckd import (
    branch_gradient_scales,
    contribution_scores,
    initial_routing_statistics,
    normalize_routing_statistics,
)
from fedckd_server import FedCKDServer
from main_fedckd import build_parser, validate
from fedckd_worker import warmup_steps_for_round
from infer import load_client_states, method_uses_router
from peft.tuners.lora import Linear


def test_warmup_runs_only_in_first_communication_round():
    assert warmup_steps_for_round(20, 0) == 20
    assert warmup_steps_for_round(20, 1) == 0
    assert warmup_steps_for_round(20, 19) == 0


def test_routing_statistics_are_restored_and_probability_normalized():
    restored = normalize_routing_statistics({
        "gain_mean": 2.0,
        "gain_var": 4.0,
        "route_global": 2.0,
        "route_private": 1.0,
    })
    assert restored["gain_mean"] == 2.0
    assert restored["gain_var"] == 4.0
    assert restored["route_global"] == 2.0 / 3.0
    assert restored["route_private"] == 1.0 / 3.0


def test_invalid_routing_statistics_fall_back_to_safe_defaults():
    restored = normalize_routing_statistics({
        "gain_mean": float("nan"),
        "gain_var": float("nan"),
        "route_global": float("nan"),
        "route_private": float("nan"),
    })
    assert restored == initial_routing_statistics()


def test_dual_lora_without_router_sums_both_adapter_outputs():
    layer = Linear(3, 2, r=2, lora_nums=2, asymmetric=False, use_router=False)
    assert not hasattr(layer, "lora_route")
    output = layer(torch.randn(2, 1, 3))
    assert output.shape == (2, 1, 2)


def test_fedckd_inference_uses_router_free_model():
    assert not method_uses_router("fedckd")
    assert not method_uses_router("feddcr")
    assert method_uses_router("fedavg")


def test_fedckd_global_checkpoint_restores_complete_client_state(tmp_path):
    checkpoint_path = tmp_path / "global.pt"
    expected = {
        "layer.lora_A0.weight": torch.tensor([1.0]),
        "layer.lora_B0.weight": torch.tensor([2.0]),
        "layer.lora_A1.weight": torch.tensor([3.0]),
        "layer.lora_B1.weight": torch.tensor([4.0]),
    }
    torch.save([expected], checkpoint_path)

    restored = load_client_states(checkpoint_path, [0])
    assert restored[0].keys() == expected.keys()
    for name, tensor in expected.items():
        assert torch.equal(restored[0][name], tensor)


def test_static_joint_and_alternating_branch_scales():
    assert branch_gradient_scales("joint", 0.5, 0.5) == {"global": 0.5, "private": 0.5}
    assert branch_gradient_scales("global", 0.7, 0.3) == {"global": 0.7, "private": 0.0}
    assert branch_gradient_scales("private", 0.7, 0.3) == {"global": 0.0, "private": 0.3}


def test_uniform_aggregation_ignores_contribution_masses():
    states = {
        0: {"layer.lora_A0.weight": torch.tensor([1.0])},
        1: {"layer.lora_A0.weight": torch.tensor([3.0])},
    }
    shared, weights = FedCKDServer.aggregate_global(states, {0: 1.0, 1: 100.0}, "uniform")
    assert weights == {0: 0.5, 1: 0.5}
    assert torch.equal(shared["layer.lora_A0.weight"], torch.tensor([2.0]))


def test_global_mass_aggregation_ignores_nonfinite_mass():
    states = {
        0: {"layer.lora_A0.weight": torch.tensor([1.0])},
        1: {"layer.lora_A0.weight": torch.tensor([3.0])},
    }
    shared, weights = FedCKDServer.aggregate_global(states, {0: float("nan"), 1: 1.0})
    assert weights[0] < 1e-6
    assert weights[1] > 0.999
    assert torch.equal(shared["layer.lora_A0.weight"], torch.tensor([3.0]))


def test_nonfinite_routing_inputs_fall_back_to_balanced_weights():
    p_global, p_private, details = contribution_scores(
        float("nan"), 0.0, 1.0, 1.0, 0.0, 1.0, 1 / 3, 1 / 3, 1 / 3, 0.5
    )
    assert (p_global, p_private) == (0.5, 0.5)
    assert details["invalid_routing_inputs"] == 1.0


def test_ablation_switches_default_to_full_fedckd_and_can_disable_state_memory():
    args = build_parser().parse_args([])
    validate(args)
    assert args.fedckd_routing_mode == "adaptive"
    assert args.fedckd_update_mode == "alternating"
    assert args.fedckd_persist_routing_statistics
    assert args.fedckd_aggregation == "global_mass"

    ablation = build_parser().parse_args([
        "--fedckd_routing_mode", "static",
        "--fedckd_update_mode", "joint",
        "--no-fedckd_persist_routing_statistics",
        "--fedckd_aggregation", "uniform",
    ])
    validate(ablation)
    assert not ablation.fedckd_persist_routing_statistics
