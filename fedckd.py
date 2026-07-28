"""Contribution-aware knowledge decoupling for FedCKD.

FedCKD keeps the original training samples unchanged.  For every micro-batch
(the repository default is batch_size=1, hence this is sample-level scoring),
it estimates whether the supervision should update the globally aggregated
LoRA-0 or the client-private LoRA-1 from three complementary signals:

1. global gradient consistency with the previous server update;
2. personalization gain, measured by global-only loss minus joint loss;
3. relative gradient response of the global and private LoRA branches.

The trainer alternates global and private optimization phases and applies a
lightweight B-subspace orthogonality regularizer.  Only LoRA-0 is aggregated.
"""

from __future__ import annotations

from contextlib import contextmanager
import math
from typing import Dict, Iterable, Mapping, Optional, Tuple

import torch
import torch.nn.functional as F

from feddcr import branch_of, cosine, project_named_tensors, state_update_sketch


@contextmanager
def temporarily_disable_private_branch(model):
    """Temporarily zero LoRA-1 B matrices for a global-only scoring forward.

    LoRA B matrices are zero-initialized in this repository and are the final
    projection of each adapter.  Zeroing only B1 therefore removes the private
    residual without touching the frozen backbone or the global adapter.  The
    tensors are restored before the training forward, so optimizer state and
    gradients remain intact.
    """
    saved = []
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "lora_B1" in name:
                saved.append((parameter, parameter.detach().clone()))
                parameter.zero_()
    try:
        yield
    finally:
        with torch.no_grad():
            for parameter, value in saved:
                parameter.copy_(value)


def initial_routing_statistics() -> Dict[str, float]:
    """Return numerically safe per-client statistics for contribution scoring."""
    return {
        "gain_mean": 0.0,
        "gain_var": 1.0,
        "route_global": 0.5,
        "route_private": 0.5,
    }


def normalize_routing_statistics(
    routing_statistics: Optional[Mapping[str, float]],
) -> Dict[str, float]:
    """Validate checkpointed contribution statistics before restoring them."""
    normalized = initial_routing_statistics()
    if not routing_statistics:
        return normalized

    for key in ("gain_mean", "gain_var", "route_global", "route_private"):
        value = routing_statistics.get(key)
        if value is not None and math.isfinite(float(value)):
            normalized[key] = float(value)
    normalized["gain_var"] = max(normalized["gain_var"], 1e-6)
    route_sum = normalized["route_global"] + normalized["route_private"]
    if route_sum <= 1e-6:
        normalized["route_global"] = normalized["route_private"] = 0.5
    else:
        normalized["route_global"] /= route_sum
        normalized["route_private"] /= route_sum
    return normalized


def branch_parameters(model, branch: str):
    return [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if branch_of(name) == branch
    ]


def normalized_branch_response(named_gradients, branch: str, eps: float = 1e-12) -> float:
    """Return RMS gradient magnitude for one LoRA branch.

    RMS rather than a raw norm reduces bias from unequal parameter counts.
    """
    squared_sum = 0.0
    count = 0
    for name, gradient in named_gradients:
        if branch_of(name) != branch or gradient is None:
            continue
        # Scores only need relative response. Clamping avoids fp32 overflow in
        # squaring an otherwise finite AMP gradient before Trainer clips it.
        tensor = gradient.detach().float().clamp(min=-1e10, max=1e10)
        squared_sum += float(tensor.square().sum().cpu())
        count += tensor.numel()
    response = (squared_sum / max(count, 1) + eps) ** 0.5
    return response if math.isfinite(response) else 0.0


def orthogonality_loss(model) -> torch.Tensor:
    """Penalize overlap between normalized global/private LoRA-B subspaces."""
    named = dict(model.named_parameters())
    losses = []
    for global_name, global_b in named.items():
        if "lora_B0" not in global_name:
            continue
        private_name = global_name.replace("lora_B0", "lora_B1")
        private_b = named.get(private_name)
        if private_b is None:
            continue
        # Linear weights are [out_features, rank]; columns span the LoRA output
        # subspace. Skip zero/near-zero columns: at LoRA initialization B is
        # exactly zero, and differentiating normalization around zero in fp16
        # training can create an excessively large regularizer gradient.
        global_value = global_b.float()
        private_value = private_b.float()
        global_norm = global_value.norm(p=2, dim=0, keepdim=True)
        private_norm = private_value.norm(p=2, dim=0, keepdim=True)
        if not torch.isfinite(global_norm).all() or not torch.isfinite(private_norm).all():
            continue
        if (global_norm <= 1e-6).any() or (private_norm <= 1e-6).any():
            continue
        global_unit = global_value / global_norm
        private_unit = private_value / private_norm
        overlap = global_unit.transpose(0, 1) @ private_unit
        losses.append(overlap.square().mean())
    if not losses:
        reference = next(model.parameters())
        return reference.new_zeros(())
    return torch.stack(losses).mean()


def proximal_loss(model, global_anchor: Mapping[str, torch.Tensor]) -> torch.Tensor:
    """Keep the client's global LoRA close to the server-downloaded anchor."""
    terms = []
    for name, parameter in model.named_parameters():
        if branch_of(name) != "global" or name not in global_anchor:
            continue
        anchor = global_anchor[name].to(parameter.device, dtype=parameter.dtype)
        terms.append((parameter - anchor).float().square().mean())
    if not terms:
        reference = next(model.parameters())
        return reference.new_zeros(())
    return torch.stack(terms).mean()


def contribution_scores(
    gradient_consistency: float,
    personalization_gain: float,
    global_response: float,
    private_response: float,
    gain_center: float,
    gain_scale: float,
    alpha: float,
    beta: float,
    gamma: float,
    temperature: float,
) -> Tuple[float, float, Dict[str, float]]:
    """Fuse the three indicators into complementary soft routing weights."""
    inputs = (gradient_consistency, personalization_gain, global_response, private_response)
    if not all(math.isfinite(value) for value in inputs):
        return 0.5, 0.5, {
            "gradient_consistency": gradient_consistency if math.isfinite(gradient_consistency) else 0.0,
            "personalization_gain": personalization_gain if math.isfinite(personalization_gain) else 0.0,
            "global_response": global_response if math.isfinite(global_response) else 0.0,
            "private_response": private_response if math.isfinite(private_response) else 0.0,
            "score_global": 0.5,
            "score_private": 0.5,
            "invalid_routing_inputs": 1.0,
        }
    c_hat = min(max((gradient_consistency + 1.0) / 2.0, 0.0), 1.0)
    standardized_gain = (personalization_gain - gain_center) / max(gain_scale, 1e-6)
    gain_hat = float(torch.sigmoid(torch.tensor(standardized_gain)))
    response_sum = global_response + private_response + 1e-12
    q_global = global_response / response_sum

    score_global = alpha * c_hat + beta * (1.0 - gain_hat) + gamma * q_global
    score_private = 1.0 - score_global
    logits = torch.tensor([score_global, score_private]) / temperature
    probabilities = torch.softmax(logits, dim=0)
    details = {
        "gradient_consistency": gradient_consistency,
        "personalization_gain": personalization_gain,
        "global_response": global_response,
        "private_response": private_response,
        "score_global": score_global,
        "score_private": score_private,
    }
    return float(probabilities[0]), float(probabilities[1]), details


def branch_gradient_scales(phase: str, p_global: float, p_private: float) -> Dict[str, float]:
    """Return the two LoRA gradient multipliers for one FedCKD update phase."""
    if phase == "global":
        return {"global": p_global, "private": 0.0}
    if phase == "private":
        return {"global": 0.0, "private": p_private}
    if phase == "joint":
        return {"global": p_global, "private": p_private}
    if phase == "warmup":
        return {"global": 0.5, "private": 0.5}
    raise ValueError(f"Unsupported FedCKD update phase: {phase}")


class FedCKDTrainerMixin:
    """Trainer mixin implementing scoring, alternating updates and regularization."""

    def configure_fedckd(
        self,
        global_reference: Optional[torch.Tensor],
        global_anchor: Mapping[str, torch.Tensor],
        sketch_dim: int,
        temperature: float,
        ema: float,
        alpha: float,
        beta: float,
        gamma: float,
        orth_lambda: float,
        prox_lambda: float,
        phase_steps: int,
        warmup_steps: int,
        routing_mode: str = "adaptive",
        update_mode: str = "alternating",
        routing_statistics: Optional[Mapping[str, float]] = None,
    ) -> None:
        if routing_mode not in {"static", "adaptive"}:
            raise ValueError(f"Unsupported FedCKD routing mode: {routing_mode}")
        if update_mode not in {"joint", "alternating"}:
            raise ValueError(f"Unsupported FedCKD update mode: {update_mode}")
        self._ckd_global_reference = global_reference
        self._ckd_global_anchor = {
            name: value.detach().cpu().clone() for name, value in global_anchor.items()
        }
        self._ckd_sketch_dim = sketch_dim
        self._ckd_temperature = temperature
        self._ckd_ema = ema
        self._ckd_alpha = alpha
        self._ckd_beta = beta
        self._ckd_gamma = gamma
        self._ckd_orth_lambda = orth_lambda
        self._ckd_prox_lambda = prox_lambda
        self._ckd_phase_steps = max(int(phase_steps), 1)
        self._ckd_warmup_steps = max(int(warmup_steps), 0)
        self._ckd_routing_mode = routing_mode
        self._ckd_update_mode = update_mode
        self._ckd_micro_step = 0
        statistics = normalize_routing_statistics(routing_statistics)
        self._ckd_gain_mean = statistics["gain_mean"]
        self._ckd_gain_var = statistics["gain_var"]
        self._ckd_route_ema = torch.tensor([
            statistics["route_global"], statistics["route_private"],
        ])
        self._ckd_metric_sums: Dict[str, float] = {}
        self._ckd_metric_steps = 0
        self._ckd_nonfinite_gradient_elements = 0
        self._ckd_nonfinite_loss_steps = 0

    def _ckd_sanitize_lora_gradients(self, model) -> None:
        """Prevent one fp16 overflow from contaminating routing or aggregation."""
        for name, parameter in model.named_parameters():
            if branch_of(name) and parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                self._ckd_nonfinite_gradient_elements += int((~torch.isfinite(parameter.grad)).sum())
                parameter.grad.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)

    def _ckd_phase(self) -> str:
        if self._ckd_update_mode == "joint":
            return "joint"
        if self._ckd_micro_step < self._ckd_warmup_steps:
            return "warmup"
        block = (self._ckd_micro_step - self._ckd_warmup_steps) // self._ckd_phase_steps
        return "global" if block % 2 == 0 else "private"

    def _ckd_accumulate_metrics(self, details: Dict[str, float], p_global: float, p_private: float, phase: str):
        values = dict(details)
        values.update({
            "p_global": p_global,
            "p_private": p_private,
            "phase_global": float(phase == "global"),
            "phase_private": float(phase == "private"),
            "phase_warmup": float(phase == "warmup"),
            "phase_joint": float(phase == "joint"),
        })
        for key, value in values.items():
            self._ckd_metric_sums[key] = self._ckd_metric_sums.get(key, 0.0) + float(value)
        self._ckd_metric_steps += 1

    def training_step(self, model, inputs, *args, **kwargs):
        model.train()
        inputs = self._prepare_inputs(inputs)
        self._ckd_sanitize_lora_gradients(model)

        # Joint forward used for actual optimization.
        outputs = model(**inputs)
        data_loss = outputs.loss
        phase = self._ckd_phase() ####获取当前是全局lora更新还是本地lora更新阶段
        if not torch.isfinite(data_loss):
            self._ckd_nonfinite_loss_steps += 1
            self._ckd_sanitize_lora_gradients(model)
            self._ckd_accumulate_metrics(
                {"routing_adaptive": float(self._ckd_routing_mode == "adaptive"), "skipped_nonfinite_loss": 1.0},
                0.5,
                0.5,
                phase,
            )
            self._ckd_micro_step += 1
            return data_loss.detach().new_zeros(())
        ####双分支路由方式，根据数据贡献路由，或者静态
        if self._ckd_routing_mode == "adaptive":
            # Scoring forward: no gradients and private residual removed.
            with torch.no_grad(), temporarily_disable_private_branch(model):
                global_only_loss = model(**inputs).loss.detach().float()
            if torch.isfinite(global_only_loss):
                joint_loss_value = float(data_loss.detach().float().cpu())
                gain = float(global_only_loss.cpu()) - joint_loss_value
            else:
                gain = None

        orth = orthogonality_loss(model)  ###全局与局部正交约束
        prox = proximal_loss(model, self._ckd_global_anchor)  ###全局约束损失
        total_loss = data_loss + self._ckd_orth_lambda * orth
        if phase in ("global", "warmup", "joint"):
            total_loss = total_loss + self._ckd_prox_lambda * prox
        if not torch.isfinite(total_loss):
            self._ckd_nonfinite_loss_steps += 1
            self._ckd_sanitize_lora_gradients(model)
            self._ckd_accumulate_metrics(
                {"routing_adaptive": float(self._ckd_routing_mode == "adaptive"), "skipped_nonfinite_loss": 1.0},
                0.5,
                0.5,
                phase,
            )
            self._ckd_micro_step += 1
            return total_loss.detach().new_zeros(())

        # FedCKD requires an unaccumulated per-sample gradient: branch selection
        # and branch masking would otherwise alter gradients from prior samples.
        self.accelerator.backward(total_loss)
        self._ckd_sanitize_lora_gradients(model)

        if self._ckd_routing_mode == "adaptive" and gain is not None:
            named_gradients = [
                (name, parameter.grad)
                for name, parameter in model.named_parameters()
                if branch_of(name) and parameter.grad is not None
            ]
            global_sketch = project_named_tensors(
                ((name, grad) for name, grad in named_gradients if branch_of(name) == "global"),
                self._ckd_sketch_dim,
            )
            consistency = (
                cosine(global_sketch, self._ckd_global_reference)
                if self._ckd_global_reference is not None else 0.0
            )
            global_response = normalized_branch_response(named_gradients, "global")
            private_response = normalized_branch_response(named_gradients, "private")

            # Online EMA normalizes loss-gain across heterogeneous clients/tasks.
            delta = gain - self._ckd_gain_mean
            self._ckd_gain_mean = self._ckd_ema * self._ckd_gain_mean + (1.0 - self._ckd_ema) * gain
            self._ckd_gain_var = self._ckd_ema * self._ckd_gain_var + (1.0 - self._ckd_ema) * delta * delta
            p_global, p_private, details = contribution_scores(
                consistency, gain, global_response, private_response,
                self._ckd_gain_mean, self._ckd_gain_var ** 0.5,
                self._ckd_alpha, self._ckd_beta, self._ckd_gamma,
                self._ckd_temperature,
            )
            current = torch.tensor([p_global, p_private])
            self._ckd_route_ema = self._ckd_ema * self._ckd_route_ema + (1.0 - self._ckd_ema) * current
            p_global, p_private = map(float, self._ckd_route_ema)
            details["routing_adaptive"] = 1.0
        else:
            p_global = p_private = 0.5
            details = {
                "routing_adaptive": 0.0,
                "invalid_scoring_forward": float(self._ckd_routing_mode == "adaptive"),
            }
        
        if self._ckd_routing_mode == "static" and self._ckd_update_mode == "alternating":
            scales = (
                {"global": 1.0, "private": 0.0}
                if phase == "global"
                else {"global": 0.0, "private": 1.0}
            )
        else:
            scales = branch_gradient_scales(phase, p_global, p_private)
        for name, parameter in model.named_parameters():
            branch = branch_of(name)
            if branch and parameter.grad is not None:
                parameter.grad.mul_(scales[branch])

        details.update({
            "orth_loss": float(orth.detach().float().cpu()),
            "prox_loss": float(prox.detach().float().cpu()),
        })
        self._ckd_accumulate_metrics(details, p_global, p_private, phase)
        self._ckd_micro_step += 1
        return total_loss.detach() / self.args.gradient_accumulation_steps

    def fedckd_metrics(self) -> Dict[str, float]:
        denominator = max(self._ckd_metric_steps, 1)
        metrics = {key: value / denominator for key, value in self._ckd_metric_sums.items()}
        metrics["nonfinite_gradient_elements"] = float(self._ckd_nonfinite_gradient_elements)
        metrics["nonfinite_loss_steps"] = float(self._ckd_nonfinite_loss_steps)
        return metrics

    def fedckd_global_mass(self) -> float:
        """Effective amount of shareable supervision used for aggregation."""
        return self._ckd_metric_sums.get("p_global", 0.0)

    def fedckd_routing_statistics(self) -> Dict[str, float]:
        """Return state that must persist for this client across FedCKD rounds."""
        return {
            "gain_mean": float(self._ckd_gain_mean),
            "gain_var": float(max(self._ckd_gain_var, 1e-6)),
            "route_global": float(self._ckd_route_ema[0]),
            "route_private": float(self._ckd_route_ema[1]),
        }


__all__ = [
    "FedCKDTrainerMixin",
    "branch_gradient_scales",
    "initial_routing_statistics",
    "normalize_routing_statistics",
    "state_update_sketch",
]
