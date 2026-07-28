"""Data-function-aware routing utilities for FedDCR.

The projection is deliberately matrix-free: tensors are folded into a fixed
size CountSketch-like vector, so its memory and communication cost are O(d)
instead of O(number_of_LoRA_parameters).
"""

import hashlib
from collections import deque
from typing import Deque, Dict, Iterable, Mapping, Optional, Tuple

import torch


GLOBAL_MARKERS = ("lora_A0", "lora_B0")
PRIVATE_MARKERS = ("lora_A1", "lora_B1")


def branch_of(name: str) -> Optional[str]:
    if any(marker in name for marker in GLOBAL_MARKERS):
        return "global"
    if any(marker in name for marker in PRIVATE_MARKERS):
        return "private"
    return None


def canonical_lora_name(name: str) -> str:
    """Map equivalent shared/private LoRA tensors to one sketch coordinate."""
    return (
        name.replace("lora_A0", "lora_A")
        .replace("lora_A1", "lora_A")
        .replace("lora_B0", "lora_B")
        .replace("lora_B1", "lora_B")
    )


def _name_offset(name: str, dim: int) -> int:
    digest = hashlib.sha1(name.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little") % dim


def project_named_tensors(
    named_tensors: Iterable[Tuple[str, torch.Tensor]], dim: int
) -> torch.Tensor:
    """Return a deterministic, linear, CPU float32 sketch."""
    if dim < 1:
        raise ValueError("FedDCR sketch dimension must be positive")
    sketch = torch.zeros(dim, dtype=torch.float32)
    for name, tensor in named_tensors:
        if tensor is None:
            continue
        flat = tensor.detach().float().reshape(-1).cpu()
        if not flat.numel():
            continue
        # Chunking bounds temporary memory even for large LoRA tensors.
        # The global prototype is built from adapter 0 updates while private
        # gradients come from adapter 1. Their corresponding A/B matrices
        # must occupy identical sketch coordinates for cosine comparisons.
        offset = _name_offset(canonical_lora_name(name), dim)
        for start in range(0, flat.numel(), 1_000_000):
            values = flat[start:start + 1_000_000]
            positions = torch.arange(start, start + values.numel())
            indices = (positions + offset) % dim
            signs = torch.where((positions // dim) % 2 == 0, 1.0, -1.0)
            sketch.scatter_add_(0, indices, values * signs)
    return sketch


def state_update_sketch(
    before: Mapping[str, torch.Tensor],
    after: Mapping[str, torch.Tensor],
    dim: int,
    branch: str = "global",
    clip_norm: Optional[float] = None,
) -> torch.Tensor:
    """Sketch one adapter's clipped client update.

    ``clip_norm`` applies a single L2 clipping coefficient to the complete
    LoRA update before projection.  This keeps the client message bounded
    without materialising a flattened copy of every adapter tensor.
    """
    squared_norm = 0.0
    for name, value in after.items():
        if branch_of(name) == branch and name in before:
            delta = value.detach().float().cpu() - before[name].detach().float().cpu()
            squared_norm += float(torch.sum(delta * delta))

    scale = 1.0
    if clip_norm is not None:
        if clip_norm <= 0:
            raise ValueError("FedDCR clip norm must be positive")
        scale = min(1.0, clip_norm / max(squared_norm ** 0.5, 1e-12))

    updates = (
        (
            name,
            (value.detach().float().cpu() - before[name].detach().float().cpu()) * scale,
        )
        for name, value in after.items()
        if branch_of(name) == branch and name in before
    )
    return normalize(project_named_tensors(updates, dim))


def normalize(vector: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    vector = vector.detach().float().cpu()
    return vector / vector.norm().clamp_min(eps)

###计算全局共识和客户端自身残差方向
def consensus_and_residuals(
    client_updates: Mapping[int, torch.Tensor],
    client_weights: Optional[Mapping[int, float]] = None,
) -> Tuple[torch.Tensor, Dict[int, torch.Tensor]]:
    """Return a weighted consensus gradient and each client's residual.

    Clients upload normalized, clipped parameter-update sketches.  The sign
    is flipped here because local optimizers move parameters opposite to the
    loss gradient used by the router.
    """
    if not client_updates:
        raise ValueError("At least one client update is required")
    if client_weights is None:
        client_weights = {client_id: 1.0 for client_id in client_updates}
    if set(client_weights) != set(client_updates):
        raise ValueError("FedDCR client weights must match the uploaded updates")
    if any(weight <= 0 for weight in client_weights.values()):
        raise ValueError("FedDCR client weights must be positive")

    total_weight = float(sum(client_weights.values()))
    consensus = sum(
        normalize(update) * (float(client_weights[client_id]) / total_weight)
        for client_id, update in client_updates.items()
    )
    consensus = normalize(consensus)
    residuals = {}
    for client_id, update in client_updates.items():
        unit = normalize(update)
        residuals[client_id] = normalize(unit - torch.dot(unit, consensus) * consensus)
    # Prototypes are gradient directions; parameter updates point the other way.
    return -consensus, {client_id: -value for client_id, value in residuals.items()}


def cosine(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> float:
    a, b = a.float().cpu(), b.float().cpu()
    if a.norm() <= eps or b.norm() <= eps:
        return 0.0
    return float(torch.dot(a, b) / (a.norm() * b.norm()).clamp_min(eps))


def routing_probabilities(
    global_gradient: torch.Tensor,
    private_gradient: torch.Tensor,
    global_prototype: Optional[torch.Tensor],
    local_prototype: Optional[torch.Tensor],
    temperature: float,
    residual_penalty: float,
    learnability: float = 0.0,
    stability: float = 0.0,
    conflict_variance: float = 0.0,
    learnability_weight: float = 0.0,
    stability_weight: float = 0.0,
    conflict_variance_weight: float = 0.0,
) -> torch.Tensor:
    """Compute [global, private, defer] probabilities for one micro-batch."""
    if global_prototype is None or local_prototype is None:
        return torch.tensor([0.5, 0.5, 0.0])
    if not all(torch.isfinite(vector).all() for vector in (
        global_gradient, private_gradient, global_prototype, local_prototype
    )):
        return torch.tensor([0.5, 0.5, 0.0])
    sg = cosine(global_gradient, global_prototype)
    sl = cosine(private_gradient, local_prototype) - residual_penalty * cosine(
        private_gradient, global_prototype
    )
    global_value = sg + learnability_weight * learnability + stability_weight * stability
    private_value = sl + learnability_weight * learnability + stability_weight * stability
    conflict = -max(global_value, private_value) + conflict_variance_weight * conflict_variance
    return torch.softmax(torch.tensor([global_value, private_value, conflict]) / temperature, dim=0)


class FedDCRTrainerMixin:
    """Mixin applied before ``transformers.Trainer`` in the MRO.

    The base Trainer performs the normal backward pass. We then sketch that
    accumulation window and rescale gradients in place. This implements
    stop-gradient branch isolation without a second model forward.
    """

    def configure_feddcr(
        self,
        global_prototype: Optional[torch.Tensor],
        local_prototype: Optional[torch.Tensor],
        sketch_dim: int,
        temperature: float,
        residual_penalty: float,
        ema: float,
        learnability_weight: float = 0.1,
        stability_weight: float = 0.1,
        conflict_variance_weight: float = 0.5,
        score_history_size: int = 16,
    ) -> None:
        self._dcr_global_prototype = global_prototype
        self._dcr_local_prototype = local_prototype
        self._dcr_sketch_dim = sketch_dim
        self._dcr_temperature = temperature
        self._dcr_residual_penalty = residual_penalty
        self._dcr_ema = ema
        self._dcr_learnability_weight = learnability_weight
        self._dcr_stability_weight = stability_weight
        self._dcr_conflict_variance_weight = conflict_variance_weight
        self._dcr_probs = torch.tensor([0.5, 0.5, 0.0])
        self._dcr_prob_sum = torch.zeros(3)
        self._dcr_value_sum = torch.zeros(3)
        self._dcr_steps = 0
        self._dcr_previous_sketches = {"global": None, "private": None}
        self._dcr_stability = 0.0
        self._dcr_score_history: Deque[float] = deque(maxlen=score_history_size)
        self._dcr_nonfinite_gradient_elements = 0

    def training_step(self, model, inputs, *args, **kwargs):
        # A non-finite accumulated gradient must not be carried into the next
        # micro-batch or used as the baseline for its delta.
        for name, parameter in model.named_parameters():
            if branch_of(name) and parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                self._dcr_nonfinite_gradient_elements += int((~torch.isfinite(parameter.grad)).sum())
                parameter.grad.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
        previous = {
            name: parameter.grad.detach().clone()
            for name, parameter in model.named_parameters()
            if branch_of(name) and parameter.grad is not None
        }
        loss = super().training_step(model, inputs, *args, **kwargs)
        current_grads = []
        squared_norms = {"global": 0.0, "private": 0.0}
        for name, parameter in model.named_parameters():
            if not branch_of(name) or parameter.grad is None:
                continue
            prior = previous.get(name)
            gradient = parameter.grad if prior is None else parameter.grad - prior
            if not torch.isfinite(gradient).all():
                self._dcr_nonfinite_gradient_elements += int((~torch.isfinite(gradient)).sum())
                gradient = torch.nan_to_num(gradient, nan=0.0, posinf=0.0, neginf=0.0)
                if prior is None:
                    parameter.grad.copy_(gradient)
                else:
                    parameter.grad.copy_(prior + gradient)
            current_grads.append((name, gradient))
            squared_norms[branch_of(name)] += float(torch.sum(gradient.detach().float() ** 2))
        global_sketch = project_named_tensors(
            ((n, g) for n, g in current_grads if branch_of(n) == "global"), self._dcr_sketch_dim
        )
        private_sketch = project_named_tensors(
            ((n, g) for n, g in current_grads if branch_of(n) == "private"), self._dcr_sketch_dim
        )
        # A first-order proxy for the one-step loss reduction.  It avoids a
        # second optimizer step per micro-batch while retaining the intended
        # preference for learnable, non-stagnant data.
        learning_rate = float(getattr(self.args, "learning_rate", 0.0))
        learnability = 1.0 - torch.exp(torch.tensor(
            -0.5 * learning_rate * (squared_norms["global"] + squared_norms["private"])
        )).item()
        similarities = []
        for branch, sketch in (("global", global_sketch), ("private", private_sketch)):
            previous_sketch = self._dcr_previous_sketches[branch]
            if previous_sketch is not None:
                similarities.append(cosine(sketch, previous_sketch))
            self._dcr_previous_sketches[branch] = normalize(sketch)
        instantaneous_stability = sum(similarities) / len(similarities) if similarities else 0.0
        self._dcr_stability = (
            self._dcr_ema * self._dcr_stability
            + (1.0 - self._dcr_ema) * instantaneous_stability
        )
        conflict_variance = (
            float(torch.tensor(list(self._dcr_score_history)).var(unbiased=False))
            if len(self._dcr_score_history) > 1 else 0.0
        )
        current = routing_probabilities(
            global_sketch, private_sketch,
            self._dcr_global_prototype, self._dcr_local_prototype,
            self._dcr_temperature, self._dcr_residual_penalty,
            learnability=learnability,
            stability=self._dcr_stability,
            conflict_variance=conflict_variance,
            learnability_weight=self._dcr_learnability_weight,
            stability_weight=self._dcr_stability_weight,
            conflict_variance_weight=self._dcr_conflict_variance_weight,
        )
        if self._dcr_global_prototype is not None and self._dcr_local_prototype is not None:
            global_value = cosine(global_sketch, self._dcr_global_prototype)
            local_value = cosine(private_sketch, self._dcr_local_prototype) - self._dcr_residual_penalty * cosine(
                private_sketch, self._dcr_global_prototype
            )
            self._dcr_score_history.append(max(global_value, local_value))
        self._dcr_probs = self._dcr_ema * self._dcr_probs + (1.0 - self._dcr_ema) * current
        # ``p_defer`` already removes probability mass from the two update
        # branches. Multiplying by ``1 - p_defer`` again would suppress the
        # total update quadratically and stall learning when defer is common.
        scales = {"global": float(self._dcr_probs[0]), "private": float(self._dcr_probs[1])}
        for name, parameter in model.named_parameters():
            branch = branch_of(name)
            if branch and parameter.grad is not None:
                prior = previous.get(name)
                if prior is None:
                    parameter.grad.mul_(scales[branch])
                else:
                    parameter.grad.copy_(prior + (parameter.grad - prior) * scales[branch])
        self._dcr_prob_sum += self._dcr_probs
        self._dcr_value_sum += torch.tensor([learnability, self._dcr_stability, conflict_variance])
        self._dcr_steps += 1
        return loss

    def feddcr_metrics(self) -> Dict[str, float]:
        mean = self._dcr_prob_sum / max(self._dcr_steps, 1)
        values = self._dcr_value_sum / max(self._dcr_steps, 1)
        return {
            "p_global": float(mean[0]),
            "p_private": float(mean[1]),
            "p_defer": float(mean[2]),
            "learnability": float(values[0]),
            "stability": float(values[1]),
            "conflict_variance": float(values[2]),
            "nonfinite_gradient_elements": float(self._dcr_nonfinite_gradient_elements),
        }
