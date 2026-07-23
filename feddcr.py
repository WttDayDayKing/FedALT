"""Data-function-aware routing utilities for FedDCR.

The projection is deliberately matrix-free: tensors are folded into a fixed
size CountSketch-like vector, so its memory and communication cost are O(d)
instead of O(number_of_LoRA_parameters).
"""

import hashlib
from typing import Dict, Iterable, Mapping, Optional, Tuple

import torch


GLOBAL_MARKERS = ("lora_A0", "lora_B0")
PRIVATE_MARKERS = ("lora_A1", "lora_B1")


def branch_of(name: str) -> Optional[str]:
    if any(marker in name for marker in GLOBAL_MARKERS):
        return "global"
    if any(marker in name for marker in PRIVATE_MARKERS):
        return "private"
    return None


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
        offset = _name_offset(name, dim)
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
) -> torch.Tensor:
    updates = []
    for name, value in after.items():
        if branch_of(name) == branch and name in before:
            updates.append((name, value.detach().cpu() - before[name].detach().cpu()))
    return project_named_tensors(updates, dim)


def normalize(vector: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    vector = vector.detach().float().cpu()
    return vector / vector.norm().clamp_min(eps)


def consensus_and_residuals(
    client_updates: Mapping[int, torch.Tensor],
) -> Tuple[torch.Tensor, Dict[int, torch.Tensor]]:
    if not client_updates:
        raise ValueError("At least one client update is required")
    consensus = normalize(torch.stack(list(client_updates.values())).mean(0))
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
) -> torch.Tensor:
    """Compute [global, private, defer] probabilities for one micro-batch."""
    if global_prototype is None or local_prototype is None:
        return torch.tensor([0.5, 0.5, 0.0])
    sg = cosine(global_gradient, global_prototype)
    sl = cosine(private_gradient, local_prototype) - residual_penalty * cosine(
        private_gradient, global_prototype
    )
    conflict = -max(sg, sl)
    return torch.softmax(torch.tensor([sg, sl, conflict]) / temperature, dim=0)


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
    ) -> None:
        self._dcr_global_prototype = global_prototype
        self._dcr_local_prototype = local_prototype
        self._dcr_sketch_dim = sketch_dim
        self._dcr_temperature = temperature
        self._dcr_residual_penalty = residual_penalty
        self._dcr_ema = ema
        self._dcr_probs = torch.tensor([0.5, 0.5, 0.0])
        self._dcr_prob_sum = torch.zeros(3)
        self._dcr_steps = 0

    def training_step(self, model, inputs, *args, **kwargs):
        previous = {
            name: parameter.grad.detach().clone()
            for name, parameter in model.named_parameters()
            if branch_of(name) and parameter.grad is not None
        }
        loss = super().training_step(model, inputs, *args, **kwargs)
        current_grads = []
        for name, parameter in model.named_parameters():
            if not branch_of(name) or parameter.grad is None:
                continue
            prior = previous.get(name)
            current_grads.append((name, parameter.grad if prior is None else parameter.grad - prior))
        global_sketch = project_named_tensors(
            ((n, g) for n, g in current_grads if branch_of(n) == "global"), self._dcr_sketch_dim
        )
        private_sketch = project_named_tensors(
            ((n, g) for n, g in current_grads if branch_of(n) == "private"), self._dcr_sketch_dim
        )
        current = routing_probabilities(
            global_sketch, private_sketch,
            self._dcr_global_prototype, self._dcr_local_prototype,
            self._dcr_temperature, self._dcr_residual_penalty,
        )
        self._dcr_probs = self._dcr_ema * self._dcr_probs + (1.0 - self._dcr_ema) * current
        keep = 1.0 - float(self._dcr_probs[2])
        scales = {"global": float(self._dcr_probs[0]) * keep, "private": float(self._dcr_probs[1]) * keep}
        for name, parameter in model.named_parameters():
            branch = branch_of(name)
            if branch and parameter.grad is not None:
                prior = previous.get(name)
                if prior is None:
                    parameter.grad.mul_(scales[branch])
                else:
                    parameter.grad.copy_(prior + (parameter.grad - prior) * scales[branch])
        self._dcr_prob_sum += self._dcr_probs
        self._dcr_steps += 1
        return loss

    def feddcr_metrics(self) -> Dict[str, float]:
        mean = self._dcr_prob_sum / max(self._dcr_steps, 1)
        return {"p_global": float(mean[0]), "p_private": float(mean[1]), "p_defer": float(mean[2])}