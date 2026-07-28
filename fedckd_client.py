"""FedCKD client built on the repository's existing model and dataset stack."""

import gc
from typing import Dict, Mapping, Optional

import torch
from transformers import DataCollatorForSeq2Seq, Trainer, TrainingArguments

from fedckd import FedCKDTrainerMixin
from util import get_lora_state_dict, set_lora_state_dict


class FedCKDTrainer(FedCKDTrainerMixin, Trainer):
    """Transformers Trainer extended with FedCKD's contribution-aware updates."""


class FedCKDClient:
    def __init__(self, client_id, client_dataset, tokenizer, cache_path, gradient_checkpointing=True):
        self.client_id = client_id
        self.client_dataset = client_dataset
        self.tokenizer = tokenizer
        self.cache_path = cache_path
        self.gradient_checkpointing = gradient_checkpointing

    @staticmethod
    def _enable_dual_lora(model) -> None:
        """Train LoRA-0 and LoRA-1 while keeping the backbone frozen."""
        for name, parameter in model.named_parameters():
            parameter.requires_grad = (
                "lora_A0" in name or "lora_B0" in name
                or "lora_A1" in name or "lora_B1" in name
            )

    def local_training(
        self,
        model,
        starting_state: Mapping[str, torch.Tensor],
        global_reference: Optional[torch.Tensor],
        lr: float,
        epochs: int,
        batch_size: int,
        gradient_accumulation_steps: int,
        config: Dict,
        routing_statistics: Optional[Mapping[str, float]] = None,
    ):
        if gradient_accumulation_steps != 1:
            raise ValueError(
                "FedCKD requires gradient_accumulation_steps=1 so each contribution "
                "score and branch mask applies to exactly one micro-batch."
            )
        set_lora_state_dict(model, starting_state)
        self._enable_dual_lora(model)
        model.train()

        training_args = TrainingArguments(
            output_dir=f"{self.cache_path}/fedckd/client_{self.client_id}_checkpoints",
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            num_train_epochs=epochs,
            learning_rate=lr,
            fp16=True,
            fp16_full_eval=True,
            half_precision_backend="auto",
            logging_steps=30,
            optim="adamw_torch",
            weight_decay=0.05,
            eval_strategy="no",
            save_strategy="no",
            remove_unused_columns=False,
            gradient_checkpointing=self.gradient_checkpointing,
        )
        trainer = FedCKDTrainer(
            model=model,
            args=training_args,
            train_dataset=self.client_dataset,
            tokenizer=self.tokenizer,
            data_collator=DataCollatorForSeq2Seq(
                self.tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
            ),
        )
        trainer.configure_fedckd(
            global_reference=global_reference,
            global_anchor=starting_state,
            sketch_dim=config["sketch_dim"],
            temperature=config["temperature"],
            ema=config["ema"],
            alpha=config["alpha"],
            beta=config["beta"],
            gamma=config["gamma"],
            orth_lambda=config["orth_lambda"],
            prox_lambda=config["prox_lambda"],
            phase_steps=config["phase_steps"],
            warmup_steps=config["warmup_steps"],
            routing_mode=config["routing_mode"],
            update_mode=config["update_mode"],
            routing_statistics=routing_statistics,
        )
        trainer.train()

        state = get_lora_state_dict(model)
        metrics = trainer.fedckd_metrics()
        global_mass = trainer.fedckd_global_mass()
        updated_routing_statistics = trainer.fedckd_routing_statistics()
        nonfinite_parameters = [
            name for name, value in state.items() if not torch.isfinite(value).all()
        ]
        if nonfinite_parameters:
            # The worker model is reused across clients. Restore the downloaded
            # state so this failed client cannot poison a later client or the
            # server aggregate; record the event for experiment diagnostics.
            print(
                f"[FedCKD client {self.client_id}] discarded non-finite update "
                f"({len(nonfinite_parameters)} LoRA tensors)",
                flush=True,
            )
            state = {
                name: value.detach().cpu().clone() for name, value in starting_state.items()
            }
            set_lora_state_dict(model, state)
            global_mass = 0.0
            metrics["nonfinite_parameter_tensors"] = float(len(nonfinite_parameters))
        else:
            metrics["nonfinite_parameter_tensors"] = 0.0
        del trainer, training_args
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return state, metrics, global_mass, updated_routing_statistics
