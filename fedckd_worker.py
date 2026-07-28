"""Persistent single-GPU worker for the FedCKD training entry point."""

import gc
import io
import math
import os
import traceback

import torch

from fedckd import state_update_sketch
from fedckd_client import FedCKDClient
from gpu_worker import _cuda_visible_device
from util import get_lora_state_dict, load_base_model, load_dataset, prepare_local_dataset
from utils.prompter import Prompter


def _data_file(args, client_id: int) -> str:
    return os.path.join(args.data_path, args.partition_dir, f"local_training_{client_id}.json")


def warmup_steps_for_round(configured_steps: int, round_idx: int) -> int:
    """Apply FedCKD's balanced warmup only during the first communication round."""
    return max(int(configured_steps), 0) if round_idx == 0 else 0


def fedckd_client_worker(
    gpu_id,
    args,
    client_states,
    client_routing_statistics,
    global_reference,
    round_idx,
    task_queue,
    result_queue,
):
    model = None
    try:
        physical_device = _cuda_visible_device(gpu_id)
        os.environ["CUDA_VISIBLE_DEVICES"] = physical_device
        torch.cuda.set_device(0)
        device = torch.device("cuda:0")
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        print(f"[FedCKD worker GPU {gpu_id}] ready on {physical_device}", flush=True)

        model, tokenizer = load_base_model(
            args.model_name, args.rank, args.lora_alpha, 2,
            device=device, gradient_checkpointing=args.gradient_checkpointing,
            use_router=False,
        )
        initial_state = get_lora_state_dict(model)
        prompter = Prompter("alpaca_short")
        config = {
            "sketch_dim": args.fedckd_sketch_dim,
            "temperature": args.fedckd_temperature,
            "ema": args.fedckd_ema,
            "alpha": args.fedckd_alpha,
            "beta": args.fedckd_beta,
            "gamma": 1.0 - args.fedckd_alpha - args.fedckd_beta,
            "orth_lambda": args.fedckd_orth_lambda,
            "prox_lambda": args.fedckd_prox_lambda,
            "phase_steps": args.fedckd_phase_steps,
            "warmup_steps": warmup_steps_for_round(args.fedckd_warmup_steps, round_idx),
            "routing_mode": args.fedckd_routing_mode,
            "update_mode": args.fedckd_update_mode,
        }

        while True:
            client_id = task_queue.get()
            if client_id is None:
                break
            try:
                client_data = load_dataset("json", data_files=_data_file(args, client_id))
                local_data = prepare_local_dataset(client_data, tokenizer, prompter)
                client = FedCKDClient(
                    client_id, local_data, tokenizer, args.result_dir,
                    gradient_checkpointing=args.gradient_checkpointing,
                )
                starting_state = client_states.get(client_id, initial_state)
                trained_state, metrics, global_mass, updated_routing_statistics = client.local_training(
                    model=model,
                    starting_state=starting_state,
                    global_reference=global_reference,
                    lr=args.lr,
                    epochs=args.local_epochs,
                    batch_size=args.batch_size,
                    gradient_accumulation_steps=args.gradient_accumulation_steps,
                    config=config,
                    routing_statistics=(
                        client_routing_statistics.get(client_id)
                        if args.fedckd_persist_routing_statistics else None
                    ),
                )
                safe_global_mass = float(global_mass)
                if not math.isfinite(safe_global_mass):
                    safe_global_mass = 0.0
                    metrics["nonfinite_global_mass"] = 1.0
                payload = {
                    "state": trained_state,
                    "global_update_sketch": state_update_sketch(
                        starting_state, trained_state, args.fedckd_sketch_dim, "global"
                    ),
                    "global_mass": max(safe_global_mass, 1e-8),
                    "metrics": metrics,
                    "routing_statistics": updated_routing_statistics,
                }
                buffer = io.BytesIO()
                torch.save(payload, buffer)
                result_queue.put(("client_result", gpu_id, client_id, buffer.getvalue(), None))
                del client, local_data, client_data
                gc.collect()
                torch.cuda.empty_cache()
            except Exception:
                error = traceback.format_exc()
                print(f"[FedCKD worker GPU {gpu_id}] client {client_id} failed:\n{error}", flush=True)
                result_queue.put(("client_result", gpu_id, client_id, None, error))
    except Exception:
        error = traceback.format_exc()
        result_queue.put(("worker_error", gpu_id, None, None, error))
    finally:
        if model is not None:
            del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        result_queue.put(("worker_done", gpu_id, None, None, None))
