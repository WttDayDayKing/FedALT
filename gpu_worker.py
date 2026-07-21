"""GPU worker entry point used by :mod:`server`.

Each process owns exactly one GPU and loads one model.  The process can train
multiple clients serially, which keeps the model resident on that GPU while
preventing two quantized model copies from competing for the same memory.
"""

import gc
import os
import traceback

import torch

from client import Client
from util import get_lora_state_dict, load_base_model, load_dataset, prepare_local_dataset
from utils.prompter import Prompter


def _data_file(args, client_id: int) -> str:
    return os.path.join(
        args.data_path, args.partition_dir, f"local_training_{client_id}.json"
    )


def _cuda_visible_device(gpu_id: int) -> str:
    """Translate a parent-visible GPU index to a CUDA_VISIBLE_DEVICES token."""
    inherited = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not inherited:
        return str(gpu_id)
    visible_devices = [value.strip() for value in inherited.split(",") if value.strip()]
    if len(visible_devices) == 1:
        return visible_devices[0]
    if gpu_id >= len(visible_devices):
        raise ValueError(
            f"GPU index {gpu_id} is not present in inherited CUDA_VISIBLE_DEVICES={inherited!r}."
        )
    return visible_devices[gpu_id]


def client_worker(gpu_id, client_ids, args, client_states, result_queue):
    """Train ``client_ids`` on one physical GPU and return their LoRA states.

    ``spawn`` starts a fresh Python interpreter. Restricting the worker to one
    card before its first CUDA call makes that card process-local ``cuda:0``,
    which is also the device Accelerate expects for an 8-bit model.
    """
    model = None
    results = {}
    errors = {}

    try:
        physical_device = _cuda_visible_device(gpu_id)
        os.environ["CUDA_VISIBLE_DEVICES"] = physical_device
        torch.cuda.set_device(0)
        device = torch.device("cuda:0")
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

        print(
            f"[Worker GPU {gpu_id} (CUDA_VISIBLE_DEVICES={physical_device})] "
            f"clients: {list(client_ids)}",
            flush=True,
        )
        model, tokenizer = load_base_model(
            args.model_name,
            args.rank,
            args.lora_alpha,
            args.lora_n,
            device=device,
            gradient_checkpointing=args.gradient_checkpointing,
        )
        initial_state = get_lora_state_dict(model)
        prompter = Prompter("alpaca_short")

        for client_id in client_ids:
            try:
                client_data = load_dataset("json", data_files=_data_file(args, client_id))
                local_data = prepare_local_dataset(client_data, tokenizer, prompter)
                client = Client(
                    client_id=client_id,
                    client_dataset=local_data,
                    tokenizer=tokenizer,
                    prompter=prompter,
                    model_name=args.model_name,
                    device=device,
                    rank=args.rank,
                    lora_n=args.lora_n,
                    asymmetric=False,
                    cache_path=args.result_dir,
                    gradient_checkpointing=args.gradient_checkpointing,
                )
                results[client_id] = client.local_training(
                    model=model,
                    # First-round clients must all start from the same seeded
                    # LoRA state, rather than inheriting a prior client that
                    # happened to share this worker's GPU.
                    global_state=client_states.get(client_id, initial_state),
                    lr=args.lr,
                    epochs=args.local_epochs,
                    batch_size=args.batch_size,
                    gradient_accumulation_steps=args.gradient_accumulation_steps,
                )
                del client, local_data, client_data
                gc.collect()
                torch.cuda.empty_cache()
            except Exception:
                errors[client_id] = traceback.format_exc()
                print(f"[Worker GPU {gpu_id}] client {client_id} failed:\n{errors[client_id]}", flush=True)
    except Exception:
        message = traceback.format_exc()
        for client_id in client_ids:
            errors.setdefault(client_id, message)
        print(f"[Worker GPU {gpu_id}] initialization failed:\n{message}", flush=True)
    finally:
        if model is not None:
            del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # One result per worker makes queue collection deterministic, including
        # when a worker has more than one assigned client.
        result_queue.put((gpu_id, results, errors))
