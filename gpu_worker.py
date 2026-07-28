"""Persistent single-GPU workers for dynamically scheduled FedALT clients."""

import gc
import io
import os
import traceback

import torch

from client import Client
from feddcr import state_update_sketch
from util import get_lora_state_dict, load_base_model, load_dataset, prepare_local_dataset
from utils.prompter import Prompter


def _data_file(args, client_id: int) -> str:
    return os.path.join(args.data_path, args.partition_dir, f"local_training_{client_id}.json")


def _private_state_file(args, client_id: int) -> str:
    return os.path.join(args.feddcr_private_state_dir, f"client_{client_id}.pt")


def _feddcr_private_state(state):
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in state.items()
        if "lora_A1" in name or "lora_B1" in name or "lora_route" in name
    }


def _feddcr_shared_state(state):
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in state.items()
        if "lora_A0" in name or "lora_B0" in name
    }


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


def client_worker(gpu_id, args, client_states, prototypes, task_queue, result_queue):
    """Keep one model on ``gpu_id`` and pull clients from ``task_queue``.

    A worker receives a new client only after it has completely released the
    previous client's trainer state. Faster GPUs therefore keep working rather
    than waiting for a statically assigned slow client on another GPU.
    """
    model = None
    try:
        physical_device = _cuda_visible_device(gpu_id)
        os.environ["CUDA_VISIBLE_DEVICES"] = physical_device
        torch.cuda.set_device(0)
        device = torch.device("cuda:0")
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

        print(
            f"[Worker GPU {gpu_id} (CUDA_VISIBLE_DEVICES={physical_device})] ready",
            flush=True,
        )
        model, tokenizer = load_base_model(
            args.model_name,
            args.rank,
            args.lora_alpha,
            args.lora_n,
            device=device,
            gradient_checkpointing=args.gradient_checkpointing,
            use_router=not args.feddcr,
        )
        initial_state = get_lora_state_dict(model)
        prompter = Prompter("alpaca_short")

        while True:
            client_id = task_queue.get()
            if client_id is None:
                break

            print(f"[Worker GPU {gpu_id}] training client {client_id}", flush=True)
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
                    use_router=not args.feddcr,
                )
                if args.feddcr:
                    # Adapter 1 and the token router never cross the client
                    # boundary.  A real deployment keeps this file on the
                    # client; the local simulator uses this directory to
                    # preserve it while GPU worker processes are restarted.
                    private_path = _private_state_file(args, client_id)
                    private_state = (
                        torch.load(private_path, map_location="cpu")
                        if os.path.isfile(private_path)
                        else _feddcr_private_state(initial_state)
                    )
                    starting_state = dict(private_state)
                    starting_state.update(client_states.get(client_id, _feddcr_shared_state(initial_state)))
                else:
                    starting_state = client_states.get(client_id, initial_state)
                trained_state, route_metrics = client.local_training(
                    model=model,
                    global_state=starting_state,
                    lr=args.lr,
                    epochs=args.local_epochs,
                    batch_size=args.batch_size,
                    gradient_accumulation_steps=args.gradient_accumulation_steps,
                    feddcr_config={
                        "sketch_dim": args.feddcr_sketch_dim,
                        "temperature": args.feddcr_temperature,
                        "residual_penalty": args.feddcr_residual_penalty,
                        "ema": args.feddcr_ema,
                        "learnability_weight": args.feddcr_learnability_weight,
                        "stability_weight": args.feddcr_stability_weight,
                        "conflict_variance_weight": args.feddcr_conflict_variance_weight,
                        "score_history_size": args.feddcr_score_history_size,
                    } if args.feddcr else None,
                    global_prototype=prototypes.get("global"),
                    local_prototype=prototypes.get("local", {}).get(client_id),
                )
                if args.feddcr:
                    os.makedirs(args.feddcr_private_state_dir, exist_ok=True)
                    torch.save(_feddcr_private_state(trained_state), private_path)
                    uploaded_state = _feddcr_shared_state(trained_state)
                else:
                    uploaded_state = trained_state
                payload = {
                    "state": uploaded_state,
                    "num_examples": len(local_data),
                    "sketch": state_update_sketch(
                        starting_state, trained_state, args.feddcr_sketch_dim, "global",
                        args.feddcr_clip_norm,
                    ) if args.feddcr else None,
                    "metrics": route_metrics,
                }
                # Do not put tensors directly on a multiprocessing queue.
                # PyTorch otherwise shares them through file descriptors; a
                # fast worker can exit before the parent rebuilds its final
                # tensor storage. Bytes are self-contained and race-free.
                payload_buffer = io.BytesIO()
                torch.save(payload, payload_buffer)
                result_queue.put(
                    ("client_result", gpu_id, client_id, payload_buffer.getvalue(), None)
                )
                print(f"[Worker GPU {gpu_id}] completed client {client_id}", flush=True)
                del client, local_data, client_data
                gc.collect()
                torch.cuda.empty_cache()
            except Exception:
                error = traceback.format_exc()
                print(f"[Worker GPU {gpu_id}] client {client_id} failed:\n{error}", flush=True)
                result_queue.put(("client_result", gpu_id, client_id, None, error))
    except Exception:
        error = traceback.format_exc()
        print(f"[Worker GPU {gpu_id}] initialization failed:\n{error}", flush=True)
        result_queue.put(("worker_error", gpu_id, None, None, error))
    finally:
        if model is not None:
            del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        result_queue.put(("worker_done", gpu_id, None, None, None))
