#!/usr/bin/env python
# coding: utf-8
"""FedALT multi-client, multi-GPU training entry point."""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# This must be configured before importing PyTorch through ``server``.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from server import Server


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="FedALT: Federated Fine-Tuning with Adaptive Local Training"
    )
    parser.add_argument("--model_name", type=str, default="/data/dataset/models/Llama-2-7b-hf")
    parser.add_argument(
        "--data_path",
        type=str,
        default="/data/wtt/2026/FedDPA/data/dataset1",
        help="Root containing the partition directory and test directory",
    )
    parser.add_argument("--result_dir", type=str, default="./results")
    parser.add_argument("--dataset", type=str, default="flan1", help="Result subdirectory name")
    parser.add_argument("--method", type=str, default="feddcr", help="Checkpoint subdirectory name")

    parser.add_argument("--rounds", type=int, default=20, help="Global communication rounds")
    parser.add_argument("--local_epochs", type=int, default=5, help="Local epochs in each round")
    parser.add_argument("--client_num", type=int, default=8, help="Number of participating clients")
    parser.add_argument("--partition_dir", type=str, default="8", help="Contains local_training_<id>.json")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--rank", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora_n", type=int, default=2, help="Number of LoRA adapters")
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42, help="Shared initial LoRA seed")
    parser.add_argument("--feddcr", action="store_true", help="Enable data-function-aware FedDCR routing")
    parser.add_argument("--feddcr_sketch_dim", type=int, default=1024)
    parser.add_argument("--feddcr_temperature", type=float, default=0.7)
    parser.add_argument("--feddcr_residual_penalty", type=float, default=0.5)
    parser.add_argument("--feddcr_ema", type=float, default=0.9)
    parser.add_argument("--feddcr_clip_norm", type=float, default=1.0)
    parser.add_argument("--feddcr_learnability_weight", type=float, default=0.1)
    parser.add_argument("--feddcr_stability_weight", type=float, default=0.1)
    parser.add_argument("--feddcr_conflict_variance_weight", type=float, default=0.0)
    parser.add_argument("--feddcr_score_history_size", type=int, default=16)

    parser.add_argument("--num_gpus", type=int, default=4, help="Number of visible GPUs to use")
    parser.add_argument(
        "--gpu_ids",
        type=str,
        default=None,
        help="Optional comma-separated visible GPU indices, for example 0,2,3",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Per-GPU batch size; 1 is the safe default for 24GB GPUs",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=8,
        help="Number of local batches accumulated before an optimizer step",
    )
    parser.set_defaults(gradient_checkpointing=True)
    parser.add_argument(
        "--no_gradient_checkpointing",
        dest="gradient_checkpointing",
        action="store_false",
        help="Disable activation checkpointing; this requires substantially more GPU memory",
    )
    return parser


def validate_args(args) -> None:
    positive_values = {
        "rounds": args.rounds,
        "local_epochs": args.local_epochs,
        "client_num": args.client_num,
        "rank": args.rank,
        "lora_n": args.lora_n,
        "lora_alpha": args.lora_alpha,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "feddcr_sketch_dim": args.feddcr_sketch_dim,
        "feddcr_clip_norm": args.feddcr_clip_norm,
        "feddcr_score_history_size": args.feddcr_score_history_size,
    }
    invalid = [name for name, value in positive_values.items() if value < 1]
    if invalid:
        raise ValueError("These arguments must be positive: " + ", ".join(invalid))
    if args.feddcr and args.lora_n != 2:
        raise ValueError("FedDCR requires exactly --lora_n 2 (shared and private)")
    if args.feddcr_temperature <= 0 or not 0 <= args.feddcr_ema < 1:
        raise ValueError("FedDCR temperature must be positive and EMA must be in [0, 1)")
    if any(value < 0 for value in (
        args.feddcr_learnability_weight,
        args.feddcr_stability_weight,
        args.feddcr_conflict_variance_weight,
    )):
        raise ValueError("FedDCR score weights must be non-negative")


def save_run_config(args, checkpoint_dir: Path) -> Path:
    """Persist a non-overwriting snapshot next to the model checkpoints."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    config_path = checkpoint_dir / f"training_config_{timestamp}.json"
    config = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv],
        "working_directory": str(Path.cwd()),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "arguments": vars(args),
    }
    with config_path.open("w", encoding="utf-8") as handle:
        json.dump(config, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    return config_path


def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)

    checkpoint_dir = Path(args.result_dir) / args.dataset / "checkpoints" / args.method / "v3"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    config_path = save_run_config(args, checkpoint_dir)
    print(f"[FedALT] checkpoints: {checkpoint_dir}", flush=True)
    print(f"[FedALT] configuration: {config_path}", flush=True)

    # ``Server`` owns data validation, GPU selection, worker creation, and
    # federated aggregation. Do not call the removed legacy setup_clients().
    server = Server(args)
    server.train(clients_checkpoints=str(checkpoint_dir))


if __name__ == "__main__":
    main()
