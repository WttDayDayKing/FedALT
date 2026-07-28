#!/usr/bin/env python
"""FedCKD entry point: original-data contribution-aware dual-LoRA training."""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from fedckd_server import FedCKDServer


def build_parser():
    parser = argparse.ArgumentParser(description="FedCKD contribution-aware knowledge decoupling")
    parser.add_argument("--model_name", type=str, default="/data/dataset/models/Llama-2-7b-hf")
    parser.add_argument("--data_path", type=str, default="/data/wtt/2026/FedDPA/data/dataset1")
    parser.add_argument("--result_dir", type=str, default="./results")
    parser.add_argument("--dataset", type=str, default="flan1")
    parser.add_argument("--method", type=str, default="fedckd")
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--local_epochs", type=int, default=5)
    parser.add_argument("--client_num", type=int, default=8)
    parser.add_argument("--partition_dir", type=str, default="8")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_gpus", type=int, default=4)
    parser.add_argument("--gpu_ids", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Use 1 for exact sample-level contribution scoring")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Must be 1: FedCKD selects and masks gradients per micro-batch.",
    )
    parser.set_defaults(gradient_checkpointing=True)
    parser.add_argument("--no_gradient_checkpointing", dest="gradient_checkpointing", action="store_false")

    parser.add_argument("--fedckd_sketch_dim", type=int, default=1024)
    parser.add_argument("--fedckd_temperature", type=float, default=0.5)
    parser.add_argument("--fedckd_ema", type=float, default=0.9)
    parser.add_argument("--fedckd_alpha", type=float, default=1.0 / 3.0,
                        help="Weight of global-gradient consistency")
    parser.add_argument("--fedckd_beta", type=float, default=1.0 / 3.0,
                        help="Weight of personalization gain; gamma=1-alpha-beta")
    parser.add_argument("--fedckd_orth_lambda", type=float, default=1e-3)
    parser.add_argument("--fedckd_prox_lambda", type=float, default=1e-3)
    parser.add_argument("--fedckd_phase_steps", type=int, default=1,
                        help="Number of micro-batches per global/private phase")
    parser.add_argument("--fedckd_warmup_steps", type=int, default=20,
                        help="Balanced micro-batches before alternating optimization")
    parser.add_argument(
        "--fedckd_routing_mode",
        choices=["static", "adaptive"],
        default="adaptive",
        help="Use fixed 0.5/0.5 branch weights or contribution-aware routing.",
    )
    parser.add_argument(
        "--fedckd_update_mode",
        choices=["joint", "alternating"],
        default="alternating",
        help="Update both LoRA branches each micro-batch or alternate their updates.",
    )
    parser.add_argument(
        "--fedckd_persist_routing_statistics",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep each client's gain and route EMA across communication rounds.",
    )
    parser.add_argument(
        "--fedckd_aggregation",
        choices=["uniform", "global_mass"],
        default="global_mass",
        help="Use equal client weights or contribution-weighted global LoRA aggregation.",
    )
    return parser


def validate(args):
    for name in ("rounds", "local_epochs", "client_num", "rank", "batch_size",
                 "gradient_accumulation_steps", "fedckd_sketch_dim", "fedckd_phase_steps"):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name} must be positive")
    if args.fedckd_temperature <= 0:
        raise ValueError("--fedckd_temperature must be positive")
    if args.gradient_accumulation_steps != 1:
        raise ValueError(
            "FedCKD requires --gradient_accumulation_steps=1 so contribution "
            "scores are computed from unaccumulated micro-batch gradients."
        )
    if not 0 <= args.fedckd_ema < 1:
        raise ValueError("--fedckd_ema must be in [0, 1)")
    gamma = 1.0 - args.fedckd_alpha - args.fedckd_beta
    if min(args.fedckd_alpha, args.fedckd_beta, gamma) < 0:
        raise ValueError("FedCKD indicator weights must be non-negative and sum to one")
    if args.fedckd_orth_lambda < 0 or args.fedckd_prox_lambda < 0:
        raise ValueError("FedCKD regularization strengths must be non-negative")


def main():
    args = build_parser().parse_args()
    validate(args)
    checkpoint_dir = Path(args.result_dir) / args.dataset / "checkpoints" / args.method / "E2"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    config_path = checkpoint_dir / f"training_config_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.json"
    with config_path.open("w", encoding="utf-8") as handle:
        json.dump({
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "command": [sys.executable, *sys.argv],
            "arguments": vars(args),
        }, handle, indent=2, sort_keys=True)
    FedCKDServer(args).train(str(checkpoint_dir))


if __name__ == "__main__":
    main()
