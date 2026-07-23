#!/usr/bin/env python
# coding: utf-8
"""Run FedALT inference and ROUGE evaluation from saved LoRA checkpoints."""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List

# Must be set before importing PyTorch.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from tqdm import tqdm

from util import load_base_model, set_lora_state_dict
from utils.prompter import Prompter


def parse_client_ids(value: str, client_num: int) -> List[int]:
    if value == "all":
        return list(range(client_num))
    try:
        client_ids = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise ValueError("--client_ids must be 'all' or a comma-separated list such as 0,2,3.") from error
    if not client_ids or len(set(client_ids)) != len(client_ids):
        raise ValueError("--client_ids must contain one or more unique client IDs.")
    invalid = [client_id for client_id in client_ids if client_id < 0 or client_id >= client_num]
    if invalid:
        raise ValueError(f"Client IDs {invalid} are outside 0..{client_num - 1}.")
    return client_ids


def default_checkpoint_path(args) -> Path:
    candidates = [
        Path(args.result_dir) / args.dataset / "checkpoints" / args.method / "global.pt",
        # Compatibility with checkpoints produced by the earlier training entry point.
        Path(args.result_dir) / args.dataset / "checkpoints" / "method" / "global.pt",
        Path(args.result_dir) / args.dataset / "checkpoints" / "global.pt",
    ]
    for path in candidates:
        if path.is_file():
            return path
    formatted = "\n".join(str(path) for path in candidates)
    raise FileNotFoundError(
        "Could not find a global checkpoint. Pass --checkpoint_path explicitly, or create one at:\n"
        + formatted
    )


def load_client_states(checkpoint_path: Path, client_ids: Iterable[int]) -> Dict[int, Dict[str, torch.Tensor]]:
    """Load either a global per-client checkpoint or one raw client state."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    client_ids = list(client_ids)

    if isinstance(checkpoint, list):
        missing = [client_id for client_id in client_ids if client_id >= len(checkpoint)]
        if missing:
            raise ValueError(
                f"Checkpoint contains {len(checkpoint)} client states, but requested clients {missing}."
            )
        return {client_id: checkpoint[client_id] for client_id in client_ids}

    if isinstance(checkpoint, dict):
        # Historical client checkpoints are stored as {'client_id': ..., 'params': {...}}.
        state = checkpoint.get("params", checkpoint)
        if not all(isinstance(value, torch.Tensor) for value in state.values()):
            raise ValueError(f"Unsupported checkpoint format in {checkpoint_path}.")
        if len(client_ids) != 1:
            raise ValueError(
                "A single-client checkpoint was supplied. Select exactly one client, e.g. "
                "--client_ids 0, or use a global.pt checkpoint for all clients."
            )
        return {client_ids[0]: state}

    raise ValueError(f"Unsupported checkpoint type {type(checkpoint).__name__} in {checkpoint_path}.")


def load_client_checkpoints(checkpoint_dir: Path, client_ids: Iterable[int]):
    """Load ``client_<id>.pt`` for each selected client from one directory."""
    if not checkpoint_dir.is_dir():
        raise NotADirectoryError(f"Checkpoint directory not found: {checkpoint_dir}")

    states = {}
    paths = {}
    for client_id in client_ids:
        checkpoint_path = checkpoint_dir / f"client_{client_id}.pt"
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Client {client_id} checkpoint not found: {checkpoint_path}")
        states[client_id] = load_client_states(checkpoint_path, [client_id])[client_id]
        paths[client_id] = str(checkpoint_path)
    return states, paths


def read_test_records(test_file: Path) -> List[dict]:
    if not test_file.is_file():
        raise FileNotFoundError(f"Test file not found: {test_file}")
    records = []
    with test_file.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if "instruction" not in record or "output" not in record:
                raise ValueError(f"{test_file}:{line_number} requires 'instruction' and 'output' fields.")
            records.append(record)
    if not records:
        raise ValueError(f"Test file has no examples: {test_file}")
    return records


def generate_predictions(model, tokenizer, prompter, records, args) -> List[str]:
    model.eval()
    device = next(model.parameters()).device
    predictions: List[str] = []

    for start in tqdm(range(0, len(records), args.eval_batch_size), desc="Generating", leave=False):
        batch = records[start : start + args.eval_batch_size]
        prompts = [
            prompter.generate_prompt(record["instruction"], record.get("input") or None)
            for record in batch
        ]
        encoded = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_input_length,
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        with torch.inference_mode():
            sequences = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                do_sample=False,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        # The generated sequence starts with the padded model input. Decoding
        # only the suffix avoids accidentally treating the instruction as part
        # of the answer.
        generated_tokens = sequences[:, input_ids.shape[1] :]
        predictions.extend(tokenizer.batch_decode(generated_tokens, skip_special_tokens=True))

    return [prediction.strip() for prediction in predictions]


def score_records(records: List[dict], predictions: List[str]) -> Dict[str, Dict[str, float]]:
    if len(records) != len(predictions):
        raise ValueError(f"Prediction count ({len(predictions)}) does not match test count ({len(records)}).")
    try:
        from rouge_score import rouge_scorer
    except ImportError as error:
        raise ImportError("ROUGE evaluation requires rouge-score. Install dependencies with: pip install -r requirements.txt") from error
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    by_category = defaultdict(list)
    for record, prediction in zip(records, predictions):
        by_category[record.get("category", "uncategorized")].append((prediction, record["output"]))

    def average(pairs):
        values = {name: [] for name in ("rouge1", "rouge2", "rougeL")}
        for prediction, reference in pairs:
            scores = scorer.score(reference, prediction)
            for name in values:
                values[name].append(scores[name].fmeasure)
        return {name: float(sum(scores) / len(scores)) if scores else 0.0 for name, scores in values.items()}

    result = {category: average(pairs) for category, pairs in sorted(by_category.items())}
    result["total"] = average([(prediction, record["output"]) for record, prediction in zip(records, predictions)])
    return result


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(description="FedALT inference and ROUGE evaluation")
    parser.add_argument("--model_name", type=str, default="/data/dataset/models/Llama-2-7b-hf")
    parser.add_argument("--data_path", type=str, default="/data/wtt/2026/FedDPA/data/dataset1")
    parser.add_argument("--result_dir", type=str, default="./results")
    parser.add_argument("--dataset", type=str, default="flan1")
    parser.add_argument("--method", type=str, default="fedalt")
    parser.add_argument("--checkpoint_path", type=Path, default=None, help="global.pt or a single client_*.pt")
    parser.add_argument(
        "--checkpoint_dir",
        type=Path,
        default="/data/wtt/2026/FedALT/results/flan1/checkpoints/fedalt",
        help="Directory containing client_<id>.pt files; loads one checkpoint per selected client",
    )
    parser.add_argument("--client_num", type=int, default=8)
    parser.add_argument("--client_ids", type=str, default="all", help="'all' or comma-separated IDs")
    parser.add_argument("--test_dir", type=Path, default=None, help="Defaults to <data_path>/test")
    parser.add_argument("--output_dir", type=Path, default=None, help="Defaults to <result_dir>/<dataset>/eval/<method>")
    parser.add_argument("--run_name", type=str, default=None, help="Name used in prediction and score filenames")
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--lora_n", type=int, default=1)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--gpu_id", type=int, default=0, help="Visible CUDA device used for inference")
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--max_input_length", type=int, default=512)
    parser.add_argument("--max_new_tokens", type=int, default=80)
    parser.add_argument("--num_beams", type=int, default=1)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; 8-bit FedALT inference requires a CUDA GPU.")
    if args.gpu_id < 0 or args.gpu_id >= torch.cuda.device_count():
        raise ValueError(f"--gpu_id must be in 0..{torch.cuda.device_count() - 1}.")
    if args.eval_batch_size < 1 or args.max_input_length < 1 or args.max_new_tokens < 1 or args.num_beams < 1:
        raise ValueError("Batch size, input length, output length, and beam count must all be positive.")

    client_ids = parse_client_ids(args.client_ids, args.client_num)
    if args.checkpoint_path and args.checkpoint_dir:
        raise ValueError("Use either --checkpoint_path or --checkpoint_dir, not both.")
    if args.checkpoint_dir:
        client_states, checkpoint_description = load_client_checkpoints(args.checkpoint_dir, client_ids)
        default_run_name = "client_checkpoints"
    else:
        checkpoint_path = args.checkpoint_path or default_checkpoint_path(args)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        client_states = load_client_states(checkpoint_path, client_ids)
        checkpoint_description = str(checkpoint_path)
        default_run_name = checkpoint_path.stem

    test_dir = args.test_dir or Path(args.data_path) / "test"
    output_dir = args.output_dir or Path(args.result_dir) / args.dataset / "eval" / args.method
    run_name = args.run_name or default_run_name
    torch.cuda.set_device(args.gpu_id)
    device = torch.device(f"cuda:{args.gpu_id}")
    model, tokenizer = load_base_model(
        args.model_name,
        args.rank,
        args.lora_alpha,
        args.lora_n,
        device=device,
        gradient_checkpointing=False,
    )
    prompter = Prompter("alpaca_short")
    summary = {"checkpoint": checkpoint_description, "clients": {}}

    try:
        for client_id in client_ids:
            set_lora_state_dict(model, client_states[client_id])
            test_file = test_dir / f"local_testing_{client_id}.jsonl"
            records = read_test_records(test_file)
            print(f"\n[Inference] client {client_id}: {len(records)} examples", flush=True)
            predictions = generate_predictions(model, tokenizer, prompter, records, args)
            scores = score_records(records, predictions)

            prediction_file = output_dir / f"eval_client{client_id}_{run_name}.jsonl"
            prediction_file.parent.mkdir(parents=True, exist_ok=True)
            with prediction_file.open("w", encoding="utf-8") as handle:
                for record, prediction in zip(records, predictions):
                    handle.write(json.dumps({
                        "text": record["instruction"],
                        "answer": prediction,
                        "reference": record["output"],
                        "category": record.get("category", ""),
                        "client_id": client_id,
                    }, ensure_ascii=False) + "\n")
            score_file = output_dir / f"scores_client{client_id}_{run_name}.json"
            write_json(score_file, scores)
            summary["clients"][str(client_id)] = {"scores": scores, "prediction_file": str(prediction_file)}
            print(f"[Inference] client {client_id} total ROUGE: {scores['total']}", flush=True)
    finally:
        del model
        torch.cuda.empty_cache()

    summary_file = output_dir / f"summary_{run_name}.json"
    write_json(summary_file, summary)
    print(f"\nSaved evaluation summary to {summary_file}", flush=True)


if __name__ == "__main__":
    main()
