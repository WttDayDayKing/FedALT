# FedALT: Federated Fine-Tuning through Adaptive Local Training with Rest-of-World LoRA

[![AAAI 2026](https://img.shields.io/badge/AAAI-2026-blue.svg)](https://aaai.org/conference/aaai/aaai-26/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Official implementation of **FedALT: Federated Fine-Tuning through Adaptive Local Training with Rest-of-World LoRA**, accepted by AAAI 2026.

**Authors**: Jieming Bian*, Lei Wang*, Letian Zhang, Jie Xu (*Equal contribution)

## Overview

FedALT is a novel federated learning framework for efficient fine-tuning of large language models (LLMs) in heterogeneous data environments. Our approach introduces a dual-LoRA architecture with intelligent routing mechanisms:

- **Dual-LoRA Architecture**: Each client maintains two LoRA adapters:
  - **LoRA0 (Rest-of-World)**: Aggregated knowledge from all other clients
  - **LoRA1 (Local)**: Client-specific personalized adapter
- **MOE-based Adaptive Routing**: Dynamic token-level selection between LoRA adapters using softmax-weighted routing
- **Rest-of-World Aggregation**: Server aggregates local LoRA parameters from other clients to create global knowledge base
- **Local Training**: Only local LoRA and routing parameters are trained

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Federated Server                    │
│  ┌──────────────────────────────────────────────────┐  │
│  │  Rest-of-World LoRA Aggregation                  │  │
│  │  • Aggregate parameters from other clients       │  │
│  │  • Adaptive MOE routing mechanism                │  │
│  └──────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
                          │
        ┌─────────────────┼─────────────────┐
        ▼                 ▼                 ▼
   ┌─────────┐       ┌─────────┐       ┌─────────┐
   │Client 1 │       │Client 2 │  ...  │Client N │
   │ ┌─────┐ │       │ ┌─────┐ │       │ ┌─────┐ │
   │ │LoRA0│ │       │ │LoRA0│ │       │ │LoRA0│ │
   │ │(RoW)│ │       │ │(RoW)│ │       │ │(RoW)│ │
   │ └─────┘ │       │ └─────┘ │       │ └─────┘ │
   │ ┌─────┐ │       │ ┌─────┐ │       │ ┌─────┐ │
   │ │LoRA1│ │       │ │LoRA1│ │       │ │LoRA1│ │
   │ │(Local)│       │ │(Local)│       │ │(Local)│
   │ └─────┘ │       │ └─────┘ │       │ └─────┘ │
   └─────────┘       └─────────┘       └─────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- CUDA 11.0+ (for GPU support)
- 16GB+ GPU memory recommended

### Installation

```bash
# Clone the repository
git clone https://github.com/jmbian/FedALT.git
cd FedALT

# Install dependencies
pip install -r requirements.txt
```

### Data Preparation

we use FedDPA/data/flan/dataset1

Organize your data in the following structure:

```
train/
├── local_training_0.json
├── local_training_1.json
├── ...
└── local_training_7.json

test/
├── local_testing_0.jsonl
├── local_testing_1.jsonl
├── ...
└── local_testing_7.jsonl
```

Each training file should be in JSON format:
```json
[
  {
    "instruction": "Premise: a woman wearing red, climbs up the giant rock.\n\nHypothesis: He eats his own shoe\n\n.Given the premise, can we conclude the hypothesis?\n\nOPTIONS:\n- yes\n- it is not possible to tell\n- no",
    "output": "no",
    "task": "snli",
    "category": "natural language inference"
  },
]
```

Each test file should be in JSONL format:
```json
{"instruction": "If \"A Woman with glasses on, cooking.\", does this mean that \"A man with glasses on, cooking.\"?\n\nOPTIONS:\n- yes\n- it is not possible to tell\n- no", "output": "no", "task": "snli", "category": "natural language inference"}
```

### Training

```bash
python main.py \
  --model_name meta-llama/Llama-2-7b-hf \
  --data_path ./data \
  --result_dir ./results \
  --rounds 20 \
  --local_epochs 5 \
  --client_num 8 \
  --lr 3e-4 \
  --num_gpus 4
```

Each training launch writes a timestamped `training_config_*.json` alongside
the checkpoints. It records the effective command, all arguments, working
directory, and `CUDA_VISIBLE_DEVICES` for reproducibility.

### Multi-client, multi-GPU training

`main.py` starts one worker process per selected GPU. Each worker keeps one
quantized model resident on its GPU and trains its assigned clients in sequence;
workers run concurrently. Therefore 8 clients on 4 GPUs run as four parallel
workers with two clients per worker, without loading two models on one GPU.

Use the first visible GPUs with `--num_gpus`, or select visible GPU indices
explicitly with `--gpu_ids`:

```bash
python main.py --client_num 8 --num_gpus 4 --partition_dir 8
python main.py --client_num 8 --gpu_ids 0,2,3 --partition_dir 8
```

The process must see all selected GPUs (for example, do not restrict them out
through `CUDA_VISIBLE_DEVICES`). Client training data is expected at
`<data_path>/<partition_dir>/local_training_<client_id>.json`.

For 24GB GPUs, the default per-GPU batch size is `1` and activation checkpointing
is enabled. The default `--gradient_accumulation_steps 8` retains an effective
batch size of 8 without keeping eight samples' activations on the GPU. If the
allocator reports fragmentation, `main.py` enables
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` before importing PyTorch.

## Project Structure

```
FedALT_code/
├── main.py              # Main training script
├── client.py            # Client implementation
├── server.py            # Server implementation
├── utils.py             # Utility functions
├── utils/
│   ├── prompter.py      # Prompt templates
│   └── callbacks.py     # Training callbacks
├── peft/                # Modified PEFT library
│   └── tuners/
│       └── lora.py      # Custom LoRA implementation
├── README.md            # This file
└── requirements.txt     # Dependencies
```

## Evaluation

Run inference and ROUGE evaluation from the final per-client `global.pt`:

```bash
python infer.py \
  --model_name meta-llama/Llama-2-7b-hf \
  --data_path ./data \
  --result_dir ./results \
  --dataset flan1 \
  --method fedavg \
  --client_num 8
```

The script automatically loads
`<result_dir>/<dataset>/checkpoints/<method>/global.pt`, reuses one 8-bit model
on `--gpu_id 0` by default, applies the state corresponding to each client, and
stores predictions, per-client ROUGE scores, and a summary under
`<result_dir>/<dataset>/eval/<method>/`. To use each client's independently
saved adapter, pass its containing directory; the script then loads
`client_<id>.pt` for every selected client:

```bash
python infer.py \
  --checkpoint_dir /data/wtt/2026/FedALT/results/flan1/checkpoints/method \
  --client_ids all \
  --client_num 4
```

To evaluate one saved client adapter, pass
`--checkpoint_path .../client_0.pt --client_ids 0`.

The framework evaluates models using ROUGE metrics across multiple NLP tasks:

- Text Classification (AG News, Sentiment140)
- Natural Language Inference (SNLI)
- Question Answering (OpenBookQA)
- Text Generation (CommonGen, Story Cloze)
- And more...

Results are saved in:
- `results/eval_client{id}_round{n}.jsonl`: Generated predictions
- `results/scores_client{id}_round{n}.json`: ROUGE scores
- `results/final_scores.json`: Aggregated final scores


## Citation

If you find this work useful, please cite our paper:

```bibtex
@article{bian2025fedalt,
  title={FedALT: Federated Fine-Tuning through Adaptive Local Training with Rest-of-World LoRA},
  author={Bian, Jieming and Wang, Lei and Zhang, Letian and Xu, Jie},
  journal={arXiv preprint arXiv:2503.11880},
  year={2025}
}
```


## 🔗 Link

- [Paper (arXiv)](https://arxiv.org/abs/2503.11880)
---
