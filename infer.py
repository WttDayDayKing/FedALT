
import os

import json
import argparse
from datasets import load_dataset
from transformers import LlamaTokenizer
from tqdm import tqdm
import torch
from utils.prompter import Prompter
from util import prepare_local_dataset, get_round_specific_paths, print_gpu_memory
from client import Client
from server import Server
from peft import LoraConfig, TaskType, get_peft_model

def main():
    parser = argparse.ArgumentParser(description='FedALT: Federated Fine-Tuning with Adaptive Local Training')
    parser.add_argument('--model_name', type=str, default='/data/dataset/models/Llama-2-7b-hf', help='Base model name')
    parser.add_argument('--data_path', type=str, default='/data/wtt/2026/FedDPA/data/dataset1', help='Path to training data directory')
    parser.add_argument('--result_dir', type=str, default='./results', help='Directory to save results')
    parser.add_argument('--rounds', type=int, default=20, help='Number of global communication rounds')
    parser.add_argument('--local_epochs', type=int, default=10, help='Number of local training epochs')
    parser.add_argument('--client_num', type=int, default=1, help='Number of clients')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--rank', type=int, default=8, help='LoRA rank')
    parser.add_argument('--dataset',default="flan1",type=str)
    parser.add_argument('--lora_n',default=1)

    args = parser.parse_args()

    client_num = args.client_num
    model_name = args.model_name
    data_path = args.data_path
    dataset=args.dataset
    result_dir=args.result_dir

    # Initialize tokenizer and prompter
    prompter = Prompter("alpaca_short")
    tokenizer = LlamaTokenizer.from_pretrained(model_name)
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"

    # Initialize clients
    clients = []
    for client_id in range(client_num):
        local_data_path = os.path.join(data_path, "8/" f"local_training_{client_id}.json")
        client_data = load_dataset("json", data_files=local_data_path)
        local_data = prepare_local_dataset(client_data, tokenizer, prompter)
        clients.append(Client(
            client_id, 
            local_data, 
            tokenizer,
            prompter,
            model_name, 
            rank=args.rank, 
            lora_n=1, 
            asymmetric=False
        ))

    save_dir=os.path.join(result_dir,"eval")
    os.makedirs(save_dir, exist_ok=True)

    test_files = {
        client_id: os.path.join(data_path, "test", f"local_testing_{client_id}.jsonl")
        for client_id in range(len(clients))
    }
    eval_files = {
        client_id: f"{save_dir}/eval_client{client_id}_fedavg.jsonl"
        for client_id in range(len(clients))
    }
    score_files = {
        client_id: f"{save_dir}/scores_client{client_id}_fedavg.json"
        for client_id in range(len(clients))
    }

    lora_dir=os.path.join(result_dir,dataset,"checkpoints")
    for client in tqdm(clients, desc="Client Training"):
        lora_path=os.path.join(lora_dir,f"client_{client.client_id}.pt")
        client.load_model()
        client.load_params(lora_path)
        client.evaluate_model(test_files[client.client_id], eval_files[client.client_id])
        scores=client.calculate_rouge_scores(test_files[client.client_id], eval_files[client.client_id],score_files[client.client_id])
        print(scores)

if __name__ == "__main__":
    main()
        

