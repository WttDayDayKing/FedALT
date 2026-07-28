#!/usr/bin/env python
# coding: utf-8
import sys
import os

import torch
import json
import gc
from typing import Dict, List
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import (
    BitsAndBytesConfig,
    LlamaForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
    GenerationConfig
)
from peft import LoraConfig, TaskType, get_peft_model
from util import prepare_model_for_kbit_training, EvalDataset, write_file,set_lora_state_dict,get_lora_state_dict
from feddcr import FedDCRTrainerMixin


class FedDCRTrainer(FedDCRTrainerMixin, Trainer):
    pass


def set_trainable_lora_parameters(model, lora_n: int, train_global: bool = False) -> None:
    """Select the parameters owned by the current federated method.

    ``lora_n == 1`` is the FedAvg baseline: the sole adapter (and its
    optional route tensor) is trained and later averaged by the server.
    Standard FedALT freezes adapter 0 and trains adapter 1 plus the route
    network. FedDCR instead trains both adapters: its gradient-space router
    assigns every micro-batch's update to the shared and/or private branch.
    """
    for name, parameter in model.named_parameters():
        if lora_n == 1:
            parameter.requires_grad = (
                "lora_A0" in name or "lora_B0" in name or "lora_route" in name
            )
        else:
            parameter.requires_grad = (
                "lora_A1" in name or "lora_B1" in name or "lora_route" in name
                or (train_global and ("lora_A0" in name or "lora_B0" in name))
            )


class Client:
    """Federated Learning Client with LoRA fine-tuning."""
    
    def __init__(self, client_id, client_dataset, tokenizer, prompter, model_name, device,
                 rank=8, lora_n=4, asymmetric=False, cache_path='/data/wtt/2026/FedALT/output',
                 gradient_checkpointing=True, use_router=True):
        self.client_id = client_id
        self.client_dataset = client_dataset
        self.tokenizer = tokenizer
        self.prompter = prompter
        self.model_name = model_name
        self.rank = rank
        self.lora_n = lora_n
        self.asymmetric = asymmetric
        self.cache_path = cache_path
        self.local_model = None
        self.current_params = None
        self.device=device
        self.gradient_checkpointing = gradient_checkpointing
        self.use_router = use_router

    def load_model(self):
        """Load model with quantization and LoRA configuration."""
        if self.local_model is None:
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
                bnb_8bit_compute_dtype=torch.float16,
                bnb_8bit_use_double_quant=True
            )
            
            self.local_model = LlamaForCausalLM.from_pretrained(
                self.model_name,
                cache_dir=self.cache_path,
                torch_dtype=torch.float16,
                # load_in_8bit=True,
                device_map="auto",
                quantization_config=quantization_config,
            )
            
            self.local_model = prepare_model_for_kbit_training(
                self.local_model,
                use_gradient_checkpointing=self.gradient_checkpointing
            )
            if self.gradient_checkpointing:
                self.local_model.config.use_cache = False
            
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                target_modules=["q_proj", "v_proj"],
                inference_mode=False,
                r=self.rank,
                lora_alpha=32,
                lora_dropout=0.05,
                lora_nums=self.lora_n,
                asymmetric=self.asymmetric,
                use_router=self.use_router,
                bias="none"
            )       
            
            self.local_model = get_peft_model(self.local_model, peft_config)
            
            if self.current_params is not None:
                self.load_params(self.current_params)
    
    def unload_model(self):
        """Unload model to free GPU memory."""
        if self.local_model is not None:
            self.current_params = self.get_lora_params()['params']
            del self.local_model
            self.local_model = None
            torch.cuda.empty_cache()
            gc.collect()
    
    def get_lora_params(self):
        """Extract LoRA parameters from the model."""
        lora_params = {
            'client_id': self.client_id,
            'params': {}
        }
        for name, param in self.local_model.named_parameters():
            if 'lora_A' in name or 'lora_B' in name or 'lora_route' in name:
                lora_params['params'][name] = param.data.clone()
        return lora_params
    
    def load_params(self, params_or_path):
        """Load parameters into the model.

        Args:
            params_or_path: Either a dict of parameters, or a path to a .pt file
                            saved by get_lora_params() or a global checkpoint.
        """
        if isinstance(params_or_path, str):
            # Load from a .pt file saved by torch.save
            checkpoint = torch.load(params_or_path, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            if isinstance(checkpoint, dict):
                if 'params' in checkpoint:
                    # Format: {'client_id': ..., 'params': {...}}  — from get_lora_params()
                    params_to_load = checkpoint['params']
                elif all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
                    # Format: {name: tensor, ...} — raw state dict
                    params_to_load = checkpoint
                else:
                    raise ValueError(f"Unknown dict format in checkpoint: keys={list(checkpoint.keys())}")
            elif isinstance(checkpoint, list):
                # Format: [{name: tensor, ...}, ...] — aggregated_params list
                # For global checkpoints; this should be used per-client index
                raise ValueError(
                    "Detected a list (aggregated global checkpoint). "
                    "Use load_params(checkpoint[client_index]) instead."
                )
            else:
                raise ValueError(f"Unexpected checkpoint type: {type(checkpoint)}")
        else:
            params_to_load = params_or_path.get('params', params_or_path)

        self.local_model.load_state_dict(params_to_load, strict=False)
        print(f"  Loaded {len(params_to_load)} parameter tensors into client {self.client_id}")
    
    def local_training(self, model, global_state=None, lr=2e-4, epochs=1, batch_size=1, gradient_accumulation_steps=1,feddcr_config=None,
                       global_prototype=None, local_prototype=None):
        """Perform local training on client data."""
        if global_state:
            set_lora_state_dict(model, global_state)
        model.train()
        model.zero_grad(set_to_none=True)
        
        set_trainable_lora_parameters(model, self.lora_n, train_global=bool(feddcr_config))

        trainable_params = [p for p in model.parameters() if p.requires_grad]
        print(f"Number of trainable parameters: {len(trainable_params)}")
        if len(trainable_params) == 0:
            raise ValueError("No trainable parameters found!")
        
        training_args = TrainingArguments(
            output_dir=f"{self.cache_path}/{self.rank}_{self.lora_n}_{self.asymmetric}/client_{self.client_id}_checkpoints",
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=0,
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

        trainer_class = FedDCRTrainer if feddcr_config else Trainer

        trainer = trainer_class(
            model=model,
            args=training_args,
            train_dataset=self.client_dataset,
            tokenizer=self.tokenizer,
            data_collator=DataCollatorForSeq2Seq(
                self.tokenizer,
                pad_to_multiple_of=8,
                return_tensors="pt",
                padding=True,
            )
        )

        if feddcr_config:
            trainer.configure_feddcr(
                global_prototype, local_prototype,
                feddcr_config["sketch_dim"], feddcr_config["temperature"],
                feddcr_config["residual_penalty"], feddcr_config["ema"],
                feddcr_config["learnability_weight"], feddcr_config["stability_weight"],
                feddcr_config["conflict_variance_weight"], feddcr_config["score_history_size"],
            )
        
        trainer.train()

        client_stat = get_lora_state_dict(model)
        nonfinite = [name for name, value in client_stat.items() if not torch.isfinite(value).all()]
        if nonfinite:
            raise FloatingPointError(
                "FedDCR produced non-finite LoRA parameters; refusing to upload: "
                + ", ".join(nonfinite[:5])
            )
        route_metrics = trainer.feddcr_metrics() if feddcr_config else {}

        # Trainer retains references to optimizer and gradients.  Release them
        # before the next client reuses this worker's resident model.
        del trainer
        del training_args

        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return client_stat,route_metrics


    def evaluate_model(self, test_file, output_file, batch_size=2, temperature=0.1, 
                      top_p=0.75, top_k=40, num_beams=4, max_new_tokens=80):
        """Evaluate model on test data."""
        device = next(self.local_model.parameters()).device
        self.local_model.eval()
        
        def evaluate_batch(input_ids=None, instruction=None):
            if input_ids is not None:
                input_ids = input_ids.to(device)
            else:
                prompt = self.prompter.generate_prompt(instruction, None)
                inputs = self.tokenizer(prompt, return_tensors="pt")
                input_ids = inputs["input_ids"].to(device)

            generation_config = GenerationConfig(
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                num_beams=num_beams,
            )

            with torch.no_grad():
                generation_output = self.local_model.generate(
                    input_ids=input_ids,
                    generation_config=generation_config,
                    return_dict_in_generate=True,
                    output_scores=True,
                    max_new_tokens=max_new_tokens,
                )

            if len(generation_output.sequences) == 1:
                s = generation_output.sequences[0]
                output = self.tokenizer.decode(s)
                answers = self.prompter.get_response(output)
            else:
                s = generation_output.sequences.cpu()
                output = self.tokenizer.batch_decode(s)
                answers = [self.prompter.get_response(t).split('</s>')[0] for t in output]
            
            return answers

        eval_dataset = EvalDataset(test_file, self.prompter, self.tokenizer)
        dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)
        
        all_results = []
        for prompts, text in tqdm(dataloader, desc="Evaluating", leave=False):
            inputs = self.tokenizer(prompts, return_tensors="pt", padding=True)
            input_ids = inputs["input_ids"]
            results = evaluate_batch(input_ids=input_ids)
            all_results.extend(results)

        lines = open(test_file).readlines()
        for i, line in enumerate(lines):
            data = json.loads(line.strip())
            result = {
                'text': data['instruction'],
                'answer': all_results[i],
                'category': data.get('category', ''),
                'client_id': self.client_id
            }
            write_file(json.dumps(result, ensure_ascii=False), output_file)

        print(f'The output file is stored at {output_file}')
    
    def calculate_rouge_scores(self, reference_file, prediction_file, score_save_path):
        """Calculate ROUGE scores between predictions and references."""
        def read_list(file, key):
            dic = {}
            lines = open(file).readlines()
            for line in lines:
                line = line.strip()
                data = json.loads(line)
                if data['category'] not in dic:
                    dic[data['category']] = []
                tmpd = data[key]
                if tmpd.endswith('</s>'):
                    tmpd = tmpd.split('</s>')[0]
                dic[data['category']].append(tmpd)
            return dic

        def compute_rouge(predictions, references):
            from rouge_score import rouge_scorer
            scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
            scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
            for pred, ref in zip(predictions, references):
                result = scorer.score(ref, pred)
                for key in scores:
                    scores[key].append(getattr(result[key], 'fmeasure'))
            return {k: float(sum(v) / len(v)) if v else 0.0 for k, v in scores.items()}

        def get_result(targets, predictions, save_path):
            results = {}
            total_target = []
            total_pred = []
            
            for category in targets.keys():
                result = compute_rouge(predictions[category], targets[category])
                results[category] = result
                total_target.extend(targets[category])
                total_pred.extend(predictions[category])
                
            results['total'] = compute_rouge(total_pred, total_target)
            print(f"\nROUGE Scores: {results}")
            
            with open(save_path, 'w') as f:
                json.dump(results, f, indent=2)
            
            return results

        targets = read_list(reference_file, 'output')
        predictions = read_list(prediction_file, 'answer')
        
        return get_result(targets, predictions, score_save_path)

    def evaluate_and_score(self, test_file, output_file, score_file):
        """Evaluate model and calculate scores."""
        self.evaluate_model(test_file, output_file)
        scores = self.calculate_rouge_scores(test_file, output_file, score_file)
        return scores
