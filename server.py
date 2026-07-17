#!/usr/bin/env python
# coding: utf-8

import torch
from typing import List, Dict
from util import set_lora_state_dict,get_lora_state_dict,load_base_model,load_dataset,prepare_local_dataset
import queue
import threading
from typing import Dict, List, Optional, Tuple
from utils.prompter import Prompter
from client import Client
import os
import concurrent.futures
class GPUWorkerPool:
    """
    多 GPU 模型池。

    每张 GPU 独立加载自己的 8-bit 模型副本 —— 模型常驻其 GPU 不迁移。
    释放时仅清空缓存，模型留在 GPU 上。下次 acquire 直接复用。
    """

    def __init__(self, model_name,rank,lora_alpha,lora_n,num_gpus: int):
        self.num_gpus = num_gpus
        # 维护每个 GPU 上的模型及其空闲状态
        self._queue: queue.Queue = queue.Queue()

        for gpu_id in range(num_gpus):
            # 每张 GPU 上独立加载模型
            device = torch.device(f"cuda:{gpu_id}")
            model, _ = load_base_model(model_name, rank,lora_alpha,lora_n, device=device)
            model.eval()
            # 模型已加载在目标 GPU 上，不再移动
            self._queue.put((model, gpu_id))
            print(f"  [Pool] GPU {gpu_id} 模型已就绪 "
                  f"({torch.cuda.memory_allocated(gpu_id)/1e9:.1f}GB / "
                  f"{torch.cuda.get_device_properties(gpu_id).total_memory/1e9:.0f}GB)")

    def acquire(self) -> Tuple[torch.nn.Module, int]:
        """
        获取一个空闲 GPU 上的模型。
        返回的模型始终在它绑定的 GPU 上（常驻，不迁移）。
        """
        model, gpu_id = self._queue.get()
        return model, gpu_id

    def release(self, model: torch.nn.Module, gpu_id: int):
        """归还模型到池中。保留在 GPU 上，不清除模型"""
        torch.cuda.synchronize(gpu_id)
        self._queue.put((model, gpu_id))

class Server:
    """Federated Learning Server for parameter aggregation."""
    
    def __init__(self, args, device: str = "cuda"):
        self.device = device
        self.select_result = None
        self.args = args
        self.clients: Dict[int, FedClient] = {}
        self.prompter = Prompter("alpaca_short")


        # ── 多 GPU 工作池 ──
        num_gpus = min(self.args.num_gpus, torch.cuda.device_count())
        self.pool: Optional[GPUWorkerPool] = None
        if num_gpus > 1:
            print(f"\n[Multi-GPU] 初始化 {num_gpus} 个 GPU 工作副本...")
            self.pool = GPUWorkerPool(self.args.model_name,self.args.rank,self.args.lora_alpha,self.args.lora_n,num_gpus)

        # ── 共享模型 ──
        #   单 GPU: 直接加载在 cuda:0 上，常驻不迁移（8-bit 模型不可迁移）
        #   多 GPU: 加载在 CPU 上，仅用于 save_final_model；评估用 pool 的模型
        print("\n===== 初始化联邦服务器 =====")
        if self.pool is None:
            self.shared_model, self.tokenizer = load_base_model(self.args.model_name,self.args.rank,self.args.lora_alpha,self.args.lora_n, device=device)
            self.shared_model.eval()
        else:
            self.shared_model, self.tokenizer = load_base_model(self.args.model_name, self.args.rank,self.args.lora_alpha,self.args.lora_n, device=torch.device("cpu"))
            self.shared_model.eval()
        
        trainable = sum(p.numel() for p in self.shared_model.parameters() if p.requires_grad)
        print(f"可训练参数量: {trainable / 1e6:.2f}M")


    
    def train(self,clients_checkpoints):
        global_state = get_lora_state_dict(self.shared_model)

        for round_idx in range(self.args.rounds):
            print(f"\n--- FedAvg 轮次 {round_idx}  ---")

            if self.pool is not None:
                def _fedavg_train(cid, model, gpu_id):
                    set_lora_state_dict(model, global_state)
                    try:
                        uploaded= self.clients[cid].local_training(
                            model=model,
                            global_state=global_state,
                            lr=self.args.lr,
                            epochs=self.args.local_epochs,
                            batch_size=self.args.batch_size
                        )
                        return uploaded
                    except Exception as e:
                        import traceback
                        print(f"[FedAvg] 客户端 {cid} 在 GPU {gpu_id} 上训练失败: {e}")
                        traceback.print_exc()
                        raise
                client_states = self._run_clients_parallel(
                    sorted(self.clients.keys()), _fedavg_train
                )
            else:
                # 单 GPU (shared_model 已常驻 GPU)
                client_states, client_metrics = {}, {}
                for cid in sorted(self.clients.keys()):
                    set_lora_state_dict(self.shared_model, global_state)
                    uploaded= self.clients[cid].local_training(
                        model=model,
                        global_state=global_state,
                        lr=self.args.lr,
                        epochs=self.args.local_epochs,
                        batch_size=self.args.batch_size
                    )
                    client_states[cid] = uploaded
            
            
            
            for client_id in client_states.keys():
                client_save_path=os.path.join(clients_checkpoints,f"client_{client_id}.pt")
                torch.save(client_states[client_id],client_save_path)
            
            client_params=[client_states[cid] for cid in range(self.args.client_num)]
            if round_idx >= 0:
                if self.args.lora_n>1:
                    aggregated_params = self.aggregation(
                        route_aggregation=False,
                        params=client_params
                    )
                else:
                    aggregated_params = self.aggregation_wtt(
                        route_aggregation=False,
                        params=client_params
                    )
                    global_path=os.path.join(clients_checkpoints,"global.pt")
                    torch.save(aggregated_params,global_path)
        
    def setup_clients(self):
        print("\n===== 初始化客户端 =====")
        for client_id in range(self.args.client_num):
            local_data_path = os.path.join(self.args.data_path, "8/" f"local_training_{client_id}.json")
            client_data = load_dataset("json", data_files=local_data_path)
            local_data = prepare_local_dataset(client_data, self.tokenizer, self.prompter)
            client=Client(
                client_id,
                local_data,
                self.tokenizer,
                self.prompter,
                self.args.model_name,
                self.device,
                rank=self.args.rank,
                lora_n=self.args.lora_n,
                asymmetric=False,
                cache_path=self.args.result_dir
            )
            self.clients[client_id]=client

    def _run_clients_parallel(
        self,
        client_ids: List[int],
        train_fn,  # callable(cid, model, gpu_id) -> (state, metrics)
    ) -> Tuple[Dict[int, Dict], Dict[int, dict]]:
        """
        在多 GPU 上并行执行 train_fn。

        从池中获取空闲 GPU，客户端训练完后自动归还。
        客户端数 > GPU 数时自动排队（如 8 客户端 / 4 GPU = 2 波）。
        """
        num_gpus = self.args.num_gpus
        client_states: Dict[int, Dict] = {}
        lock = threading.Lock()

        def _worker(cid):
            # 备份原始 device，训练时临时切换到目标 GPU
            orig_device = self.clients[cid].device
            model, gpu_id = self.pool.acquire()
            device_c = torch.device(f"cuda:{gpu_id}")
            self.clients[cid].device = device_c
            try:
                state = train_fn(cid, model, gpu_id)
            except Exception as e:
                import traceback
                print(f"[Worker] 客户端 {cid} 在 GPU {gpu_id} 上训练失败: {e}")
                traceback.print_exc()
                raise
            finally:
                self.clients[cid].device = orig_device  # 恢复原始 device
                self.pool.release(model, gpu_id)

            with lock:
                client_states[cid] = state
               
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_gpus) as executor:
            futures = [executor.submit(_worker, cid) for cid in client_ids]
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    import traceback
                    print(f"[Parallel] 客户端训练异常: {e}")
                    traceback.print_exc()
                    raise

        return client_states

    def aggregation(self, route_aggregation: bool, params: List) -> List[Dict]:
        """
        Aggregate client parameters using FedALT strategy.
        
        Args:
            route_aggregation: Whether to aggregate routing parameters
            params: List of client parameter dictionaries
            
        Returns:
            List of aggregated parameters for each client
        """
        gpu_params = [
            {k: v.to(self.device) for k, v in client_params.items()}
            for client_params in params
        ]

        num_clients = len(gpu_params)
        aggregated_results = [{} for _ in range(num_clients)]
        param_names = gpu_params[0].keys()

        for client_idx in range(num_clients):
            for param_name in param_names:
                # Handle routing parameters
                if 'lora_route' in param_name:
                    if route_aggregation:
                        stacked_params = torch.stack([
                            gpu_params[i][param_name]
                            for i in range(num_clients)
                        ]).to(self.device)
                        aggregated_results[client_idx][param_name] = stacked_params.mean(dim=0)
                    continue

                # Aggregate local LoRA parameters from other clients (Rest-of-World)
                if 'lora_A1' in param_name or 'lora_B1' in param_name:
                    aggregated_name = param_name.replace('A1', 'A0').replace('B1', 'B0')
                    stacked_params = torch.stack([
                        gpu_params[i][param_name]
                        for i in range(num_clients) if i != client_idx 
                    ]).to(self.device)
                    aggregated_results[client_idx][aggregated_name] = stacked_params.mean(dim=0)
        
        return aggregated_results
    
    def aggregation_wtt(self, route_aggregation: bool, params: List) -> List[Dict]:
        gpu_params = [
            {k: v.to(self.device) for k, v in client_params.items()}
            for client_params in params
        ]

        num_clients = len(gpu_params)
        aggregated_results = [{} for _ in range(num_clients)]
        param_names = gpu_params[0].keys()

        for client_idx in range(num_clients):
            for param_name in param_names:
                # Handle routing parameters
                if 'lora_route' in param_name:
                    if route_aggregation:
                        stacked_params = torch.stack([
                            gpu_params[i][param_name]
                            for i in range(num_clients)
                        ]).to(self.device)
                        aggregated_results[client_idx][param_name] = stacked_params.mean(dim=0)
                    continue

                # Aggregate local LoRA parameters from other clients (Rest-of-World)
                if 'lora_A' in param_name:
                    # aggregated_name = param_name.replace('A1', 'A0').replace('B1', 'B0')
                    stacked_params = torch.stack([
                        gpu_params[i][param_name]
                        for i in range(num_clients) if i != client_idx
                    ]).to(self.device)
                    aggregated_results[client_idx][param_name] = stacked_params.mean(dim=0)

                if 'lora_B0' in param_name or 'lora_B' in param_name:
                    aggregated_results[client_idx][param_name] = gpu_params[client_idx][param_name]
                

        return aggregated_results

