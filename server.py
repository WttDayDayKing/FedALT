#!/usr/bin/env python
# coding: utf-8
"""Federated server and multi-client, multi-GPU scheduler."""

import os
import queue
from typing import Dict, Iterable, List

import torch
import torch.multiprocessing as mp

from gpu_worker import _cuda_visible_device, client_worker


class Server:
    """Coordinate federated rounds without placing a shared model on the server.

    The server only holds LoRA tensors on CPU.  A spawned worker owns one GPU
    and one quantized base model, so several clients can train concurrently on
    separate GPUs while clients assigned to the same GPU reuse that model.
    """

    def __init__(self, args):
        self.args = args
        self.device = torch.device("cpu")
        self.gpu_ids = self._resolve_gpu_ids()
        self.client_ids = list(range(args.client_num))
        self.client_states: Dict[int, Dict[str, torch.Tensor]] = {}
        self._validate_training_files()

        print(
            f"[Multi-GPU] {len(self.gpu_ids)} GPU worker(s): {self.gpu_ids}; "
            f"{len(self.client_ids)} client(s)",
            flush=True,
        )

    def _resolve_gpu_ids(self) -> List[int]:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; FedALT local training requires at least one CUDA GPU.")

        visible_count = torch.cuda.device_count()
        if self.args.gpu_ids:
            try:
                gpu_ids = [int(value.strip()) for value in self.args.gpu_ids.split(",") if value.strip()]
            except ValueError as error:
                raise ValueError("--gpu_ids must be a comma-separated list such as 0,1,2,3.") from error
            if not gpu_ids:
                raise ValueError("--gpu_ids did not contain any GPU index.")
        else:
            if self.args.num_gpus < 1:
                raise ValueError("--num_gpus must be at least 1.")
            gpu_ids = list(range(min(self.args.num_gpus, visible_count)))

        if len(set(gpu_ids)) != len(gpu_ids):
            raise ValueError("--gpu_ids must not contain duplicate GPU indices.")
        invalid = [gpu_id for gpu_id in gpu_ids if gpu_id < 0 or gpu_id >= visible_count]
        if invalid:
            raise ValueError(
                f"Requested GPU(s) {invalid}, but only visible GPU indices 0..{visible_count - 1} are available."
            )
        return gpu_ids

    def _validate_training_files(self) -> None:
        missing = [
            os.path.join(self.args.data_path, self.args.partition_dir, f"local_training_{cid}.json")
            for cid in self.client_ids
            if not os.path.isfile(os.path.join(self.args.data_path, self.args.partition_dir, f"local_training_{cid}.json"))
        ]
        if missing:
            raise FileNotFoundError("Missing client training data:\n" + "\n".join(missing))

    def train(self, clients_checkpoints: str) -> None:
        os.makedirs(clients_checkpoints, exist_ok=True)

        for round_idx in range(self.args.rounds):
            print(f"\n--- FedAvg round {round_idx + 1}/{self.args.rounds} ---", flush=True)
            uploaded_states = self._run_clients_parallel(self.client_ids, self.client_states)

            for client_id, state in uploaded_states.items():
                torch.save(state, os.path.join(clients_checkpoints, f"client_{client_id}.pt"))

            ordered_states = [uploaded_states[client_id] for client_id in self.client_ids]
            aggregated = (
                self.aggregation(False, ordered_states)
                if self.args.lora_n > 1
                else self.aggregation_wtt(False, ordered_states)
            )

            # FedALT aggregation only replaces the Rest-of-World adapter for
            # lora_n > 1.  Keep each client's local adapter and router state.
            self.client_states = {
                client_id: self._merge_states(uploaded_states[client_id], aggregated[index])
                for index, client_id in enumerate(self.client_ids)
            }
            torch.save(
                [self.client_states[client_id] for client_id in self.client_ids],
                os.path.join(clients_checkpoints, "global.pt"),
            )

    @staticmethod
    def _merge_states(base: Dict[str, torch.Tensor], updates: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        merged = {name: tensor.detach().cpu().clone() for name, tensor in base.items()}
        merged.update({name: tensor.detach().cpu().clone() for name, tensor in updates.items()})
        return merged

    def _run_clients_parallel(
        self,
        client_ids: Iterable[int],
        client_states: Dict[int, Dict[str, torch.Tensor]],
    ) -> Dict[int, Dict[str, torch.Tensor]]:
        """Run at most one training process per GPU and collect every result."""
        client_ids = list(client_ids)
        assignments = {
            gpu_id: client_ids[index::len(self.gpu_ids)]
            for index, gpu_id in enumerate(self.gpu_ids)
            if client_ids[index::len(self.gpu_ids)]
        }
        ctx = mp.get_context("spawn")
        result_queue = ctx.Queue()
        processes = []
        inherited_cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        worker_cuda_visible_devices = {
            gpu_id: _cuda_visible_device(gpu_id) for gpu_id in assignments
        }
        try:
            for gpu_id, assigned_clients in assignments.items():
                # ``spawn`` inherits this environment before importing
                # PyTorch. Each worker consequently sees exactly one GPU as
                # cuda:0, matching both bitsandbytes and Accelerate.
                os.environ["CUDA_VISIBLE_DEVICES"] = worker_cuda_visible_devices[gpu_id]
                process = ctx.Process(
                    target=client_worker,
                    args=(gpu_id, assigned_clients, self.args, client_states, result_queue),
                    name=f"fedalt-gpu-{gpu_id}",
                )
                process.start()
                processes.append(process)
        finally:
            if inherited_cuda_visible_devices is None:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = inherited_cuda_visible_devices

        results: Dict[int, Dict[str, torch.Tensor]] = {}
        errors: Dict[int, str] = {}
        received = 0
        try:
            while received < len(processes):
                try:
                    _, worker_results, worker_errors = result_queue.get(timeout=1)
                    received += 1
                    results.update(worker_results)
                    errors.update(worker_errors)
                except queue.Empty:
                    failed = [process for process in processes if process.exitcode not in (None, 0)]
                    if failed:
                        details = ", ".join(f"{process.name} (exit {process.exitcode})" for process in failed)
                        raise RuntimeError(f"GPU worker exited before reporting results: {details}")
        finally:
            for process in processes:
                process.join()
            result_queue.close()
            result_queue.join_thread()

        missing = set(client_ids) - set(results) - set(errors)
        if missing:
            errors.update({client_id: "worker returned no result" for client_id in sorted(missing)})
        if errors:
            detail = "\n\n".join(f"Client {client_id}:\n{error}" for client_id, error in sorted(errors.items()))
            raise RuntimeError(f"One or more clients failed during local training:\n{detail}")
        return {client_id: results[client_id] for client_id in client_ids}

    def aggregation(self, route_aggregation: bool, params: List[Dict[str, torch.Tensor]]) -> List[Dict[str, torch.Tensor]]:
        """Build each client's Rest-of-World state from all other clients."""
        if len(params) == 1:
            return [{}]
        aggregated_results = [{} for _ in params]
        for client_idx in range(len(params)):
            for param_name in params[0]:
                if "lora_route" in param_name:
                    if route_aggregation:
                        aggregated_results[client_idx][param_name] = torch.stack(
                            [client[param_name] for client in params]
                        ).mean(dim=0).cpu()
                elif "lora_A1" in param_name or "lora_B1" in param_name:
                    output_name = param_name.replace("A1", "A0").replace("B1", "B0")
                    aggregated_results[client_idx][output_name] = torch.stack(
                        [client[param_name] for index, client in enumerate(params) if index != client_idx]
                    ).mean(dim=0).cpu()
        return aggregated_results

    def aggregation_wtt(self, route_aggregation: bool, params: List[Dict[str, torch.Tensor]]) -> List[Dict[str, torch.Tensor]]:
        """Aggregate the shared A adapter while retaining the client's B adapter."""
        if len(params) == 1:
            return [{name: value.detach().cpu().clone() for name, value in params[0].items()}]
        aggregated_results = [{} for _ in params]
        for client_idx in range(len(params)):
            for param_name in params[0]:
                if "lora_route" in param_name:
                    if route_aggregation:
                        aggregated_results[client_idx][param_name] = torch.stack(
                            [client[param_name] for client in params]
                        ).mean(dim=0).cpu()
                elif "lora_A" in param_name:
                    aggregated_results[client_idx][param_name] = torch.stack(
                        [client[param_name] for index, client in enumerate(params) if index != client_idx]
                    ).mean(dim=0).cpu()
                elif "lora_B" in param_name:
                    aggregated_results[client_idx][param_name] = params[client_idx][param_name].detach().cpu().clone()
        return aggregated_results
