"""Federated server for contribution-weighted FedCKD aggregation."""

import io
import json
import math
import os
import queue
import time
from typing import Dict, Iterable, List

import torch
import torch.multiprocessing as mp

from fedckd import branch_of
from fedckd_worker import fedckd_client_worker
from gpu_worker import _cuda_visible_device


class FedCKDServer:
    def __init__(self, args):
        self.args = args
        self.gpu_ids = self._resolve_gpu_ids()
        self.client_ids = list(range(args.client_num))
        self.client_states: Dict[int, Dict[str, torch.Tensor]] = {}
        self.client_routing_statistics: Dict[int, Dict[str, float]] = {}
        self.global_reference = None
        self._validate_training_files()

    def _resolve_gpu_ids(self) -> List[int]:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; FedCKD requires at least one CUDA GPU.")
        visible_count = torch.cuda.device_count()
        if self.args.gpu_ids:
            gpu_ids = [int(x.strip()) for x in self.args.gpu_ids.split(",") if x.strip()]
        else:
            gpu_ids = list(range(min(self.args.num_gpus, visible_count)))
        if not gpu_ids or len(set(gpu_ids)) != len(gpu_ids):
            raise ValueError("GPU selection is empty or contains duplicates.")
        invalid = [gpu for gpu in gpu_ids if gpu < 0 or gpu >= visible_count]
        if invalid:
            raise ValueError(f"Invalid visible GPU indices: {invalid}")
        return gpu_ids

    def _validate_training_files(self):
        missing = [
            os.path.join(self.args.data_path, self.args.partition_dir, f"local_training_{cid}.json")
            for cid in self.client_ids
            if not os.path.isfile(os.path.join(
                self.args.data_path, self.args.partition_dir, f"local_training_{cid}.json"
            ))
        ]
        if missing:
            raise FileNotFoundError("Missing client training data:\n" + "\n".join(missing))

    @staticmethod
    def _merge_states(base, updates):
        merged = {name: tensor.detach().cpu().clone() for name, tensor in base.items()}
        merged.update({name: tensor.detach().cpu().clone() for name, tensor in updates.items()})
        return merged

    @staticmethod
    def aggregate_global(states, masses, aggregation: str = "global_mass"):
        """Aggregate LoRA-0 while retaining LoRA-1 as a private client state."""
        if aggregation == "global_mass":
            safe_masses = {
                cid: max(float(mass), 1e-8) if math.isfinite(float(mass)) else 1e-8
                for cid, mass in masses.items()
            }
            total_mass = sum(safe_masses.values())
            weights = {cid: safe_masses[cid] / total_mass for cid in safe_masses}
        elif aggregation == "uniform":
            weights = {cid: 1.0 / len(states) for cid in states}
        else:
            raise ValueError(f"Unsupported FedCKD aggregation: {aggregation}")
        shared = {}
        first_state = next(iter(states.values()))
        for name in first_state:
            if branch_of(name) != "global":
                continue
            shared[name] = sum(
                states[cid][name].detach().float().cpu() * weights[cid] for cid in states
            ).to(first_state[name].dtype)
        return shared, weights

    def train(self, checkpoints: str):
        os.makedirs(checkpoints, exist_ok=True)
        timings_path = os.path.join(checkpoints, "round_timings.jsonl")
        for round_idx in range(self.args.rounds):
            started = time.perf_counter()
            print(f"\n--- FedCKD round {round_idx + 1}/{self.args.rounds} ---", flush=True)
            payloads = self._run_clients_parallel(
                self.client_ids,
                self.client_states,
                self.client_routing_statistics,
                round_idx,
            )
            local_seconds = time.perf_counter() - started

            states = {cid: payload["state"] for cid, payload in payloads.items()}
            nonfinite_clients = {
                cid: [name for name, value in state.items() if not torch.isfinite(value).all()]
                for cid, state in states.items()
            }
            nonfinite_clients = {cid: names for cid, names in nonfinite_clients.items() if names}
            if nonfinite_clients:
                raise FloatingPointError(
                    "FedCKD rejected non-finite client state before aggregation: "
                    + "; ".join(f"client {cid} ({len(names)} tensors)" for cid, names in nonfinite_clients.items())
                )
            masses = {cid: payload["global_mass"] for cid, payload in payloads.items()}
            shared, weights = self.aggregate_global(states, masses, self.args.fedckd_aggregation)
            self.client_states = {
                cid: self._merge_states(states[cid], shared) for cid in self.client_ids
            }
            if self.args.fedckd_persist_routing_statistics:
                self.client_routing_statistics = {
                    cid: payloads[cid]["routing_statistics"] for cid in self.client_ids
                }
            else:
                self.client_routing_statistics = {}
            # The next round compares sample/global gradients with the direction
            # agreed upon by current client updates, without exposing raw data.
            sketches = [payloads[cid]["global_update_sketch"] for cid in self.client_ids]
            reference = torch.stack(sketches).mean(0)
            if torch.isfinite(reference).all() and reference.norm() > 1e-12:
                self.global_reference = -reference / reference.norm()
            else:
                self.global_reference = None

            for cid, state in states.items():
                torch.save(state, os.path.join(checkpoints, f"client_{cid}.pt"))
            torch.save(
                [self.client_states[cid] for cid in self.client_ids],
                os.path.join(checkpoints, "global.pt"),
            )
            torch.save(self.global_reference, os.path.join(checkpoints, "fedckd_global_reference.pt"))
            torch.save(
                self.client_routing_statistics,
                os.path.join(checkpoints, "fedckd_client_routing_statistics.pt"),
            )
            metrics = {cid: payloads[cid]["metrics"] for cid in self.client_ids}
            with open(os.path.join(checkpoints, f"fedckd_metrics_round_{round_idx + 1}.json"), "w") as handle:
                json.dump({"aggregation_weights": weights, "clients": metrics}, handle, indent=2)

            total_seconds = time.perf_counter() - started
            with open(timings_path, "a") as handle:
                handle.write(json.dumps({
                    "round": round_idx + 1,
                    "local_training_seconds": round(local_seconds, 3),
                    "total_seconds": round(total_seconds, 3),
                }) + "\n")
            print(f"[FedCKD] aggregation weights: {weights}", flush=True)

    def _run_clients_parallel(
        self,
        client_ids: Iterable[int],
        client_states,
        client_routing_statistics,
        round_idx: int,
    ):
        client_ids = list(client_ids)
        ctx = mp.get_context("spawn")
        task_queue, result_queue = ctx.Queue(), ctx.Queue()
        worker_gpu_ids = self.gpu_ids[: min(len(self.gpu_ids), len(client_ids))]
        for cid in client_ids:
            task_queue.put(cid)
        for _ in worker_gpu_ids:
            task_queue.put(None)

        inherited = os.environ.get("CUDA_VISIBLE_DEVICES")
        processes = []
        # Resolve every physical GPU before mutating the parent's environment.
        # Otherwise the first worker's one-entry CUDA_VISIBLE_DEVICES value is
        # reused for all following workers, placing them all on GPU 0.
        worker_visible_devices = {
            gpu_id: _cuda_visible_device(gpu_id) for gpu_id in worker_gpu_ids
        }
        try:
            for gpu_id in worker_gpu_ids:
                os.environ["CUDA_VISIBLE_DEVICES"] = worker_visible_devices[gpu_id]
                process = ctx.Process(
                    target=fedckd_client_worker,
                    args=(
                        gpu_id,
                        self.args,
                        client_states,
                        client_routing_statistics,
                        self.global_reference,
                        round_idx,
                        task_queue,
                        result_queue,
                    ),
                    name=f"fedckd-gpu-{gpu_id}",
                )
                process.start()
                processes.append(process)
        finally:
            if inherited is None:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = inherited

        results, errors, done_clients, done_workers = {}, {}, set(), set()
        try:
            while len(done_clients) < len(client_ids) or len(done_workers) < len(processes):
                try:
                    kind, gpu_id, client_id, payload, error = result_queue.get(timeout=1)
                    if kind == "client_result":
                        done_clients.add(client_id)
                        if error:
                            errors[client_id] = error
                        else:
                            results[client_id] = torch.load(io.BytesIO(payload), map_location="cpu")
                    elif kind == "worker_error":
                        errors[-(gpu_id + 1)] = error
                    elif kind == "worker_done":
                        done_workers.add(gpu_id)
                except queue.Empty:
                    failed = [p for p in processes if p.exitcode not in (None, 0)]
                    if failed:
                        raise RuntimeError("FedCKD GPU worker exited unexpectedly: " + ", ".join(p.name for p in failed))
        finally:
            for process in processes:
                process.join()
            task_queue.close(); result_queue.close()
        if errors:
            raise RuntimeError("FedCKD client failure:\n" + "\n\n".join(
                f"Client {cid}:\n{error}" for cid, error in sorted(errors.items())
            ))
        return {cid: results[cid] for cid in client_ids}
