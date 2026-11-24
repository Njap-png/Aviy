"""Resource monitoring, quotas, bounded workload scheduling and reports."""

import math
import os
import shutil
from collections import deque
from dataclasses import dataclass
from importlib import import_module


@dataclass(frozen=True)
class ResourceSnapshot:
    cpu_percent: float | None
    gpu_memory_bytes: int | None
    ram_used_bytes: int | None
    storage_free_bytes: int

    def __post_init__(self):
        for name in (
            "cpu_percent",
            "gpu_memory_bytes",
            "ram_used_bytes",
            "storage_free_bytes",
        ):
            value = getattr(self, name)
            if value is not None and (not math.isfinite(float(value)) or value < 0):
                raise ValueError(f"Invalid resource value: {name}")
        if self.cpu_percent is not None and self.cpu_percent > 100:
            raise ValueError("CPU percentage cannot exceed 100.")


@dataclass(frozen=True)
class ResourceQuota:
    max_cpu_percent: float = 100.0
    max_gpu_memory_bytes: int | None = None
    max_ram_bytes: int | None = None
    min_storage_free_bytes: int = 0
    max_queued_jobs: int = 10

    def __post_init__(self):
        if not 0 <= self.max_cpu_percent <= 100:
            raise ValueError("Invalid CPU quota.")
        for name in ("max_gpu_memory_bytes", "max_ram_bytes", "min_storage_free_bytes"):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise ValueError(f"Invalid quota: {name}")
        if self.max_queued_jobs < 0:
            raise ValueError("Queue limit cannot be negative.")


class ResourceMonitor:
    def __init__(self, provider=None):
        self.provider = provider or self._system_snapshot

    @staticmethod
    def _system_snapshot():
        cpu = ram = gpu = None
        try:
            psutil = import_module("psutil")
            cpu_percent = psutil.cpu_percent
            virtual_memory = psutil.virtual_memory

            cpu = float(cpu_percent(interval=None))
            ram = int(virtual_memory().used)
        except ImportError:
            pass
        try:
            import torch

            if torch.cuda.is_available():
                gpu = int(torch.cuda.memory_allocated())
        except (ImportError, RuntimeError):
            pass
        return ResourceSnapshot(cpu, gpu, ram, int(shutil.disk_usage(os.getcwd()).free))

    def sample(self):
        snapshot = self.provider()
        if not isinstance(snapshot, ResourceSnapshot):
            raise TypeError("Resource provider must return ResourceSnapshot.")
        return snapshot


class WorkloadScheduler:
    def __init__(self, quota, monitor):
        if not isinstance(quota, ResourceQuota) or not isinstance(
            monitor, ResourceMonitor
        ):
            raise TypeError("ResourceQuota and ResourceMonitor are required.")
        self.quota = quota
        self.monitor = monitor
        self.queue = deque()
        self.history = []

    def _admission_errors(self, snapshot):
        errors = []
        q = self.quota
        if (
            snapshot.cpu_percent is not None
            and snapshot.cpu_percent > q.max_cpu_percent
        ):
            errors.append("cpu_quota")
        if (
            q.max_gpu_memory_bytes is not None
            and snapshot.gpu_memory_bytes is not None
            and snapshot.gpu_memory_bytes > q.max_gpu_memory_bytes
        ):
            errors.append("gpu_memory_quota")
        if (
            q.max_ram_bytes is not None
            and snapshot.ram_used_bytes is not None
            and snapshot.ram_used_bytes > q.max_ram_bytes
        ):
            errors.append("ram_quota")
        if snapshot.storage_free_bytes < q.min_storage_free_bytes:
            errors.append("storage_quota")
        if len(self.queue) >= q.max_queued_jobs:
            errors.append("queue_quota")
        return errors

    def submit(self, job_id, task):
        if not isinstance(job_id, str) or not job_id or not callable(task):
            raise ValueError("A job ID and callable task are required.")
        if any(item["job_id"] == job_id for item in self.queue):
            raise ValueError("Duplicate queued job ID.")
        snapshot = self.monitor.sample()
        errors = self._admission_errors(snapshot)
        if errors:
            self.history.append(
                {"job_id": job_id, "status": "rejected", "reasons": errors}
            )
            return {"accepted": False, "reasons": errors}
        self.queue.append({"job_id": job_id, "task": task})
        return {"accepted": True, "queue_size": len(self.queue)}

    def run_next(self):
        if not self.queue:
            return None
        item = self.queue.popleft()
        try:
            result = item["task"]()
            record = {"job_id": item["job_id"], "status": "completed", "result": result}
        except Exception as exc:  # noqa: BLE001 - queued tasks must be recorded even for unexpected failures
            record = {
                "job_id": item["job_id"],
                "status": "failed",
                "error": type(exc).__name__,
            }
        self.history.append(record)
        return record

    def report(self):
        counts = {}
        for item in self.history:
            counts[item["status"]] = counts.get(item["status"], 0) + 1
        return {
            "queued": len(self.queue),
            "completed": counts.get("completed", 0),
            "failed": counts.get("failed", 0),
            "rejected": counts.get("rejected", 0),
            "history": [dict(item) for item in self.history],
        }
