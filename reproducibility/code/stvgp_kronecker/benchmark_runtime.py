"""Runtime and resource helpers shared by ERA5 benchmark entry points."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import socket
import subprocess
import time
from typing import Any, Callable


def peak_rss_mib() -> float:
    """Return process peak RSS in MiB on Linux and macOS."""

    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if platform.system() == "Darwin":
        value /= 1024.0
    return value / 1024.0


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: str | Path, payload: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    os.replace(temporary, destination)


def git_snapshot(root: str | Path) -> dict[str, Any]:
    root = Path(root)

    def run(*args: str) -> str | None:
        try:
            return subprocess.check_output(
                ["git", "-C", str(root), *args],
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=10,
            ).strip()
        except (OSError, subprocess.SubprocessError):
            return None

    status = run("status", "--porcelain")
    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "dirty": bool(status) if status is not None else None,
    }


def host_snapshot(root: str | Path | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "processor": platform.processor(),
        "pid": os.getpid(),
    }
    if root is not None:
        payload["git"] = git_snapshot(root)
    return payload


@dataclass(frozen=True)
class TorchRuntime:
    device: Any
    dtype: Any
    requested_device: str
    requested_dtype: str

    @property
    def uses_cuda(self) -> bool:
        return self.device.type == "cuda"

    def synchronize(self) -> None:
        if self.uses_cuda:
            import torch

            torch.cuda.synchronize(self.device)

    def reset_peak_memory(self) -> None:
        if self.uses_cuda:
            import torch

            self.synchronize()
            torch.cuda.reset_peak_memory_stats(self.device)

    def resources(self) -> dict[str, Any]:
        import torch

        payload: dict[str, Any] = {
            "device": str(self.device),
            "dtype": str(self.dtype).removeprefix("torch."),
            "peak_rss_mib": peak_rss_mib(),
            "torch": torch.__version__,
            "torch_cuda_build": torch.version.cuda,
            "cuda_available": bool(torch.cuda.is_available()),
        }
        if self.uses_cuda:
            properties = torch.cuda.get_device_properties(self.device)
            payload.update(
                {
                    "gpu_name": properties.name,
                    "gpu_compute_capability": list(
                        torch.cuda.get_device_capability(self.device)
                    ),
                    "gpu_total_memory_mib": properties.total_memory / 1024.0**2,
                    "peak_cuda_allocated_mib": torch.cuda.max_memory_allocated(
                        self.device
                    )
                    / 1024.0**2,
                    "peak_cuda_reserved_mib": torch.cuda.max_memory_reserved(
                        self.device
                    )
                    / 1024.0**2,
                }
            )
        return payload


def resolve_torch_runtime(device: str, dtype: str) -> TorchRuntime:
    import torch

    requested_device = str(device).lower()
    requested_dtype = str(dtype).lower()
    if requested_device == "auto":
        requested_device = "cuda" if torch.cuda.is_available() else "cpu"
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but torch.cuda.is_available() is false. "
            "Install a CUDA-enabled PyTorch wheel and check the NVIDIA driver."
        )
    if requested_device not in {"cpu", "cuda"} and not requested_device.startswith(
        "cuda:"
    ):
        raise ValueError(f"Unsupported torch device: {device}")
    dtype_map = {"float32": torch.float32, "float64": torch.float64}
    if requested_dtype not in dtype_map:
        raise ValueError(f"Unsupported torch dtype: {dtype}")
    return TorchRuntime(
        device=torch.device(requested_device),
        dtype=dtype_map[requested_dtype],
        requested_device=str(device),
        requested_dtype=requested_dtype,
    )


class SynchronizedTimer:
    """Wall-clock timer with backend synchronization at both boundaries."""

    def __init__(self, synchronize: Callable[[], None] | None = None) -> None:
        self._synchronize = synchronize or (lambda: None)
        self.elapsed = 0.0
        self._started = 0.0

    def __enter__(self) -> "SynchronizedTimer":
        self._synchronize()
        self._started = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self._synchronize()
        self.elapsed = time.perf_counter() - self._started


def configure_tensorflow(tf: Any, *, device: str, dtype: str) -> dict[str, Any]:
    """Select a TensorFlow device before model variables are created."""

    requested = str(device).lower()
    gpus = tf.config.list_physical_devices("GPU")
    if requested == "auto":
        requested = "cuda" if gpus else "cpu"
    if requested == "cpu":
        tf.config.set_visible_devices([], "GPU")
        selected = "cpu"
    elif requested == "cuda" or requested.startswith("cuda:"):
        if not gpus:
            raise RuntimeError(
                "CUDA was requested but TensorFlow did not discover a GPU. "
                "Check the TensorFlow/CUDA compatibility in this environment."
            )
        index = int(requested.split(":", 1)[1]) if ":" in requested else 0
        if index >= len(gpus):
            raise ValueError(f"TensorFlow GPU index {index} is unavailable")
        tf.config.set_visible_devices([gpus[index]], "GPU")
        try:
            tf.config.experimental.set_memory_growth(gpus[index], True)
        except RuntimeError:
            pass
        selected = f"cuda:{index}"
    else:
        raise ValueError(f"Unsupported TensorFlow device: {device}")
    if dtype not in {"float32", "float64"}:
        raise ValueError(f"Unsupported TensorFlow dtype: {dtype}")
    return {
        "device": selected,
        "dtype": dtype,
        "physical_gpus": [gpu.name for gpu in gpus],
    }


def tensorflow_memory(tf: Any, selected_device: str) -> dict[str, Any]:
    payload: dict[str, Any] = {"peak_rss_mib": peak_rss_mib()}
    if selected_device.startswith("cuda"):
        try:
            info = tf.config.experimental.get_memory_info("GPU:0")
            payload.update(
                {
                    "peak_cuda_allocated_mib": float(info["peak"]) / 1024.0**2,
                    "current_cuda_allocated_mib": float(info["current"])
                    / 1024.0**2,
                }
            )
        except (KeyError, RuntimeError, ValueError):
            payload["peak_cuda_allocated_mib"] = None
    return payload
