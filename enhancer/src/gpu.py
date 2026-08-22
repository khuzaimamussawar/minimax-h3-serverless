from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

EXPECTED_CUDA_PREFIX = "13.0"
EXPECTED_TORCH = "2.13.0"
EXPECTED_TRT = "10.14.1.48"
EXPECTED_TRT_PYTHON_VERSIONS = {EXPECTED_TRT, f"{EXPECTED_TRT}.post1"}


def _cmd(args: list[str], timeout: int = 20) -> str:
    try:
        result = subprocess.run(
            args,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as error:
        output = str(error.stdout or "").strip()
        command = " ".join(str(arg) for arg in args)
        detail = output[-4000:] if output else "no process output"
        raise RuntimeError(
            f"COMMAND_FAILED:{error.returncode}:{command}: {detail}"
        ) from error


def _nvidia_query() -> dict[str, Any]:
    query = _cmd([
        "nvidia-smi",
        "--query-gpu=name,driver_version,memory.total,compute_cap",
        "--format=csv,noheader,nounits",
    ])
    row = query.splitlines()[0]
    parts = [item.strip() for item in row.split(",")]
    if len(parts) < 4:
        raise RuntimeError(f"Unexpected nvidia-smi result: {row}")
    name, driver, memory_mb, compute_capability = parts[:4]
    detail = _cmd(["nvidia-smi", "-q"], timeout=20)
    lowered = f"{name}\n{detail}".lower()
    mig_enabled = re.search(r"mig mode\s*:\s*current\s*:\s*enabled", lowered) is not None
    partitioned = (
        " mig " in f" {name.lower()} "
        or "mig device" in lowered
        or mig_enabled
        or "high frequency" in lowered
        or "high-frequency" in lowered
        or "high_frequency" in lowered
    )
    return {
        "name": name,
        "driverVersion": driver,
        "vramMb": int(float(memory_mb)),
        "computeCapability": compute_capability,
        "partitioned": partitioned,
    }


def _nvenc_smoke() -> None:
    encoders = _cmd(["ffmpeg", "-hide_banner", "-encoders"], timeout=20)
    if "hevc_nvenc" not in encoders:
        raise RuntimeError("NVENC_HEVC_ENCODER_MISSING")
    with tempfile.TemporaryDirectory(prefix="sb-nvenc-") as tmp:
        output = Path(tmp) / "smoke.mp4"
        # Do not use a 128px-wide HEVC qualification frame. NVENC on newer
        # generations rejects that tiny HEVC width before the real Enhancer job
        # ever starts. Exercise the same HEVC Main10/p010 path as production with
        # a small but valid 256x256 frame instead.
        _cmd([
            "ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=c=black:s=256x256:r=1",
            "-frames:v", "1", "-c:v", "hevc_nvenc", "-profile:v", "main10",
            "-pix_fmt", "p010le", "-y", str(output),
        ], timeout=30)
        if not output.is_file() or output.stat().st_size <= 0:
            raise RuntimeError("NVENC_SMOKE_EMPTY")


def qualify_gpu(*, require_nvenc: bool = True) -> dict[str, Any]:
    import torch
    import cupy as cp
    import tensorrt as trt

    if torch.__version__.split("+")[0] != EXPECTED_TORCH:
        raise RuntimeError(f"PYTORCH_VERSION_MISMATCH:{torch.__version__}")
    if not str(torch.version.cuda or "").startswith(EXPECTED_CUDA_PREFIX):
        raise RuntimeError(f"PYTORCH_CUDA_MISMATCH:{torch.version.cuda}")
    if trt.__version__ not in EXPECTED_TRT_PYTHON_VERSIONS:
        raise RuntimeError(f"TENSORRT_VERSION_MISMATCH:{trt.__version__}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA_UNAVAILABLE")

    details = _nvidia_query()
    device = torch.device("cuda:0")
    a = torch.arange(32, device=device, dtype=torch.float32)
    torch_value = float((a * a).sum().item())
    if torch_value <= 0:
        raise RuntimeError("CUDA_TORCH_OP_FAILED")

    # Real CuPy device kernel, not merely import/version inspection.
    x = cp.arange(32, dtype=cp.float32)
    cp_value = float(cp.asnumpy((x * x).sum()))
    if cp_value <= 0:
        raise RuntimeError("CUDA_CUPY_OP_FAILED")

    if require_nvenc:
        _nvenc_smoke()

    props = torch.cuda.get_device_properties(0)
    details.update({
        "torchVersion": torch.__version__,
        "torchCudaVersion": torch.version.cuda,
        "cupyVersion": cp.__version__,
        "tensorrtVersion": trt.__version__,
        "torchDeviceName": props.name,
        "torchVramMb": int(props.total_memory // (1024 * 1024)),
        "qualifiedAt": int(time.time() * 1000),
        "gpuOnly": True,
        "nvenc": require_nvenc,
    })
    return details


def telemetry(current_job: dict[str, Any] | None = None) -> dict[str, Any]:
    import torch

    payload: dict[str, Any] = {
        "at": int(time.time() * 1000),
        "currentJob": current_job or None,
    }
    if not torch.cuda.is_available():
        payload["cudaAvailable"] = False
        return payload
    payload["cudaAvailable"] = True
    payload["allocatedMb"] = int(torch.cuda.memory_allocated(0) // (1024 * 1024))
    payload["reservedMb"] = int(torch.cuda.memory_reserved(0) // (1024 * 1024))
    payload["maxAllocatedMb"] = int(torch.cuda.max_memory_allocated(0) // (1024 * 1024))
    try:
        raw = _cmd([
            "nvidia-smi",
            "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
            "--format=csv,noheader,nounits",
        ], timeout=5)
        parts = [item.strip() for item in raw.splitlines()[0].split(",")]
        payload.update({
            "gpuUtilPercent": float(parts[0]),
            "memoryUsedMb": float(parts[1]),
            "memoryTotalMb": float(parts[2]),
            "temperatureC": float(parts[3]),
            "powerW": float(parts[4]),
        })
    except Exception as error:
        payload["nvidiaSmiError"] = str(error)
    return payload
