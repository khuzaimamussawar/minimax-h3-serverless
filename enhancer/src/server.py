from __future__ import annotations

import os
import subprocess
import threading
import time
import traceback
from dataclasses import dataclass, field
from typing import Any

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse

from .callbacks import post_event
from .config import RuntimeConfig
from .engine_builder import run_engine_build
from .fast_pipeline import run_fast_video, run_image_upscale, run_image_upscale_batch
from .gpu import qualify_gpu, telemetry
from .quality_pipeline import run_video_upscale as run_quality_video
from .video_encoder import normalize_video_encoder

app = FastAPI(title="SceneBuilder Enhancer Pod", version="2.0")

_CONFIG: RuntimeConfig | None = None
_READY = False
_QUALIFICATION: dict[str, Any] = {}
_STARTUP_ERROR = ""
_JOBS: dict[str, "JobRecord"] = {}
_LOCK = threading.RLock()
_CURRENT_JOB_ID: str | None = None
_IDLE_SINCE: float | None = None
_IDLE_TIMEOUT_SENT = False
_DRAINING = False


def config() -> RuntimeConfig:
    global _CONFIG
    if _CONFIG is None:
        _CONFIG = RuntimeConfig.from_env()
    return _CONFIG


@dataclass
class JobRecord:
    id: str
    payload: dict[str, Any]
    status: str = "queued"
    stage: str = "queued"
    progress: float = 0.0
    result: dict[str, Any] | None = None
    error_code: str | None = None
    error: str = ""
    debug: list[str] = field(default_factory=list)
    created_at: int = field(default_factory=lambda: int(time.time() * 1000))
    started_at: int | None = None
    completed_at: int | None = None
    updated_at: int = field(default_factory=lambda: int(time.time() * 1000))
    cancel_event: threading.Event = field(default_factory=threading.Event, repr=False)
    last_callback_at: float = 0.0

    def public(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "status": self.status,
            "stage": self.stage,
            "progress": self.progress,
            "result": self.result,
            "errorCode": self.error_code,
            "error": self.error,
            "debug": self.debug[-40:],
            "createdAt": self.created_at,
            "startedAt": self.started_at,
            "completedAt": self.completed_at,
            "updatedAt": self.updated_at,
        }


def _event(event_type: str, **extra: Any) -> None:
    payload = {
        "eventType": event_type,
        "event": event_type,
        "workerId": config().worker_id,
        "serviceKind": config().service_kind,
        "timestamp": time.time(),
        **extra,
    }
    threading.Thread(target=_post_event_background, args=(event_type, payload), daemon=True, name=f"enhancer-callback-{event_type}").start()


def _post_event_background(event_type: str, payload: dict[str, Any]) -> None:
    try:
        post_event(config(), payload)
    except Exception as error:
        print(f"[enhancer callback] {event_type} failed: {error}", flush=True)


def _progress(record: JobRecord, stage: str, value: float, detail: dict[str, Any] | None = None) -> None:
    now = time.time()
    with _LOCK:
        record.stage = str(stage)
        record.progress = max(0.0, min(100.0, float(value)))
        record.updated_at = int(now * 1000)
    if now - record.last_callback_at >= 1.0 or value >= 100:
        record.last_callback_at = now
        _event("job_progress", jobId=record.id, status=record.status, stage=record.stage,
               progress=record.progress, detail=detail or {}, telemetry=telemetry(record.public()))


def _error_code(error: BaseException) -> str:
    text = str(error)
    known = [
        "CUDA_UNAVAILABLE", "CUDA_DRIVER_TOO_OLD", "CUDA_OOM",
        "GPU_CAPABILITY_MISMATCH", "GPU_VRAM_BELOW_POLICY", "TRT_ENGINE_NOT_FOUND",
        "TRT_DESERIALIZE_FAILED", "TRT_BUILD_FAILED", "MODEL_LOAD_FAILED", "RIFE_RUNTIME_FAILED",
        "GIMM_LICENSE_NOT_CLEARED", "FLASHVSR_SELF_TEST_FAILED", "FFMPEG_DECODE_FAILED",
        "NVENC_HEVC_ENCODER_MISSING", "NVENC_ENCODE_FAILED", "X265_ENCODER_MISSING", "X265_ENCODE_FAILED",
        "R2_INPUT_FAILED", "R2_OUTPUT_FAILED", "CANCELLED",
    ]
    for code in known:
        if code in text:
            return code
    if "out of memory" in text.lower():
        return "CUDA_OOM"
    return "UNKNOWN"


def _run_job(record: JobRecord) -> None:
    global _CURRENT_JOB_ID, _IDLE_SINCE, _IDLE_TIMEOUT_SENT, _DRAINING
    with _LOCK:
        record.status = "processing"; record.stage = "starting"; record.started_at = int(time.time() * 1000)
        record.updated_at = record.started_at; _CURRENT_JOB_ID = record.id; _IDLE_SINCE = None; _IDLE_TIMEOUT_SENT = False; _DRAINING = False
    _event("job_started", jobId=record.id, status=record.status, stage=record.stage)
    try:
        job_type = str(record.payload.get("jobType") or record.payload.get("job_type") or "").strip()
        progress = lambda stage, value, detail=None: _progress(record, stage, value, detail)
        if job_type == "image_upscale":
            if config().service_kind != "enhancer_fast":
                raise RuntimeError("MODEL_LOAD_FAILED:image_upscale requires FAST runtime")
            result = run_image_upscale(record.payload, record.cancel_event, progress)
        elif job_type == "image_upscale_batch":
            if config().service_kind != "enhancer_fast":
                raise RuntimeError("MODEL_LOAD_FAILED:image_upscale_batch requires FAST runtime")
            result = run_image_upscale_batch(record.payload, record.cancel_event, progress)
        elif job_type == "video_upscale":
            settings = record.payload.get("settings") or {}
            if normalize_video_encoder(settings) == "nvenc":
                # Encoder choice comes from the job/Admin UI. Only NVENC jobs
                # pay the real hardware smoke check; x265 jobs must not be
                # rejected merely because a GPU has no working NVENC block.
                qualify_gpu(require_nvenc=True)
            result = run_quality_video(record.payload, record.cancel_event, progress) if config().service_kind == "enhancer_quality" else run_fast_video(record.payload, record.cancel_event, progress)
        elif job_type == "engine_build":
            if config().service_kind != "enhancer_engine_builder":
                raise RuntimeError("TRT_BUILD_FAILED:engine_build requires enhancer_engine_builder runtime")
            result = run_engine_build(record.payload, record.cancel_event, progress)
        else:
            raise ValueError(f"Unsupported jobType: {job_type}")
        with _LOCK:
            record.result = result; record.status = "completed"; record.stage = "completed"; record.progress = 100.0
            record.completed_at = int(time.time() * 1000); record.updated_at = record.completed_at
        _event("job_completed", jobId=record.id, status="completed", stage="completed", progress=100, output=result,
               telemetry=telemetry(record.public()))
    except Exception as error:
        code = _error_code(error)
        cancelled = record.cancel_event.is_set() or code == "CANCELLED"
        with _LOCK:
            record.status = "cancelled" if cancelled else "failed"
            record.stage = record.status
            record.error_code = "CANCELLED" if cancelled else code
            record.error = str(error)[:4000]
            record.debug = traceback.format_exc().splitlines()[-80:]
            record.completed_at = int(time.time() * 1000); record.updated_at = record.completed_at
        _event("job_cancelled" if cancelled else "job_failed", jobId=record.id, status=record.status, stage=record.stage,
               errorCode=record.error_code, error=record.error, debug=record.debug[-40:], telemetry=telemetry(record.public()))
    finally:
        with _LOCK:
            _CURRENT_JOB_ID = None; _IDLE_SINCE = time.time(); _IDLE_TIMEOUT_SENT = False; _DRAINING = False
            idle_since = _IDLE_SINCE
            timeout = config().idle_timeout_seconds
        _event("worker_idle", idleSince=idle_since, idleTimeoutSeconds=timeout,
               terminateAfter=(idle_since + timeout) if timeout > 0 else None)


def _require_auth(authorization: str | None) -> None:
    expected = config().pod_token
    raw = str(authorization or "")
    supplied = raw[7:] if raw.lower().startswith("bearer ") else raw
    import hmac
    if not supplied or not hmac.compare_digest(supplied, expected):
        raise HTTPException(status_code=401, detail="Invalid pod token")


def _boot() -> None:
    global _READY, _QUALIFICATION, _STARTUP_ERROR, _IDLE_SINCE
    try:
        _QUALIFICATION = qualify_gpu(require_nvenc=False)
        if config().service_kind == "enhancer_fast":
            from .models import esrgan
            esrgan("RealESRGAN_x4plus_anime_6B")
            esrgan("RealESRGAN_x4plus")
            esrgan("realesr-animevideov3")
            esrgan("realesr-general-x4v3")
        elif config().service_kind == "enhancer_quality":
            import torch
            if not torch.cuda.is_bf16_supported():
                raise RuntimeError("GPU_CAPABILITY_MISMATCH:BF16 required for QUALITY")
        elif config().service_kind == "enhancer_engine_builder":
            import tensorrt as trt
            if trt.Builder(trt.Logger(trt.Logger.WARNING)) is None:
                raise RuntimeError("TRT_BUILD_FAILED:TensorRT builder unavailable")
        _READY = True
        _IDLE_SINCE = None
        _event("worker_ready", readyAt=time.time(), qualification=_QUALIFICATION, capabilities=_capabilities())
    except Exception as error:
        _STARTUP_ERROR = str(error)
        _READY = False
        print(f"[enhancer boot] qualification failed: {_STARTUP_ERROR}", flush=True)
        try: _event("worker_unhealthy", errorCode=_error_code(error), error=_STARTUP_ERROR)
        except Exception: pass


def _idle_monitor() -> None:
    global _IDLE_TIMEOUT_SENT, _DRAINING
    while True:
        time.sleep(2)
        with _LOCK:
            idle_since = _IDLE_SINCE; busy = _CURRENT_JOB_ID is not None; already = _IDLE_TIMEOUT_SENT
        if busy or idle_since is None or already or config().idle_timeout_seconds <= 0:
            continue
        if time.time() - idle_since >= config().idle_timeout_seconds:
            timeout = config().idle_timeout_seconds
            _event("idle_expired", idleSince=idle_since, idleTimeoutSeconds=timeout,
                   terminateAfter=idle_since + timeout)
            with _LOCK:
                _IDLE_TIMEOUT_SENT = True
                _DRAINING = True


def _capabilities() -> dict[str, Any]:
    fast = config().service_kind == "enhancer_fast"
    builder = config().service_kind == "enhancer_engine_builder"
    return {
        "serviceKind": config().service_kind,
        "gpuOnly": True,
        "imageUpscale": fast,
        "videoUpscale": not builder,
        "engineBuild": builder,
        "models": (["realesr-animevideov3", "realesr-general-x4v3", "rife-4.9"] if builder else (["RealESRGAN_x4plus_anime_6B", "RealESRGAN_x4plus", "realesr-animevideov3", "realesr-general-x4v3", "rife-4.9"] if fast else ["flashvsr-v1.1", "rife-4.9"])),
        "gimmPackaged": True,
        "gimmProductionEnabled": False,
        "targetFps": [30, 48, 60],
        "videoEncoders": {
            "nvenc": {"codec": "hevc_nvenc", "qualityControl": "cq", "min": 12, "max": 25},
            "x265": {"codec": "libx265", "qualityControl": "crf", "min": 12, "max": 25},
        },
        "precision": {"esrgan": "fp16", "rife": "fp16", "gimm": "fp32", "flashvsr": "bf16"},
        "cuda": "13.0.2", "pytorch": "2.13.0-cu130", "tensorrt": "10.14.1.48",
    }


@app.on_event("startup")
def on_startup() -> None:
    config()
    threading.Thread(target=_boot, daemon=True, name="enhancer-boot").start()
    threading.Thread(target=_idle_monitor, daemon=True, name="enhancer-idle-monitor").start()


@app.get("/health")
def health() -> dict[str, Any]:
    return {"ok": True, "ready": _READY, "workerId": config().worker_id, "serviceKind": config().service_kind,
            "busy": _CURRENT_JOB_ID is not None, "draining": _DRAINING, "startupError": _STARTUP_ERROR or None}


@app.get("/ready")
def ready():
    if not _READY:
        return JSONResponse({"ready": False, "error": _STARTUP_ERROR or "qualifying"}, status_code=503)
    if _DRAINING:
        return JSONResponse({"ready": False, "status": "draining", "qualification": _QUALIFICATION}, status_code=503)
    return {"ready": True, "qualification": _QUALIFICATION}


@app.get("/capabilities")
def capabilities() -> dict[str, Any]:
    return _capabilities()


@app.get("/telemetry")
def get_telemetry() -> dict[str, Any]:
    with _LOCK: current = _JOBS.get(_CURRENT_JOB_ID) if _CURRENT_JOB_ID else None
    return telemetry(current.public() if current else None)


def _disk_summary() -> dict[str, Any]:
    try:
        completed = subprocess.run(["df", "-Pk", "/"], capture_output=True, text=True, timeout=5, check=False)
        return {"ok": completed.returncode == 0, "stdout": completed.stdout[-2000:], "stderr": completed.stderr[-1000:]}
    except Exception as error:
        return {"ok": False, "error": f"{type(error).__name__}: {error}"}


def _nvidia_smi_summary() -> dict[str, Any]:
    command = [
        "nvidia-smi",
        "--query-gpu=name,uuid,driver_version,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu,power.draw",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=10, check=True)
        rows = []
        for line in completed.stdout.splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) >= 9:
                rows.append({
                    "name": parts[0], "uuid": parts[1], "driverVersion": parts[2],
                    "memoryTotalMiB": parts[3], "memoryUsedMiB": parts[4], "memoryFreeMiB": parts[5],
                    "utilizationGpuPercent": parts[6], "temperatureGpuC": parts[7], "powerDrawW": parts[8],
                })
        return {"ok": True, "gpus": rows}
    except Exception as error:
        return {"ok": False, "error": f"{type(error).__name__}: {error}", "gpus": []}


def _diagnostics() -> dict[str, Any]:
    with _LOCK:
        current = _JOBS.get(_CURRENT_JOB_ID) if _CURRENT_JOB_ID else None
        jobs = [job.public() for job in _JOBS.values()]
        worker = {
            "workerId": config().worker_id,
            "serviceKind": config().service_kind,
            "ready": _READY,
            "busy": _CURRENT_JOB_ID is not None,
            "currentJobId": _CURRENT_JOB_ID,
            "idleSince": int(_IDLE_SINCE * 1000) if _IDLE_SINCE else None,
            "idleTimeoutSent": _IDLE_TIMEOUT_SENT,
            "draining": _DRAINING,
            "startupError": _STARTUP_ERROR or None,
        }
    return {"ok": True, "worker": worker, "qualification": _QUALIFICATION, "capabilities": _capabilities(),
            "telemetry": telemetry(current.public() if current else None), "jobs": jobs[-20:],
            "disk": _disk_summary(), "nvidiaSmi": _nvidia_smi_summary()}


@app.get("/diagnostics")
def diagnostics(authorization: str | None = Header(default=None)) -> dict[str, Any]:
    _require_auth(authorization)
    return _diagnostics()


@app.get("/diagnostics/gpu")
def diagnostics_gpu(authorization: str | None = Header(default=None)) -> dict[str, Any]:
    _require_auth(authorization)
    return {"ok": True, "qualification": _QUALIFICATION, "telemetry": telemetry(None), "nvidiaSmi": _nvidia_smi_summary()}


@app.post("/jobs")
async def create_job(request: Request, authorization: str | None = Header(default=None)):
    _require_auth(authorization)
    if not _READY:
        raise HTTPException(status_code=503, detail=_STARTUP_ERROR or "Worker not ready")
    payload = await request.json()
    job_id = str(payload.get("id") or payload.get("jobId") or "").strip()
    if not job_id:
        raise HTTPException(status_code=400, detail="job id is required")
    with _LOCK:
        if _DRAINING:
            raise HTTPException(status_code=409, detail="Worker is draining after idle timeout")
        existing = _JOBS.get(job_id)
        if existing:
            return existing.public()
        if _CURRENT_JOB_ID:
            raise HTTPException(status_code=409, detail=f"Worker busy with {_CURRENT_JOB_ID}")
        record = JobRecord(id=job_id, payload=payload)
        _JOBS[job_id] = record
        thread = threading.Thread(target=_run_job, args=(record,), daemon=True, name=f"enhancer-job-{job_id[:12]}")
        thread.start()
    return record.public()


@app.get("/jobs/{job_id}")
def get_job(job_id: str, authorization: str | None = Header(default=None)):
    _require_auth(authorization)
    with _LOCK: record = _JOBS.get(job_id)
    if not record: raise HTTPException(status_code=404, detail="Job not found")
    return record.public()


@app.post("/jobs/{job_id}/cancel")
def cancel_job(job_id: str, authorization: str | None = Header(default=None)):
    _require_auth(authorization)
    with _LOCK: record = _JOBS.get(job_id)
    if not record: raise HTTPException(status_code=404, detail="Job not found")
    if record.status in {"completed", "failed", "cancelled"}: return record.public()
    record.cancel_event.set(); record.stage = "cancelling"; record.updated_at = int(time.time() * 1000)
    _event("job_cancelling", jobId=record.id, status=record.status, stage=record.stage)
    return record.public()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("H3_POD_PORT", "8000")))
