from __future__ import annotations

import math
import subprocess
from pathlib import Path
from typing import Any, Callable

import av
import cv2
import numpy as np

from .engine_runtime import try_interpolate_rife_trt
from .models import interpolate_rife
from .video_encoder import normalize_video_encoder, video_encoder_args, video_encoder_failure_code

Progress = Callable[[str, float, dict[str, Any] | None], None]


def _hard_cut(a: np.ndarray, b: np.ndarray) -> bool:
    aa = cv2.resize(cv2.cvtColor(a, cv2.COLOR_BGR2GRAY), (64, 36), interpolation=cv2.INTER_AREA)
    bb = cv2.resize(cv2.cvtColor(b, cv2.COLOR_BGR2GRAY), (64, 36), interpolation=cv2.INTER_AREA)
    mad = float(np.mean(np.abs(aa.astype(np.float32) - bb.astype(np.float32)))) / 255.0
    return mad >= 0.32


def _encoder(path: Path, width: int, height: int, fps: float, settings: dict[str, Any]) -> subprocess.Popen:
    return subprocess.Popen([
        "ffmpeg", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}",
        "-r", f"{fps:.8f}", "-i", "pipe:0", "-an", *video_encoder_args(settings),
        "-movflags", "+faststart", "-y", str(path)
    ], stdin=subprocess.PIPE)


def interpolate_file(
    source: Path,
    output: Path,
    *,
    target_fps: int,
    playback_speed: float = 1.0,
    timing_baked: bool = False,
    interpolation_model: str = "rife-4.9",
    cq: int = 17,
    cancel_event=None,
    progress: Progress | None = None,
    settings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if target_fps not in {30, 48, 60}:
        raise ValueError("target_fps must be 30, 48, or 60")
    if playback_speed <= 0: raise ValueError("playback_speed must be positive")
    model = interpolation_model.lower()
    if model == "gimm-vfi-f": raise RuntimeError("GIMM_LICENSE_NOT_CLEARED")
    if model not in {"rife", "rife-4.9", "none"}: raise ValueError(f"Unsupported VFI model: {model}")

    encoder_settings = dict(settings or {})
    encoder_settings.setdefault("nvencCq", cq)
    video_encoder = normalize_video_encoder(encoder_settings)

    inp = av.open(str(source)); stream = inp.streams.video[0]
    nominal_fps = float(stream.average_rate or 24.0)
    frames = iter(inp.decode(stream))
    try: first = next(frames)
    except StopIteration: raise RuntimeError("FFMPEG_DECODE_FAILED:no frames")
    first_arr = first.to_ndarray(format="bgr24")
    height, width = first_arr.shape[:2]
    enc = _encoder(output, width, height, float(target_fps), encoder_settings); assert enc.stdin is not None
    effective_speed = playback_speed if timing_baked else 1.0
    neural = model != "none" and target_fps > nominal_fps * effective_speed + 1e-6
    prev = first_arr
    prev_t = float(first.pts * stream.time_base) if first.pts is not None else 0.0
    next_out = 0.0; emitted = 0; decoded = 1

    def emit(frame: np.ndarray) -> None:
        nonlocal emitted
        if cancel_event is not None and cancel_event.is_set(): raise RuntimeError("CANCELLED")
        enc.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes()); emitted += 1

    last_interval = 1.0 / nominal_fps
    for frame in frames:
        current = frame.to_ndarray(format="bgr24")
        current_t = float(frame.pts * stream.time_base) if frame.pts is not None else prev_t + last_interval
        if current_t <= prev_t: current_t = prev_t + last_interval
        last_interval = current_t - prev_t
        cut = _hard_cut(prev, current)
        while next_out * effective_speed <= current_t + 1e-9:
            src_t = next_out * effective_speed
            if src_t < prev_t - 1e-9:
                next_out += 1.0 / target_fps; continue
            alpha = max(0.0, min(1.0, (src_t - prev_t) / (current_t - prev_t)))
            if alpha <= 1e-6: out = prev
            elif alpha >= 1 - 1e-6: out = current
            elif cut or not neural: out = prev if alpha < 0.5 else current
            else:
                prev_rgb = cv2.cvtColor(prev, cv2.COLOR_BGR2RGB)
                current_rgb = cv2.cvtColor(current, cv2.COLOR_BGR2RGB)
                try:
                    rgb = try_interpolate_rife_trt(prev_rgb, current_rgb, alpha, settings or {})
                except Exception as error:
                    if settings and not settings.get("allowNativeFallback", True):
                        raise
                    print(f"[enhancer] TensorRT RIFE fallback to native PyTorch: {error}", flush=True)
                    rgb = None
                if rgb is None:
                    rgb = interpolate_rife(prev_rgb, current_rgb, alpha)
                out = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            emit(out); next_out += 1.0 / target_fps
        decoded += 1; prev = current; prev_t = current_t
        if progress: progress("interpolating" if neural else "resampling", min(92.0, 40 + decoded / max(2, decoded + 8) * 50), {"decoded": decoded, "emitted": emitted, "videoEncoder": video_encoder})

    source_end = prev_t + max(last_interval, 1.0 / nominal_fps)
    output_end = source_end / effective_speed
    while next_out < output_end - 1e-9:
        emit(prev); next_out += 1.0 / target_fps
    inp.close(); enc.stdin.close()
    if enc.wait(timeout=600) != 0: raise RuntimeError(video_encoder_failure_code(encoder_settings))
    return {"frames": emitted, "targetFps": target_fps, "timingBaked": timing_baked, "sourceSpeed": playback_speed, "neuralVfi": neural, "videoEncoder": video_encoder}
