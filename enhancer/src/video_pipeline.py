from __future__ import annotations

import json
import math
import os
import subprocess
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import av
import cv2
import numpy as np

from .models import interpolate_rife, upscale_bgr
from .r2_store import upload_file
from .engine_runtime import try_interpolate_rife_trt, try_upscale_bgr_trt
from .video_encoder import normalize_video_encoder, video_encoder_args, video_encoder_failure_code

Progress = Callable[[str, float, dict[str, Any] | None], None]

TARGETS = {
    "1080p": (1920, 1080),
    "1440p": (2560, 1440),
    "2k": (2560, 1440),
    "2160p": (3840, 2160),
    "4k": (3840, 2160),
}


@dataclass
class Probe:
    width: int
    height: int
    fps: float
    duration: float
    has_audio: bool


def _download(url: str, path: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "SceneBuilder-Enhancer/2.0"})
    with urllib.request.urlopen(request, timeout=120) as response, path.open("wb") as output:
        while True:
            chunk = response.read(8 * 1024 * 1024)
            if not chunk:
                break
            output.write(chunk)


def _probe(path: Path) -> Probe:
    output = subprocess.check_output([
        "ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)
    ], text=True)
    data = json.loads(output)
    video = next(stream for stream in data.get("streams", []) if stream.get("codec_type") == "video")
    rate = str(video.get("avg_frame_rate") or video.get("r_frame_rate") or "0/1")
    num, den = rate.split("/", 1)
    fps = float(num) / float(den) if float(den) else 0.0
    duration = float(video.get("duration") or data.get("format", {}).get("duration") or 0.0)
    return Probe(
        width=int(video["width"]), height=int(video["height"]), fps=fps,
        duration=duration, has_audio=any(s.get("codec_type") == "audio" for s in data.get("streams", [])),
    )


def _target_dimensions(target: str, width: int, height: int) -> tuple[int, int]:
    dims = TARGETS.get(str(target).lower())
    if not dims:
        raise ValueError(f"Unsupported target resolution: {target}")
    return dims[::-1] if height > width else dims


def _center_crop_to_aspect(frame: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    h, w = frame.shape[:2]
    source_ar = w / h
    target_ar = target_w / target_h
    if abs(source_ar - target_ar) < 1e-5:
        return frame
    if source_ar > target_ar:
        new_w = max(1, round(h * target_ar))
        left = max(0, (w - new_w) // 2)
        return frame[:, left:left + new_w]
    new_h = max(1, round(w / target_ar))
    top = max(0, (h - new_h) // 2)
    return frame[top:top + new_h, :]


def _hard_cut(a: np.ndarray, b: np.ndarray) -> bool:
    # Conservative hard-cut detector used only to prevent cross-cut synthesis.
    # Compare small luma proxies so the detector cost is negligible next to SR.
    aa = cv2.resize(cv2.cvtColor(a, cv2.COLOR_BGR2GRAY), (64, 36), interpolation=cv2.INTER_AREA)
    bb = cv2.resize(cv2.cvtColor(b, cv2.COLOR_BGR2GRAY), (64, 36), interpolation=cv2.INTER_AREA)
    mad = float(np.mean(np.abs(aa.astype(np.float32) - bb.astype(np.float32)))) / 255.0
    hist_a = cv2.calcHist([aa], [0], None, [32], [0, 256]); cv2.normalize(hist_a, hist_a)
    hist_b = cv2.calcHist([bb], [0], None, [32], [0, 256]); cv2.normalize(hist_b, hist_b)
    corr = float(cv2.compareHist(hist_a, hist_b, cv2.HISTCMP_CORREL))
    return mad >= 0.32 and corr <= 0.55


def _spatial(frame_bgr: np.ndarray, model_name: str, target_w: int, target_h: int, settings: dict[str, Any]) -> np.ndarray:
    cropped = _center_crop_to_aspect(frame_bgr, target_w, target_h)
    h, w = cropped.shape[:2]
    scale = min(4.0, max(1.0, max(target_w / w, target_h / h)))
    if target_w <= w and target_h <= h:
        enhanced = cropped
    else:
        try:
            enhanced = try_upscale_bgr_trt(cropped, settings, target_w, target_h)
        except Exception as error:
            if not settings.get("allowNativeFallback", True):
                raise
            print(f"[enhancer] TensorRT spatial fallback to native PyTorch: {error}", flush=True)
            enhanced = None
        if enhanced is None:
            enhanced = upscale_bgr(cropped, model_name, outscale=scale)
    if enhanced.shape[1] != target_w or enhanced.shape[0] != target_h:
        enhanced = cv2.resize(enhanced, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
    return enhanced


def _open_encoder(output: Path, width: int, height: int, fps: float, settings: dict[str, Any]) -> subprocess.Popen:
    args = [
        "ffmpeg", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{width}x{height}", "-r", f"{fps:.8f}", "-i", "pipe:0", "-an",
        *video_encoder_args(settings),
        "-movflags", "+faststart", "-y", str(output),
    ]
    return subprocess.Popen(args, stdin=subprocess.PIPE)


def _mux_audio(video_only: Path, source: Path, output: Path, *, speed: float, timing_baked: bool, has_audio: bool) -> None:
    if not has_audio:
        video_only.replace(output)
        return
    args = ["ffmpeg", "-v", "error", "-i", str(video_only), "-i", str(source), "-map", "0:v:0", "-map", "1:a:0?"]
    if timing_baked and not math.isclose(speed, 1.0):
        # atempo supports 0.5..2.0 per stage. Chain when needed.
        remaining = speed
        stages: list[float] = []
        while remaining < 0.5:
            stages.append(0.5); remaining /= 0.5
        while remaining > 2.0:
            stages.append(2.0); remaining /= 2.0
        stages.append(remaining)
        args += ["-filter:a", ",".join(f"atempo={value:.8f}" for value in stages), "-c:a", "aac", "-b:a", "192k"]
    else:
        args += ["-c:a", "copy"]
    args += ["-c:v", "copy", "-shortest", "-tag:v", "hvc1", "-y", str(output)]
    subprocess.run(args, check=True, timeout=600)
    video_only.unlink(missing_ok=True)


def run_fast_video(job: dict[str, Any], cancel_event, progress: Progress) -> dict[str, Any]:
    input_data = job.get("input") or {}
    output_data = job.get("output") or {}
    settings = job.get("settings") or {}
    source_url = str(input_data.get("url") or "").strip()
    output_key = str(output_data.get("objectKey") or "").strip()
    if not source_url.startswith(("http://", "https://")) or not output_key:
        raise ValueError("video job requires input.url and output.objectKey")

    model_key = str(job.get("modelFamily") or settings.get("upscalerModel") or "realesrgan-real")
    model_name = "realesr-animevideov3" if "anime" in model_key.lower() else "realesr-general-x4v3"
    interpolation = str(settings.get("interpolationModel") or "none").lower()
    target_fps = int(settings.get("targetFps") or 0)
    if target_fps and target_fps not in {30, 48, 60}:
        raise ValueError("targetFps must be 30, 48 or 60")
    timing_baked = bool(settings.get("smoothSlowMotion") or settings.get("timingBaked"))
    speed = float(settings.get("directorSpeed") or 1.0)
    if speed <= 0:
        raise ValueError("directorSpeed must be positive")
    if interpolation == "gimm-vfi-f":
        # Runtime is packaged, but production use remains intentionally gated by
        # the unresolved commercial license in the canonical plan.
        raise RuntimeError("GIMM_LICENSE_NOT_CLEARED")
    if interpolation not in {"none", "rife-4.9", "rife"}:
        raise ValueError(f"unsupported local interpolation model: {interpolation}")
    video_encoder = normalize_video_encoder(settings)

    with tempfile.TemporaryDirectory(prefix="sb-enhancer-video-") as tmp:
        root = Path(tmp); source = root / "input.mp4"; video_only = root / "video.mp4"; final = root / "final.mp4"
        progress("downloading", 3, None); _download(source_url, source)
        probe = _probe(source)
        out_w, out_h = _target_dimensions(str(settings.get("targetResolution") or "1080p"), probe.width, probe.height)
        fps_out = float(target_fps or probe.fps or 24.0)
        encoder = _open_encoder(video_only, out_w, out_h, fps_out, settings)
        assert encoder.stdin is not None

        container = av.open(str(source)); stream = container.streams.video[0]
        time_base = float(stream.time_base)
        frames = iter(container.decode(stream))
        try:
            first = next(frames)
        except StopIteration:
            raise RuntimeError("FFMPEG_DECODE_FAILED:no frames")
        prev_raw = first.to_ndarray(format="bgr24")
        prev_pts = float(first.pts or 0) * time_base
        prev_sr = _spatial(prev_raw, model_name, out_w, out_h, settings)
        next_output_t = 0.0
        emitted = 0
        decoded = 1
        effective_speed = speed if timing_baked else 1.0
        nominal_source_fps = probe.fps or float(stream.average_rate or 24.0)
        neural_vfi = interpolation in {"rife-4.9", "rife"} and fps_out > nominal_source_fps * effective_speed + 1e-6

        def emit(frame: np.ndarray) -> None:
            nonlocal emitted
            if cancel_event.is_set():
                raise RuntimeError("CANCELLED")
            encoder.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())
            emitted += 1

        for current in frames:
            if cancel_event.is_set(): raise RuntimeError("CANCELLED")
            cur_raw = current.to_ndarray(format="bgr24")
            cur_pts = float(current.pts if current.pts is not None else decoded) * time_base
            if cur_pts <= prev_pts:
                cur_pts = prev_pts + 1.0 / nominal_source_fps
            cur_sr = _spatial(cur_raw, model_name, out_w, out_h, settings)
            cut = _hard_cut(prev_raw, cur_raw)
            while next_output_t * effective_speed <= cur_pts + 1e-9:
                src_t = next_output_t * effective_speed
                if src_t < prev_pts - 1e-9:
                    next_output_t += 1.0 / fps_out
                    continue
                alpha = max(0.0, min(1.0, (src_t - prev_pts) / (cur_pts - prev_pts)))
                if alpha <= 1e-6:
                    out = prev_sr
                elif alpha >= 1.0 - 1e-6:
                    out = cur_sr
                elif cut or not neural_vfi:
                    out = prev_sr if alpha < 0.5 else cur_sr
                else:
                    prev_rgb = cv2.cvtColor(prev_sr, cv2.COLOR_BGR2RGB)
                    cur_rgb = cv2.cvtColor(cur_sr, cv2.COLOR_BGR2RGB)
                    try:
                        out_rgb = try_interpolate_rife_trt(prev_rgb, cur_rgb, alpha, settings)
                    except Exception as error:
                        if not settings.get("allowNativeFallback", True):
                            raise
                        print(f"[enhancer] TensorRT RIFE fallback to native PyTorch: {error}", flush=True)
                        out_rgb = None
                    if out_rgb is None:
                        out_rgb = interpolate_rife(prev_rgb, cur_rgb, alpha)
                    out = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)
                emit(out)
                next_output_t += 1.0 / fps_out
            decoded += 1
            prev_raw, prev_pts, prev_sr = cur_raw, cur_pts, cur_sr
            percent = min(82.0, 8.0 + 74.0 * (cur_pts / max(probe.duration, cur_pts, 0.001)))
            progress("interpolating" if neural_vfi else "upscaling", percent, {"decoded": decoded, "emitted": emitted, "videoEncoder": video_encoder})

        # Emit tail through the authoritative selected duration.
        source_duration = probe.duration if probe.duration > 0 else prev_pts + 1.0 / nominal_source_fps
        output_duration = source_duration / effective_speed
        while next_output_t < output_duration - 1e-9:
            emit(prev_sr); next_output_t += 1.0 / fps_out
        container.close(); encoder.stdin.close()
        if encoder.wait(timeout=600) != 0:
            raise RuntimeError(video_encoder_failure_code(settings))
        progress("encoding", 88, {"frames": emitted, "targetFps": fps_out, "videoEncoder": video_encoder})
        _mux_audio(video_only, source, final, speed=speed, timing_baked=timing_baked, has_audio=probe.has_audio)
        progress("uploading", 94, None)
        stored = upload_file(final, output_key, "video/mp4")
        final_probe = _probe(final)
        progress("completed", 100, None)
        return {
            **stored,
            "runtime": "scenebuilder-enhancer-fast",
            "modelFamily": model_name,
            "interpolationModel": "rife-4.9" if neural_vfi else "none",
            "videoEncoder": video_encoder,
            "targetFps": fps_out,
            "timingBaked": timing_baked,
            "sourceSpeed": speed,
            "durationMs": round(final_probe.duration * 1000),
            "width": out_w,
            "height": out_h,
            "frames": emitted,
        }
