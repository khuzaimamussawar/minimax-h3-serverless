from __future__ import annotations

import importlib.util
import json
import math
import os
import subprocess
import tempfile
import urllib.request
from pathlib import Path
from typing import Any, Callable

from .r2_store import upload_file
from .vfi_postprocess import interpolate_file
from .video_encoder import normalize_video_encoder, video_encoder_args, video_encoder_failure_code

FLASH_ROOT = Path(os.environ.get("FLASHVSR_ROOT", "/opt/FlashVSR"))
SCRIPT = FLASH_ROOT / "examples/WanVSR/infer_flashvsr_v1.1_tiny_long_video.py"
_PIPE = None
_MODULE = None
Progress = Callable[[str, float, dict[str, Any] | None], None]

TARGETS = {"1080p": (1920, 1080), "1440p": (2560, 1440), "2k": (2560, 1440), "2160p": (3840, 2160), "4k": (3840, 2160)}


def _module():
    global _MODULE
    if _MODULE is None:
        os.chdir(str(FLASH_ROOT / "examples/WanVSR"))
        spec = importlib.util.spec_from_file_location("flashvsr_runtime", SCRIPT)
        if spec is None or spec.loader is None: raise RuntimeError("FLASHVSR_SELF_TEST_FAILED:module spec")
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); _MODULE = module
    return _MODULE


def _pipe():
    global _PIPE
    if _PIPE is None:
        _PIPE = _module().init_pipeline()
    return _PIPE


def _download(url: str, path: Path) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "SceneBuilder-Enhancer/2.0"})
    with urllib.request.urlopen(req, timeout=120) as response, path.open("wb") as out:
        while True:
            chunk = response.read(8 * 1024 * 1024)
            if not chunk: break
            out.write(chunk)


def _probe(path: Path) -> dict[str, Any]:
    data = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)], text=True))
    v = next(s for s in data["streams"] if s.get("codec_type") == "video")
    duration = float(v.get("duration") or data.get("format", {}).get("duration") or 0)
    return {"width": int(v["width"]), "height": int(v["height"]), "duration": duration, "hasAudio": any(s.get("codec_type") == "audio" for s in data["streams"])}


def _atempo(speed: float) -> str:
    remaining = speed; stages: list[float] = []
    while remaining < 0.5: stages.append(0.5); remaining /= 0.5
    while remaining > 2.0: stages.append(2.0); remaining /= 2.0
    stages.append(remaining)
    return ",".join(f"atempo={value:.8f}" for value in stages)


def _finish_video(video: Path, source: Path, output: Path, width: int, height: int, *, speed: float, timing_baked: bool, has_audio: bool, settings: dict[str, Any]) -> None:
    args = ["ffmpeg", "-v", "error", "-i", str(video)]
    if has_audio: args += ["-i", str(source)]
    args += ["-map", "0:v:0"]
    if has_audio: args += ["-map", "1:a:0?"]
    args += ["-vf", f"scale={width}:{height}:flags=lanczos", *video_encoder_args(settings)]
    if has_audio and timing_baked and not math.isclose(speed, 1.0): args += ["-filter:a", _atempo(speed), "-c:a", "aac", "-b:a", "192k"]
    elif has_audio: args += ["-c:a", "copy"]
    args += ["-shortest", "-movflags", "+faststart", "-y", str(output)]
    completed = subprocess.run(args, capture_output=True, text=True, timeout=1200, check=False)
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "no ffmpeg output")[-4000:]
        raise RuntimeError(f"{video_encoder_failure_code(settings)}:{detail}")


def run_video_upscale(job: dict[str, Any], cancel_event, progress: Progress) -> dict[str, Any]:
    source_url = str((job.get("input") or {}).get("url") or "").strip()
    output_key = str((job.get("output") or {}).get("objectKey") or "").strip()
    settings = job.get("settings") or {}
    if not source_url.startswith(("http://", "https://")): raise ValueError("video_upscale input.url must be HTTP(S)")
    if not output_key: raise ValueError("output.objectKey is required")
    interpolation = str(settings.get("interpolationModel") or "none").lower()
    if interpolation == "gimm-vfi-f": raise RuntimeError("GIMM_LICENSE_NOT_CLEARED")
    if interpolation not in {"none", "rife", "rife-4.9"}: raise ValueError(f"Unsupported local VFI: {interpolation}")
    video_encoder = normalize_video_encoder(settings)

    import torch
    if not torch.cuda.is_available(): raise RuntimeError("CUDA_UNAVAILABLE")
    if not torch.cuda.is_bf16_supported(): raise RuntimeError("GPU_CAPABILITY_MISMATCH:BF16")

    with tempfile.TemporaryDirectory(prefix="sb-flashvsr-") as tmp:
        root = Path(tmp); source = root / "input.mp4"; flash_raw = root / "flash.mp4"; temporal = root / "temporal.mp4"; final = root / "final.mp4"
        progress("downloading", 3, None); _download(source_url, source)
        source_probe = _probe(source)
        target = TARGETS.get(str(settings.get("targetResolution") or "1080p").lower())
        if not target: raise ValueError("Unsupported target resolution")
        target_w, target_h = target[::-1] if source_probe["height"] > source_probe["width"] else target
        scale = min(4.0, max(1.0, max(target_w / source_probe["width"], target_h / source_probe["height"])))
        progress("preparing_model", 10, {"precision": "bf16", "scale": scale, "videoEncoder": video_encoder})
        pipe = _pipe(); mod = _module()
        if cancel_event.is_set(): raise RuntimeError("CANCELLED")
        lq, th, tw, frame_count, fps = mod.prepare_input_tensor(str(source), scale=scale, dtype=torch.bfloat16, device="cuda")
        if not lq.is_cuda:
            # Official helper currently assembles on CPU; move as one explicit GPU boundary.
            lq = lq.to("cuda", dtype=torch.bfloat16)
        progress("upscaling", 20, {"frames": frame_count, "internalWidth": tw, "internalHeight": th})
        with torch.inference_mode():
            video = pipe(prompt="", negative_prompt="", cfg_scale=1.0, num_inference_steps=1, seed=0,
                         LQ_video=lq, num_frames=frame_count, height=th, width=tw, is_full_block=False,
                         if_buffer=True, topk_ratio=2.0 * 768 * 1280 / (th * tw), kv_ratio=3.0,
                         local_range=11, color_fix=True)
        if cancel_event.is_set(): raise RuntimeError("CANCELLED")
        frames = mod.tensor2video(video); mod.save_video(frames, str(flash_raw), fps=fps, quality=5)
        del video, lq, frames; torch.cuda.empty_cache()

        speed = float(settings.get("directorSpeed") or 1.0); timing_baked = bool(settings.get("smoothSlowMotion") or settings.get("timingBaked"))
        target_fps = int(settings.get("targetFps") or 0)
        base_for_finish = flash_raw
        vfi_meta: dict[str, Any] = {"neuralVfi": False, "videoEncoder": video_encoder}
        if target_fps:
            progress("interpolating", 72, {"model": interpolation, "targetFps": target_fps, "videoEncoder": video_encoder})
            vfi_meta = interpolate_file(flash_raw, temporal, target_fps=target_fps, playback_speed=speed,
                                        timing_baked=timing_baked, interpolation_model=interpolation,
                                        cq=int(settings.get("nvencCq") or 17), cancel_event=cancel_event, progress=progress,
                                        settings=settings)
            base_for_finish = temporal
        progress("encoding", 91, {"videoEncoder": video_encoder})
        _finish_video(base_for_finish, source, final, target_w, target_h, speed=speed,
                      timing_baked=timing_baked and not target_fps, has_audio=source_probe["hasAudio"], settings=settings)
        # interpolate_file changes video timing but intentionally emits video-only;
        # mux/stretches audio here when VFI ran.
        if target_fps and source_probe["hasAudio"]:
            remux = root / "remux.mp4"
            args = ["ffmpeg", "-v", "error", "-i", str(final), "-i", str(source), "-map", "0:v:0", "-map", "1:a:0?", "-c:v", "copy"]
            if timing_baked and not math.isclose(speed, 1.0): args += ["-filter:a", _atempo(speed), "-c:a", "aac", "-b:a", "192k"]
            else: args += ["-c:a", "copy"]
            args += ["-shortest", "-tag:v", "hvc1", "-y", str(remux)]
            subprocess.run(args, check=True, timeout=600); remux.replace(final)
        progress("uploading", 96, None); stored = upload_file(final, output_key, "video/mp4")
        final_probe = _probe(final); progress("completed", 100, None)
        return {**stored, "runtime": "scenebuilder-enhancer-quality", "modelFamily": "flashvsr-v1.1", "precision": "bf16",
                "interpolationModel": "rife-4.9" if vfi_meta.get("neuralVfi") else "none", "videoEncoder": video_encoder,
                "targetFps": target_fps or fps, "timingBaked": timing_baked, "sourceSpeed": speed,
                "durationMs": round(final_probe["duration"] * 1000), "width": target_w, "height": target_h}
