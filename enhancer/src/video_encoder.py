from __future__ import annotations

import functools
import subprocess
from typing import Any

QUALITY_MIN = 12
QUALITY_MAX = 25


def normalize_video_encoder(settings: dict[str, Any] | None = None) -> str:
    raw = str((settings or {}).get("videoEncoder") or (settings or {}).get("video_encoder") or "nvenc").strip().lower()
    if raw in {"nvenc", "hevc_nvenc", "nvidia"}:
        return "nvenc"
    if raw in {"x265", "libx265", "hevc_x265"}:
        return "x265"
    raise ValueError(f"Unsupported video encoder: {raw}")


def _clamp_quality(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(QUALITY_MIN, min(QUALITY_MAX, parsed))


@functools.lru_cache(maxsize=2)
def _ensure_encoder_available(encoder: str) -> None:
    completed = subprocess.run(
        ["ffmpeg", "-hide_banner", "-encoders"],
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    text = f"{completed.stdout}\n{completed.stderr}"
    expected = "hevc_nvenc" if encoder == "nvenc" else "libx265"
    if completed.returncode != 0 or expected not in text:
        code = "NVENC_HEVC_ENCODER_MISSING" if encoder == "nvenc" else "X265_ENCODER_MISSING"
        raise RuntimeError(code)


def video_encoder_args(settings: dict[str, Any] | None = None) -> list[str]:
    cfg = settings or {}
    encoder = normalize_video_encoder(cfg)
    _ensure_encoder_available(encoder)
    if encoder == "x265":
        crf = _clamp_quality(cfg.get("x265Crf"), 15)
        return [
            "-c:v", "libx265",
            "-preset", "medium",
            "-crf", str(crf),
            "-pix_fmt", "yuv420p10le",
            "-tag:v", "hvc1",
        ]
    cq = _clamp_quality(cfg.get("nvencCq"), 17)
    return [
        "-c:v", "hevc_nvenc",
        "-profile:v", "main10",
        "-pix_fmt", "p010le",
        "-preset", "p6",
        "-rc", "vbr",
        "-cq", str(cq),
        "-tag:v", "hvc1",
    ]


def video_encoder_failure_code(settings: dict[str, Any] | None = None) -> str:
    return "X265_ENCODE_FAILED" if normalize_video_encoder(settings) == "x265" else "NVENC_ENCODE_FAILED"


__all__ = ["normalize_video_encoder", "video_encoder_args", "video_encoder_failure_code", "QUALITY_MIN", "QUALITY_MAX"]
