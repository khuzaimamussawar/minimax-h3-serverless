from pathlib import Path
import subprocess
from unittest.mock import patch

from enhancer.src import gpu


def test_nvenc_smoke_uses_valid_hevc_main10_geometry():
    calls: list[list[str]] = []

    def fake_cmd(args: list[str], timeout: int = 20) -> str:
        calls.append(list(args))
        if "-encoders" in args:
            return " V..... hevc_nvenc NVIDIA NVENC hevc encoder"
        output = Path(args[-1])
        output.write_bytes(b"smoke")
        return ""

    with patch.object(gpu, "_cmd", side_effect=fake_cmd):
        gpu._nvenc_smoke()

    encode = next(args for args in calls if "hevc_nvenc" in args and "-encoders" not in args)
    assert "color=c=black:s=256x256:r=1" in encode
    assert "-profile:v" in encode
    assert "main10" in encode
    assert "p010le" in encode
    assert "128x128" not in " ".join(encode)


def test_command_failure_includes_process_output():
    failure = subprocess.CalledProcessError(234, ["ffmpeg"], output="NVENC diagnostic detail")
    with patch.object(gpu.subprocess, "run", side_effect=failure):
        try:
            gpu._cmd(["ffmpeg", "-version"])
        except RuntimeError as error:
            text = str(error)
            assert "COMMAND_FAILED:234" in text
            assert "NVENC diagnostic detail" in text
        else:
            raise AssertionError("expected RuntimeError")
