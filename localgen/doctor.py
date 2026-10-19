"""Device probe: what accelerators does this machine have, and what can localgen use?

Providers are expected to call probe() once and pick their backend from the
result. Everything is a plain dict — JSON-serializable for agent consumption.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess


def _openvino_devices() -> list[str]:
    try:
        import openvino as ov  # type: ignore

        core = ov.Core()
        return list(core.available_devices)
    except Exception:
        return []


def _has_vulkan() -> bool:
    if shutil.which("vulkaninfo"):
        r = subprocess.run(["vulkaninfo", "--summary"], capture_output=True, text=True, timeout=20)
        return r.returncode == 0 and "deviceName" in (r.stdout + r.stderr)
    return bool(shutil.which("vkjson") or _ldconfig_has("libvulkan"))


def _ldconfig_has(lib: str) -> bool:
    r = subprocess.run(["sh", "-c", f"ldconfig -p 2>/dev/null | grep -c {lib}"], capture_output=True, text=True)
    return r.stdout.strip().isdigit() and int(r.stdout.strip()) > 0


def _has_cuda() -> bool:
    return bool(shutil.which("nvidia-smi"))


def probe() -> dict:
    """Return a device report: accelerators found + localgen backends they enable."""
    ov_devices = _openvino_devices()
    vulkan = _has_vulkan()
    cuda = _has_cuda()
    report = {
        "openvino_devices": ov_devices,
        "vulkan": vulkan,
        "cuda": cuda,
        "cpu_cores": os.cpu_count() or 0,
        "ffmpeg": bool(shutil.which("ffmpeg")),
        "backends": {},
    }
    # Image generation runs on OpenVINO: GPU (Intel iGPU/dGPU) when present, else CPU.
    report["backends"]["image"] = "openvino-gpu" if any(d.startswith("GPU") for d in ov_devices) else (
        "openvino-cpu" if "CPU" in ov_devices else None
    )
    # Breeze TTS runs on Vulkan (GPU) or CPU.
    report["backends"]["tts"] = "breeze-vulkan" if vulkan else "breeze-cpu"
    # Whisper-family STT: CPU everywhere, CUDA when present.
    report["backends"]["transcribe"] = "faster-whisper-cuda" if cuda else "faster-whisper-cpu"
    report["ready"] = {
        "image": report["backends"]["image"] is not None,
        "tts": True,
        "transcribe": True,  # CPU always possible; model download is the gate
        "ffmpeg_required_for_tts": report["ffmpeg"],
    }
    return report


def main() -> int:
    print(json.dumps(probe(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
