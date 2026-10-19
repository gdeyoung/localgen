"""Provider registry: which capabilities are live on THIS device right now.

The "library instead of instructions" core — an agent imports this, calls
plan(), and learns what it can generate here and through which backend, with
setup gaps named explicitly.
"""
from __future__ import annotations

import os

from . import doctor


def _image_status() -> dict:
    backends = doctor.probe()
    backend = backends["backends"]["image"]
    home = os.environ.get("LOCALSWAP_HOME", os.path.expanduser("~/.localswap"))
    model = os.environ.get("LOCALSWAP_IMAGE_MODEL", os.path.join(home, "models", "image"))
    ok = backend is not None and os.path.isdir(model)
    return {"ok": ok, "backend": backend,
            "missing": [] if ok else ([f"image model dir: {model}"] if backend else ["openvino (pip install openvino openvino-genai)"])}


def _tts_status() -> dict:
    home = os.environ.get("LOCALSWAP_HOME", os.path.expanduser("~/.localswap"))
    cli = os.environ.get("LOCALSWAP_BREEZE_CLI", os.path.join(home, "breeze", "breeze-cli"))
    model = os.environ.get("LOCALSWAP_BREEZE_MODEL", os.path.join(home, "breeze", "breeze-tts-2-q8_0.gguf"))
    missing = [m for m, p in (("breeze-cli (run scripts/build-breeze.sh)", cli),
                              ("breeze model gguf", model)) if not os.path.exists(p)]
    return {"ok": not missing, "backend": "breeze", "missing": missing,
            "note": "voice cloning requires YOUR reference wav + transcript per call"}


def _stt_status() -> dict:
    try:
        import faster_whisper  # type: ignore  # noqa: F401
        return {"ok": True, "backend": "faster-whisper", "missing": []}
    except ImportError:
        return {"ok": False, "backend": "faster-whisper",
                "missing": ["faster-whisper (pip install faster-whisper)"]}


def _video_status() -> dict:
    backends = doctor.probe()
    backend = backends["backends"]["image"]      # same OpenVINO substrate
    home = os.environ.get("LOCALSWAP_HOME", os.path.expanduser("~/.localswap"))
    model = os.environ.get("LOCALSWAP_VIDEO_MODEL",
                           os.path.join(home, "models", "video", "ltx-video-int8"))
    if backend is None:
        missing = ["openvino (pip install openvino openvino-genai)"]
    elif not os.path.isdir(model):
        missing = [f"video model dir: {model} (run scripts/export-ltx.sh)"]
    else:
        missing = []
    return {"ok": not missing, "backend": backend, "missing": missing,
            "tier": "heavy",
            "note": "~23s per second of video at defaults on a Core Ultra "
                    "iGPU (LTX-Video int8, measured 2026-10); long-form/"
                    "hi-res → discrete CUDA"}


CAPABILITIES = {
    "image": _image_status,
    "video": _video_status,
    "tts": _tts_status,
    "transcribe": _stt_status,
}


def plan() -> dict:
    """What can this device generate locally right now, and what's missing."""
    return {name: fn() for name, fn in CAPABILITIES.items()}
