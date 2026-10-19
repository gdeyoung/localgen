"""Local image generation via OpenVINO GenAI (Intel iGPU / GPU / CPU).

Works with any converted SD-family model directory (e.g. LCM Dreamshaper int8,
diffusers format exported for OpenVINO). Verify your model supports few-step
LCM sampling before lowering --steps below 8.

Paths: LOCALSWAP_IMAGE_MODEL (model dir). Default: $LOCALSWAP_HOME/models/image.
"""
from __future__ import annotations

import os
import time


def _resolve_model(model: str | None) -> str:
    if model:
        return os.path.expanduser(model)
    home = os.environ.get("LOCALSWAP_HOME", os.path.expanduser("~/.localswap"))
    return os.path.join(home, "models", "image")


def _pick_device(device: str, available: list[str]) -> str:
    if device and device != "auto":
        return device
    gpus = [d for d in available if d.startswith("GPU")]
    return gpus[0] if gpus else "CPU"


def generate(prompt: str, out: str = "out.png", *, width: int = 768, height: int = 432,
             steps: int = 4, seed: int = 42, device: str = "auto",
             model: str | None = None, guidance_scale: float = 1.0) -> dict:
    """Generate one image locally. Returns timing + paths dict."""
    import numpy as np
    import openvino_genai
    from PIL import Image

    model_dir = _resolve_model(model)
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(
            f"model dir not found: {model_dir} — set LOCALSWAP_IMAGE_MODEL or pass model="
        )
    import openvino as ov

    dev = _pick_device(device, list(ov.Core().available_devices))

    t0 = time.time()
    pipe = openvino_genai.Text2ImagePipeline(model_dir, device=dev)
    t1 = time.time()
    tensor = pipe.generate(prompt, width=width, height=height,
                           num_inference_steps=steps, guidance_scale=guidance_scale,
                           seed=seed)
    t2 = time.time()

    arr = np.array(tensor.data[0])
    if arr.dtype != np.uint8:
        arr = arr.clip(0, 255).astype("uint8")
    Image.fromarray(arr).convert("RGB").save(out)
    return {"out": os.path.abspath(out), "device": dev, "steps": steps, "seed": seed,
            "compile_s": round(t1 - t0, 1), "generate_s": round(t2 - t1, 1)}
