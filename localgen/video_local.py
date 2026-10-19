"""Local video generation via OpenVINO GenAI (LTX-Video family).

This is the iGPU lane for drafts and B-roll: expect minutes per second of
video at modest resolutions on integrated GPUs. Long-form / high-res video
belongs on discrete CUDA hardware (e.g. ComfyUI with Wan or LTX-2).

Models: any OpenVINO-export of the LTX family. `scripts/export-ltx.sh`
produces the recommended int8 export of Lightricks/LTX-Video.

LTX-2/LTX-2.3 exports additionally generate a synchronized audio track;
if present it is written alongside and muxed in (audio path exercised only
on LTX-Video, which has none).

Paths: LOCALGEN_VIDEO_MODEL (model dir). Default:
$LOCALGEN_HOME/models/video/ltx-video-int8.
"""
from __future__ import annotations

import os
import subprocess
import time


def _resolve_model(model: str | None) -> str:
    if model:
        return os.path.expanduser(model)
    env = os.environ.get("LOCALGEN_VIDEO_MODEL")
    if env:
        return env
    home = os.environ.get("LOCALGEN_HOME", os.path.expanduser("~/.localgen"))
    return os.path.join(home, "models", "video", "ltx-video-int8")


def _pick_device(device: str, available: list[str]) -> str:
    if device and device != "auto":
        return device
    gpus = [d for d in available if d.startswith("GPU")]
    return gpus[0] if gpus else "CPU"


def _ffmpeg_cmd(out: str, width: int, height: int, fps: int) -> list[str]:
    """Encoder command; frames arrive on stdin as raw rgb24."""
    return ["ffmpeg", "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{width}x{height}", "-r", str(fps), "-i", "-",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", out]


def _encode(frames, out: str, width: int, height: int, fps: int) -> float:
    """Write an [F,H,W,3] uint8 frame stack to an mp4 via ffmpeg stdin."""
    t0 = time.time()
    proc = subprocess.Popen(_ffmpeg_cmd(out, width, height, fps),
                            stdin=subprocess.PIPE)
    assert proc.stdin is not None
    for frame in frames:
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError(f"ffmpeg failed with code {proc.returncode}")
    return time.time() - t0


def _save_audio(result, out: str) -> str | None:
    """LTX-2 family: write the joint audio track and mux it in. None otherwise."""
    audio = getattr(result, "audio", None)
    rate = getattr(result, "audio_sample_rate", 0)
    if audio is None or not rate:
        return None
    import wave

    import numpy as np

    wav_path = out.rsplit(".", 1)[0] + ".wav"
    data = np.asarray(audio.data if hasattr(audio, "data") else audio)
    if data.dtype != np.int16:
        data = data.clip(-1.0, 1.0) if data.dtype.kind == "f" else data
        data = (data * 32767).astype("<i2") if data.dtype.kind == "f" else data.astype("<i2")
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    with wave.open(wav_path, "wb") as wf:
        wf.setnchannels(data.shape[1])
        wf.setsampwidth(2)
        wf.setframerate(int(rate))
        wf.writeframes(data.tobytes())
    muxed = out.rsplit(".", 1)[0] + "_muxed.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", out, "-i", wav_path,
                    "-c:v", "copy", "-c:a", "aac", "-shortest", muxed], check=True)
    os.replace(muxed, out)
    return wav_path


def generate_video(prompt: str, out: str = "out.mp4", *, width: int = 768,
                   height: int = 448, num_frames: int = 61, steps: int = 20,
                   seed: int = 42, fps: int = 25, device: str = "auto",
                   model: str | None = None, guidance_scale: float = 3.0,
                   negative_prompt: str | None = None,
                   finish: str | None = None) -> dict:
    """Text-to-video on this device's Intel silicon. Returns timing + paths.

    LTX snaps num_frames down to the nearest 8n+1 (61 -> 57).

    finish=None (raw) | "upscale2x" (2x Real-ESRGAN) | "smooth2x"
    (RIFE 2x fps -> 2x upscale; the production pattern)."""
    model_dir = _resolve_model(model)
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(
            f"model dir not found: {model_dir} — run scripts/export-ltx.sh "
            f"or set LOCALGEN_VIDEO_MODEL"
        )
    import numpy as np
    import openvino as ov
    import openvino_genai

    dev = _pick_device(device, list(ov.Core().available_devices))

    t0 = time.time()
    pipe = openvino_genai.Text2VideoPipeline(model_dir, device=dev)
    t1 = time.time()
    kwargs: dict[str, object] = dict(width=width, height=height, num_frames=num_frames,
                                     num_inference_steps=steps, guidance_scale=guidance_scale,
                                     seed=seed)
    if negative_prompt:
        kwargs["negative_prompt"] = negative_prompt
    result = pipe.generate(prompt, **kwargs)
    t2 = time.time()

    stack = np.asarray(result.video.data)          # [B, F, H, W, 3] uint8
    frames = stack[0]
    encode_s = _encode(frames, out, width, height, fps)
    wav = _save_audio(result, out)
    res = {"out": os.path.abspath(out), "audio": wav, "device": dev,
           "frames": int(frames.shape[0]), "fps": fps,
           "seconds": round(frames.shape[0] / fps, 2),
           "resolution": f"{width}x{height}", "steps": steps, "seed": seed,
           "compile_s": round(t1 - t0, 1), "generate_s": round(t2 - t1, 1),
           "encode_s": round(encode_s, 1)}
    if finish is not None:
        if finish not in ("upscale2x", "smooth2x"):
            raise ValueError("finish must be None | 'upscale2x' | 'smooth2x'")
        from .enhance_local import interpolate, upscale
        stem = out.rsplit(".", 1)[0]
        if finish == "smooth2x":
            res["interp"] = interpolate(out, stem + "_50fps.mp4", mult=2)
            out = stem + "_50fps.mp4"
        res["upscale"] = upscale(out, stem + "_2x.mp4", factor=2)
        res["out"] = res["upscale"]["out"]
        res["finish"] = finish
    res["note"] = ("iGPU lane: ~23s per second of video at defaults on a "
                   "Core Ultra iGPU (CPU fallback ~12x slower); route "
                   "long-form/hi-res to discrete CUDA")
    return res


def image_to_video(prompt: str, image: str, out: str = "out_i2v.mp4", *,
                   num_frames: int = 61, steps: int = 20, seed: int = 42,
                   fps: int = 25, device: str = "auto", model: str | None = None,
                   guidance_scale: float = 3.0) -> dict:
    """Animate a conditioning image (Image2VideoPipeline). Resolution follows
    the input image; width/height are not overridden."""
    model_dir = _resolve_model(model)
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(
            f"model dir not found: {model_dir} — run scripts/export-ltx.sh "
            f"or set LOCALGEN_VIDEO_MODEL"
        )
    import numpy as np
    import openvino as ov
    import openvino_genai
    from PIL import Image

    dev = _pick_device(device, list(ov.Core().available_devices))

    img = Image.open(os.path.expanduser(image)).convert("RGB")
    arr = np.asarray(img)[np.newaxis, ...]         # [1, H, W, 3] uint8
    ov_img = ov.Tensor(arr)

    t0 = time.time()
    pipe = openvino_genai.Image2VideoPipeline(model_dir, device=dev)
    t1 = time.time()
    result = pipe.generate(prompt, image=ov_img, num_frames=num_frames,
                           num_inference_steps=steps,
                           guidance_scale=guidance_scale, seed=seed)
    t2 = time.time()

    stack = np.asarray(result.video.data)
    frames = stack[0]
    w, h = frames.shape[2], frames.shape[1]
    encode_s = _encode(frames, out, w, h, fps)
    return {"out": os.path.abspath(out), "audio": _save_audio(result, out),
            "device": dev, "frames": int(frames.shape[0]), "fps": fps,
            "seconds": round(frames.shape[0] / fps, 2),
            "resolution": f"{w}x{h}", "steps": steps, "seed": seed,
            "compile_s": round(t1 - t0, 1), "generate_s": round(t2 - t1, 1),
            "encode_s": round(encode_s, 1)}
