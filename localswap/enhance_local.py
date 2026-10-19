"""Post-generation enhancement: neural upscale (Real-ESRGAN) + frame
interpolation (RIFE), both as ncnn-vulkan binaries — GPU lane via Vulkan,
no torch, no cloud.

The production pattern is generate -> interpolate -> upscale (generate at the
model's native resolution/fps, then smooth, then add detail).
``video_local.generate_video`` chains it via ``finish="smooth2x"``.

Binaries are BYO (~50MB each, no weights bundled):
  realesrgan-ncnn-vulkan  <- realesrgan-ncnn-vulkan-20220424-ubuntu.zip
                            (github.com/xinntao/Real-ESRGAN release v0.2.5.0)
  rife-ncnn-vulkan        <- rife-ncnn-vulkan-20221029-ubuntu.zip
                            (github.com/nihui/rife-ncnn-vulkan)
Default home: $LOCALSWAP_HOME/tools/ ; overrides LOCALSWAP_REALESRGAN_BIN /
LOCALSWAP_RIFE_BIN. Verified on Vulkan iGPU 2026-10: 2x upscale of a 57-frame
768x448 clip in ~16s; model dirs ship inside each zip.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import time


def _tools_home() -> str:
    home = os.environ.get("LOCALSWAP_HOME", os.path.expanduser("~/.localswap"))
    return os.path.join(home, "tools")


def realesrgan_bin() -> str:
    env = os.environ.get("LOCALSWAP_REALESRGAN_BIN")
    if env:
        return env
    return os.path.join(_tools_home(), "realesrgan-ncnn-vulkan")


def rife_bin() -> str:
    env = os.environ.get("LOCALSWAP_RIFE_BIN")
    if env:
        return env
    return os.path.join(_tools_home(), "rife-ncnn-vulkan-20221029-ubuntu",
                        "rife-ncnn-vulkan")


def _rsg_cmd(binpath: str, indir: str, outdir: str, model: str, factor: int) -> list[str]:
    return [binpath, "-i", indir, "-o", outdir, "-n", model, "-s", str(factor)]


def _rife_cmd(binpath: str, indir: str, outdir: str, model: str, n_target: int) -> list[str]:
    # rife's -m takes the MODEL DIR path (e.g. rife-v4.6 inside the zip)
    model_dir = os.path.join(os.path.dirname(binpath), model)
    return [binpath, "-i", indir, "-o", outdir, "-n", str(n_target), "-m", model_dir]


def _fps_of(rate: str) -> float:
    """'25/1' -> 25.0"""
    num, _, den = rate.partition("/")
    den = int(den) if den else 1
    return int(num) / den


def probe_video(path: str) -> dict:
    """fps / frame count / audio presence via ffprobe."""
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=r_frame_rate,nb_frames",
         "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1", path],
        capture_output=True, text=True, check=True)
    info = dict(line.split("=", 1) for line in r.stdout.strip().splitlines() if "=" in line)
    has_audio = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries",
         "stream=index", "-of", "csv=p=0", path],
        capture_output=True, text=True).stdout.strip() != ""
    fps = _fps_of(info.get("r_frame_rate", "25/1"))
    n = int(info.get("nb_frames") or 0)
    if n == 0 and "duration" in info:            # nb_frames can be missing
        n = round(float(info["duration"]) * fps)
    return {"fps": fps, "frames": n, "audio": has_audio}


def _extract(video: str, workdir: str) -> None:
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", video,
                    os.path.join(workdir, "in", "%06d.png")], check=True)


def _renumber(directory: str) -> None:
    """Normalize tool output names to %06d.png in sorted order."""
    names = sorted(n for n in os.listdir(directory)
                   if n.lower().endswith((".png", ".jpg")))
    for i, name in enumerate(names, start=1):
        if name != f"{i:06d}.png":
            os.replace(os.path.join(directory, name),
                       os.path.join(directory, f"{i:06d}.png"))


def _assemble(directory: str, out: str, fps: float, audio_from: str | None) -> None:
    cmd = ["ffmpeg", "-y", "-loglevel", "error",
           "-framerate", f"{fps:g}", "-i", os.path.join(directory, "%06d.png")]
    if audio_from:
        cmd += ["-i", audio_from, "-map", "0:v:0", "-map", "1:a:0?",
                "-c:a", "aac", "-shortest"]
    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "17",
            "-movflags", "+faststart", out]
    subprocess.run(cmd, check=True)


def _is_image(path: str) -> bool:
    return path.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))


def upscale(in_path: str, out_path: str, *, factor: int = 2,
            model: str = "realesr-animevideov3") -> dict:
    """Neural upscale of a video (or single image). Returns timings."""
    binpath = realesrgan_bin()
    if not os.path.isfile(binpath):
        raise FileNotFoundError(
            f"realesrgan-ncnn-vulkan not found: {binpath} — fetch the ubuntu "
            f"zip from the Real-ESRGAN v0.2.5.0 release into $LOCALSWAP_HOME/tools/")
    t0 = time.time()
    if _is_image(in_path):
        subprocess.run(_rsg_cmd(binpath, os.path.abspath(in_path),
                                os.path.abspath(out_path), model, factor),
                       check=True, stdout=subprocess.DEVNULL)
        return {"out": os.path.abspath(out_path), "factor": factor, "model": model,
                "took_s": round(time.time() - t0, 1)}
    info = probe_video(in_path)
    with tempfile.TemporaryDirectory(prefix="localswap-up-") as work:
        src, dst = os.path.join(work, "in"), os.path.join(work, "out")
        os.makedirs(src), os.makedirs(dst)
        _extract(in_path, work)
        subprocess.run(_rsg_cmd(binpath, src, dst, model, factor),
                       check=True, stdout=subprocess.DEVNULL)
        _renumber(dst)
        _assemble(dst, out_path, info["fps"],
                  in_path if info["audio"] else None)
    return {"out": os.path.abspath(out_path), "factor": factor, "model": model,
            "frames": info["frames"], "fps": info["fps"],
            "took_s": round(time.time() - t0, 1)}


def interpolate(in_path: str, out_path: str, *, mult: float = 2,
                model: str = "rife-v4.6") -> dict:
    """Raise video fps by RIFE interpolation (duration unchanged, audio kept)."""
    binpath = rife_bin()
    if not os.path.isfile(binpath):
        raise FileNotFoundError(
            f"rife-ncnn-vulkan not found: {binpath} — fetch the ubuntu zip "
            f"from the nihui/rife-ncnn-vulkan 20221029 release into $LOCALSWAP_HOME/tools/")
    t0 = time.time()
    info = probe_video(in_path)
    n_target = max(2, round(info["frames"] * mult))
    with tempfile.TemporaryDirectory(prefix="localswap-rife-") as work:
        src, dst = os.path.join(work, "in"), os.path.join(work, "out")
        os.makedirs(src), os.makedirs(dst)
        _extract(in_path, work)
        _renumber(src)                              # rife wants contiguous names
        subprocess.run(_rife_cmd(binpath, src, dst, model, n_target),
                       check=True, stdout=subprocess.DEVNULL)
        _renumber(dst)
        _assemble(dst, out_path, info["fps"] * (n_target / max(1, info["frames"])),
                  in_path if info["audio"] else None)
    return {"out": os.path.abspath(out_path), "frames_in": info["frames"],
            "frames_out": n_target, "model": model,
            "took_s": round(time.time() - t0, 1)}
