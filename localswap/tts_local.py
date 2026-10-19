"""Local cloned-voice narration via Breeze-TTS-2.cpp (Vulkan or CPU).

You supply the voice: a reference WAV + its verbatim transcript. Nothing
personal ships with this library. Build the CLI once with
scripts/build-breeze.sh, point LOCALSWAP_BREEZE_CLI / LOCALSWAP_BREEZE_MODEL
at it (defaults under $LOCALSWAP_HOME/breeze/), and narrate:

    from localswap.tts_local import narrate
    narrate("Long script...", "out.wav", ref_wav="me.wav", ref_text="...")

Pipeline per chunk (VERIFIED against breeze-cli flag surface): deterministic
seeds, edge-silence trim, concat, narration mastering, 48 kHz.
"""
from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

WORDS_PER_SEC = 1 / 0.42          # ~0.42 s/word planning rate
MAX_CHUNK_SEC = 9.0               # breeze reference-audio conditioning window
TRIM_FILTER = (
    "areverse,silenceremove=start_periods=1:start_silence=0.12:start_threshold=-45dB,areverse,"
    "silenceremove=start_periods=1:start_silence=0.05:start_threshold=-45dB"
)
MASTER_FILTER = "deesser=i=0.35:f=0.55,highshelf=f=8000:g=-2,loudnorm=I=-18:TP=-2.0:LRA=9"


def _sh(cmd: list[str]) -> None:
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"command failed ({cmd[0]}): {r.stderr[-800:]}")


def _dur(path: str) -> float:
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                        "-of", "csv=p=0", path], capture_output=True, text=True)
    return float(r.stdout.strip())


def chunk_script(text: str, max_sec: float = MAX_CHUNK_SEC) -> list[str]:
    """Split at sentence boundaries into chunks sized for the conditioning window."""
    sents = re.split(r"(?<=[.!?])\s+", text.strip())
    out: list[str] = []
    cur = ""
    for s in sents:
        trial = (cur + " " + s).strip()
        if len(trial.split()) * 0.42 <= max_sec or not cur:
            cur = trial
        else:
            out.append(cur)
            cur = s
    if cur:
        out.append(cur)
    return out


def seeds_for(n: int, base: int = 1000, offset: int = 0) -> list[int]:
    """Deterministic seed per chunk: base + offset*10 + index (stable across reruns)."""
    return [base + offset * 10 + i for i in range(n)]


def narrate(text: str, out: str, *, ref_wav: str, ref_text: str,
            instruction: str = "Narrate warmly and clearly.",
            seed_base: int = 1000, offset: int = 0, cpu: bool = False,
            workdir: str | None = None, cli: str | None = None,
            model: str | None = None) -> dict:
    """Synthesize `text` in the cloned voice. Returns file + duration + chunk count."""
    home = Path(os.environ.get("LOCALSWAP_HOME", os.path.expanduser("~/.localswap")))
    cli = cli or os.environ.get("LOCALSWAP_BREEZE_CLI", str(home / "breeze" / "breeze-cli"))
    model = model or os.environ.get("LOCALSWAP_BREEZE_MODEL", str(home / "breeze" / "breeze-tts-2-q8_0.gguf"))
    for p, label in ((cli, "breeze-cli"), (model, "breeze model"), (ref_wav, "ref_wav")):
        if not os.path.exists(os.path.expanduser(p)):
            raise FileNotFoundError(f"{label} not found: {p}")

    chunks = chunk_script(text)
    seeds = seeds_for(len(chunks), seed_base, offset)
    work = Path(workdir or os.path.dirname(os.path.abspath(out)) or ".")
    work.mkdir(parents=True, exist_ok=True)
    stem = Path(out).stem

    pieces: list[str] = []
    for i, (chunk, seed) in enumerate(zip(chunks, seeds)):
        raw = str(work / f"{stem}_raw_{i:02d}.wav")
        cmd = [os.path.expanduser(cli), os.path.expanduser(model),
               "--text", chunk, "--instruction", instruction,
               "--ref-audio", os.path.expanduser(ref_wav), "--ref-text", ref_text,
               "--seed", str(seed), "--output", raw]
        if cpu:
            cmd.append("--cpu")
        _sh(cmd)
        trimmed = str(work / f"{stem}_trim_{i:02d}.wav")
        _sh(["ffmpeg", "-y", "-v", "error", "-i", raw, "-af", TRIM_FILTER, trimmed])
        pieces.append(trimmed)

    lst = str(work / f"{stem}.txt")
    Path(lst).write_text("\n".join(f"file '{p}'" for p in pieces))
    cat = str(work / f"{stem}_cat.wav")
    _sh(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", lst, cat])
    _sh(["ffmpeg", "-y", "-v", "error", "-i", cat, "-af", MASTER_FILTER, "-ar", "48000", out])

    for p in [*pieces, lst, cat]:
        try:
            os.remove(p)
        except OSError:
            pass
    return {"file": os.path.abspath(out), "duration_s": round(_dur(out), 3), "chunks": len(chunks)}
