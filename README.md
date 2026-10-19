# localgen

Swap cloud media-generation services for **whatever compute your device already has**.

An installable Python library — not a how-to guide. Point it at a machine, it
probes the accelerators (Intel iGPU/GPU via OpenVINO, Vulkan, CUDA, CPU) and
gives you local image generation, cloned-voice narration, and transcription
through whatever backend that device can run. Built so AI agents (Hermes,
Claude Code, any coding agent) can generate media locally with one import.

```python
from localgen import registry
registry.plan()          # what THIS device can do, and exactly what's missing

from localgen.image_local import generate
generate("a red bicycle on mars", "out.png")          # OpenVINO, auto device

from localgen.video_local import generate_video      # LTX-Video, iGPU lane
generate_video("a red bicycle rolling across mars", "clip.mp4",
               width=768, height=448, num_frames=61)  # ~2.4 s clip, seeded
generate_video("...", "clip.mp4", finish="smooth2x")  # + RIFE 2x fps + 2x upscale

from localgen.tts_local import narrate               # YOUR reference voice
narrate(script, "vo.wav", ref_wav="me.wav", ref_text="verbatim transcript of me.wav")
```

CLI: `localgen doctor` · `localgen plan` · `localgen image "prompt" out.png` · `localgen video "prompt" clip.mp4` · `localgen tts ...`

## Scope boundary (read this first)

**Verified on:** Intel Core Ultra (Lunar Lake) iGPU + CPU, Linux, ffmpeg 9,
openvino-genai 2026.x, Breeze-TTS-2.cpp current release. Image and **video**
generation are **Intel-only** (OpenVINO). On NVIDIA, image/video generation is
not wired — use CUDA-native stacks (diffusers + torch cuda, ComfyUI);
transcription does use CUDA via faster-whisper when present. On Apple Silicon,
none of this is verified — MLX/whisper.cpp are the native options there.

**Video is the heavy tier.** Measured on a Core Ultra (Lunar Lake) iGPU with
LTX-Video int8: **~23 s of compute per second of 768×448 video** at library
defaults (61-frame request → 57 frames; LTX snaps frame counts to 8n+1).
CPU fallback is ~12× slower. Good for drafts and B-roll; long-form and
high-resolution video belong on discrete CUDA hardware. `registry.plan()`
reports `video` with `"tier": "heavy"` so an agent can route accordingly.

**The finish pipeline (generate → interpolate → upscale).** Neural upscale
(Real-ESRGAN ncnn, Vulkan) and frame interpolation (RIFE ncnn, Vulkan) are
built in — `finish="smooth2x"` on `generate_video`, or standalone
`enhance_local.upscale()` / `.interpolate()` / CLI `localgen upscale|interp`.
Measured on the same iGPU: 2× fps ≈ 3.5 s per 57-frame clip, 2× upscale ≈ 0.33 s
per frame. One call turns the raw 768×448@25 clip into 1536×896@50 in ~101 s
total. Binaries are BYO from their GitHub releases into `$LOCALGEN_HOME/tools`
(see `enhance_local.py` docstring); no weights bundled.

No model weights are bundled or downloaded by this library. You fetch models
from their upstream releases (see ATTRIBUTION.md for the projects); localgen
finds them under `$LOCALGEN_HOME` (default `~/.localgen`) or env overrides.

## Install

```bash
pip install localgen            # or: pip install -e . from a checkout
pip install "localgen[image]"   # + OpenVINO image generation
localgen doctor                 # what this machine has
localgen plan                   # gaps named, as a checklist
```

Video (LTX-Video int8, ~6 GB export): `pip install "localgen[video]"` then
`bash scripts/export-ltx.sh` (converts from Hugging Face; optimum-intel +
nncf installed on demand).

TTS needs the Breeze CLI built once: `bash scripts/build-breeze.sh` (fetches
upstream source tarballs, vendored Vulkan headers only if your system lacks
them), plus a Breeze gguf model in the same directory. Then any reference
voice: ~10s clean WAV + its exact transcript.

## How it adapts

`doctor.probe()` is the contract everything reads: OpenVINO device list,
Vulkan presence, CUDA presence, ffmpeg. Backends are chosen from facts, not
config: iGPU found → `openvino-gpu`; only CPU → `openvino-cpu`; Vulkan →
Breeze on GPU, else CPU; CUDA → faster-whisper on GPU, else CPU.

The narration pipeline is deterministic by construction: fixed seed per chunk
(`base + offset*10 + index`), sentence-boundary chunking sized for a 9-second
conditioning window, bilateral edge-silence trim, concat, then a broadcast
mastering chain (de-esser, HF shelf, EBU R128 loudness at −18 LUFS). Same
script + same seeds → same audio.

## Why "swap"

Cloud media APIs are great until the budget, the rate limit, or the privacy
posture matters. This library is the drop-in local lane: same call shape an
agent already uses, none of the network. It pairs cleanly with HTML-to-video
engines (HyperFrames, OpenFilm) — generate assets locally, compose anywhere.

## License

MIT (see LICENSE). Attribution for upstream projects: ATTRIBUTION.md.
Operator's manual (stand-up, workflow, routing, recovery): **docs/WORKFLOW.md**.
Music generation doctrine (fleet kit distillate): **docs/music-kit.md**.
