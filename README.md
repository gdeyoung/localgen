# localswap

Swap cloud media-generation services for **whatever compute your device already has**.

An installable Python library — not a how-to guide. Point it at a machine, it
probes the accelerators (Intel iGPU/GPU via OpenVINO, Vulkan, CUDA, CPU) and
gives you local image generation, cloned-voice narration, and transcription
through whatever backend that device can run. Built so AI agents (Hermes,
Claude Code, any coding agent) can generate media locally with one import.

```python
from localswap import registry
registry.plan()          # what THIS device can do, and exactly what's missing

from localswap.image_local import generate
generate("a red bicycle on mars", "out.png")          # OpenVINO, auto device

from localswap.tts_local import narrate               # YOUR reference voice
narrate(script, "vo.wav", ref_wav="me.wav", ref_text="verbatim transcript of me.wav")
```

CLI: `localswap doctor` · `localswap plan` · `localswap image "prompt" out.png` · `localswap tts ...`

## Scope boundary (read this first)

**Verified on:** Intel Core Ultra (Lunar Lake) iGPU + CPU, Linux, ffmpeg 9,
openvino-genai 2026.x, Breeze-TTS-2.cpp current release. Image generation is
**Intel-only** (OpenVINO). On NVIDIA, image generation is not wired — use
CUDA-native stacks (diffusers + torch cuda, ComfyUI); transcription does use
CUDA via faster-whisper when present. On Apple Silicon, none of this is
verified — MLX/whisper.cpp are the native options there.

No model weights are bundled or downloaded by this library. You fetch models
from their upstream releases (see ATTRIBUTION.md for the projects); localswap
finds them under `$LOCALSWAP_HOME` (default `~/.localswap`) or env overrides.

## Install

```bash
pip install localswap            # or: pip install -e . from a checkout
pip install "localswap[image]"   # + OpenVINO image generation
localswap doctor                 # what this machine has
localswap plan                   # gaps named, as a checklist
```

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
