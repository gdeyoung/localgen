# The media swap workflow

This is the operator's manual for the local media-generation infrastructure:
how a device goes from bare OS to generating finished images, video, and
voice locally, and how work routes between the laptop lane and bigger
hardware. The library reference is the README; this document is the
**workflow** — stand-up, daily use, routing, and recovery.

Design contract in one line: **probe → plan → generate (→ finish)**. Nothing
is promised without the device proving it (`doctor`/`plan`), and nothing
ships unfinished when quality matters (`finish`).

## 1. Architecture: two lanes, one library

| Lane | Hardware | Role | Typical use |
|---|---|---|---|
| **Laptop lane** | Intel iGPU (OpenVINO GPU) + Vulkan (ncnn) + CPU fallback | drafts, B-roll, images, voice, transcription | zero marginal cost, offline, iterate freely |
| **Discrete lane** | CUDA (fleet) | long-form / hi-res video finals | ComfyUI, Wan / LTX-2 class models |

Both lanes run the same pattern: **generate at the model's native
resolution → interpolate → upscale.** Generating big directly costs ~4× the
compute and looks *worse* (models compose best at trained resolution);
enhancement backfills smoothness and detail for a fraction of the cost.
On the laptop this whole pattern is one call:

```python
from localgen.video_local import generate_video
r = generate_video("prompt", "clip.mp4", finish="smooth2x")   # gen + RIFE + 2x upscale
```

`registry.plan()` marks `video` as `"tier": "heavy"` with measured cost
notes so an agent (or human) can decide laptop-vs-CUDA per clip.

## 2. Stand-up on a new device (Arch/Omarchy reference)

```bash
# 1. GPU compute drivers (VAAPI decode ≠ compute; this is the missing piece)
sudo pacman -S intel-compute-runtime level-zero-loader
python3 -c "import openvino as ov; print(ov.Core().available_devices)"  # expect GPU

# 2. ffmpeg + a venv with the runtime deps
sudo pacman -S ffmpeg
python3 -m venv ~/venvs/media && ~/venvs/media/bin/pip install \
  "localgen[image,video]"   # openvino + openvino-genai + numpy

# 3. Models are BYO (nothing ships in the repo):
~/venvs/media/bin/python -m pip install 'optimum-intel==2.2.0' 'nncf>=3.4' \
  'transformers==4.57.6' 'diffusers==0.39.0' 'huggingface-hub<1.22' \
  tiktoken sentencepiece protobuf
bash scripts/export-ltx.sh          # LTX-Video int8 -> ~/.localgen (≈6.7 GB)

# 4. Enhancement binaries (BYO, ~50 MB each):
#    realesrgan-ncnn-vulkan  <- github.com/xinntao/Real-ESRGAN v0.2.5.0 ubuntu zip
#    rife-ncnn-vulkan        <- github.com/nihui/rife-ncnn-vulkan 20221029 ubuntu zip
#    (realesrgan zip extracts FLAT into ~/.localgen/tools/; rife keeps its dir)

# 5. Verify — never promise without this
localgen doctor && localgen plan   # every capability green or gap-named
```

Notes baked into the tooling (do not relearn them):
- `scripts/export-ltx.sh` wraps the **pinned export toolchain** and the
  **Python 3.14 shim driver** (`scripts/optimum_py314_driver.py`) — CPython
  3.14 made `functools.partial` a descriptor, which breaks optimum's config
  classes. Run the export through the script, not raw optimum-cli.
- TTS (Breeze) has its own one-time build: `scripts/build-breeze.sh`.

## 3. Daily workflow

1. **Probe**: `localgen doctor` (accelerators) / `localgen plan`
   (capabilities + exact gaps). Per-environment truth — a venv without
   openvino reports no video backend even on a capable machine.
2. **Generate**:
   - images: `localgen image "prompt" out.png` (LCM 4-step, seeded)
   - video: `localgen video "prompt" clip.mp4` — raw output
   - video, finished: `finish="smooth2x"` → 2× fps + 2× upscale
   - voice: `localgen tts` with YOUR reference wav + verbatim transcript
   - transcribe: faster-whisper adapter
3. **Everything is seeded.** Same prompt + same seed → same bytes. Reruns are
   free; A/B is one seed apart.
4. **Route by tier**: seconds-long clips and drafts → laptop. Anything you'd
   put in front of a customer → discrete lane with bigger models. The
   laptop's honest ceiling is ~23 s of compute per second of 768×448 video.

## 4. Measured (Core Ultra 258V, LTX-Video int8, 2026-10)

| Stage | Cost |
|---|---|
| Image 768×432 @ 4 steps | ~21 s (CPU) |
| Video 768×448@25, 57 f, 20 steps (GPU) | 52–59 s (~23 s per s of video) |
| Same on CPU | ~12× slower |
| RIFE 2× fps (rife-v4.6), 57→114 f | 3.5 s |
| Real-ESRGAN 2× (animevideov3) | ~0.33 s/frame (16–18 s per 57 f) |
| **One-call finished clip** (768×448@25 → 1536×896@50) | **~101 s** |

Enhancement order matters: **interpolate before upscale** — interpolation
cost scales with pixels, upscale with frames; the cheap order is ~2× less
work than the reverse.

## 5. Recovery & backups posture (sammypro5 reference install)

- **The repo is the recipe.** Everything expensive is either in git (code,
  pins, this document) or reproducible from it (`export-ltx.sh`,
  `build-breeze.sh`, the two ncnn zip URLs). A fresh device stands up from
  §2 with zero private data.
- **Whole-home time-machine** (rsync snapshots, 09:20/21:20) covers
  `~/.localgen` — model + tools ride along at hardlink cost (7.2 GB once,
  ~0 after). The HuggingFace cache (27 GB) is deliberately excluded:
  re-downloadable bulk.
- No secrets, keys, or personal voice references live in this repo — BYO
  inputs stay on-device (scrubbed before every public push).

## 6. Provenance

Upstream projects, licenses, and what each contributes: **ATTRIBUTION.md**.
Verified flag surfaces and versions are cited there; nothing is bundled.
