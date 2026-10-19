# Attribution

localswap is original code (MIT). It builds on, calls, or is informed by these
upstream projects. **No upstream source code or model weights are
redistributed in this repository** — scripts fetch them from upstream at build
time.

| Project | Upstream | License | Contribution |
|---|---|---|---|
| Breeze-TTS-2.cpp | https://github.com/HoppouAI/Breeze-TTS-2.cpp | see upstream | Voice-cloning TTS engine; `scripts/build-breeze.sh` builds it, `tts_local.py` wraps its CLI (flag surface verified 2026-10-18) |
| OpenVINO / OpenVINO GenAI | https://github.com/openvinotoolkit/openvino.genai | Apache-2.0 | Image generation backend (`image_local.py`) |
| LCM DreamShaper (int8) | Hugging Face model card (Lykon/dreamshaper-8, LCM LoRA variant) | see model card | Verified few-step image model; referenced, not bundled |
| faster-whisper | https://github.com/SYSTRAN/faster-whisper | MIT | Transcription backend adapter |
| Vulkan-Headers | https://github.com/KhronosGroup/Vulkan-Headers | Apache-2.0/MIT | Vendored at build time only when system headers are absent |

Operational knowledge (chunk sizing, seed scheme, edge-trim, mastering chain)
was derived from real production use on an Intel Lunar Lake laptop, 2026-10.
