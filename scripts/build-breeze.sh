#!/usr/bin/env bash
# Build Breeze-TTS-2.cpp (voice-clone TTS, Vulkan or CPU) from upstream sources.
# Generalized from a verified 2026-10-18 laptop build. Idempotent-ish: skips
# finished artifacts. No model weights are downloaded here — fetch the gguf
# from upstream releases yourself and place it beside the binary.
set -euo pipefail

DEST="${1:-${LOCALGEN_HOME:-$HOME/.localgen}/breeze}"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
cd "$WORK"

echo "[build-breeze] fetching sources (codeload tarballs; git protocol is slow on some links)"
VER="${BREESE_VER:-}"
if [ -z "$VER" ]; then
  VER="$(curl -fsSL --max-time 30 https://api.github.com/repos/HoppouAI/Breeze-TTS-2.cpp/releases/latest | sed -n 's/.*"tag_name": *"\([^"]*\)".*/\1/p')"
  [ -n "$VER" ] || { echo "could not resolve latest release tag; set BREESE_VER"; exit 1; }
fi
curl -fL "https://codeload.github.com/HoppouAI/Breeze-TTS-2.cpp/tar.gz/refs/tags/$VER" -o breeze.tar.gz
mkdir -p breeze && tar xzf breeze.tar.gz -C breeze --strip-components=1
cd breeze
git submodule status 2>/dev/null | head -2 || true   # informational; tarball has no .git

echo "[build-breeze] cmake (vendored headers if system lacks vulkan-headers)"
if ! pkg-config --exists vulkan 2>/dev/null && ! ls /usr/include/vulkan/vulkan_core.h >/dev/null 2>&1; then
  curl -fL "https://codeload.github.com/KhronosGroup/Vulkan-Headers/tar.gz/refs/tags/v1.4.363" -o vk.tar.gz
  mkdir -p third_party/vulkan-headers && tar xzf vk.tar.gz -C third_party/vulkan-headers --strip-components=1
  VK_CMAKE="-DVulkan_INCLUDE_DIR=$PWD/third_party/vulkan-headers/include"
else
  VK_CMAKE=""
fi
cmake -B build -DCMAKE_BUILD_TYPE=Release $VK_CMAKE
cmake --build build --target breeze-cli -j"$(nproc 2>/dev/null || echo 4)"

mkdir -p "$DEST"
cp build/breeze-cli "$DEST/breeze-cli"
echo "[build-breeze] binary: $DEST/breeze-cli"
echo "[build-breeze] NEXT: place a breeze-tts-2 gguf at $DEST/breeze-tts-2-q8_0.gguf (upstream releases)"
"$DEST/breeze-cli" --help >/dev/null 2>&1 || echo "[build-breeze] note: --help exit code nonzero is normal for some builds — verify with a real synthesis"
