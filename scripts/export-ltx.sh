#!/usr/bin/env bash
# Export LTX-Video to OpenVINO int8 — the iGPU-viable video model (~6 GB out).
# Runtime deps (openvino, openvino-genai) are NOT installed here; this script
# only produces the model directory that localswap.video_local consumes.
#
# Usage: bash scripts/export-ltx.sh [MODEL_ID] [OUT_DIR]
#   MODEL_ID  default Lightricks/LTX-Video (2B, fast; no audio)
#             LTX-2 family (e.g. diffusers/LTX-2.3-Diffusers) also exports
#             with --task text-to-video and adds a synchronized audio track,
#             but needs far more memory than an iGPU targets.
#   OUT_DIR   default $LOCALSWAP_HOME/models/video/ltx-video-int8
set -euo pipefail

MODEL_ID="${1:-Lightricks/LTX-Video}"
OUT="${2:-${LOCALSWAP_HOME:-$HOME/.localswap}/models/video/ltx-video-int8}"
PY="${LOCALSWAP_PY:-python3}"

if ! "$PY" -c 'import optimum.intel, nncf' >/dev/null 2>&1; then
  echo "installing export tooling into the interpreter at: $PY" >&2
  # Version sandwich (verified 2026-10): optimum-intel 2.2 is built against
  # transformers 4.5x — on 5.x its NormalizedConfig call crashes ("got multiple
  # values for argument 'allow_new'"). diffusers must stay <=0.39 (0.40+
  # imports Gemma4 classes that only exist in transformers>=5.6).
  "$PY" -m pip install 'optimum-intel==2.2.0' 'nncf>=3.4' \
    'transformers==4.57.6' 'diffusers==0.39.0' 'huggingface-hub<1.22' \
    tiktoken sentencepiece protobuf >&2
fi

CLI="$PY"
DRIVER="$(dirname "$0")/optimum_py314_driver.py"

mkdir -p "$(dirname "$OUT")"
# Python 3.14 made functools.partial a descriptor, breaking optimum's
# with_args class attributes — route through the shim driver always
# (harmless on <=3.13).
exec "$CLI" "$DRIVER" export openvino --model "$MODEL_ID" --task text-to-video \
  --weight-format int8 --trust-remote-code "$OUT"
