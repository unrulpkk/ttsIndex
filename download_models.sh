#!/usr/bin/env bash
set -e
MODEL_ID=${1:-"IndexTeam/IndexTTS-1.5"}
OUTDIR=${2:-"./checkpoints"}
HF_TOKEN=${HF_TOKEN:-""}

python3 - <<PY
from huggingface_hub import snapshot_download
import os
mid = os.environ.get("MODEL_ID", "${MODEL_ID}")
token = os.environ.get("HF_TOKEN", "${HF_TOKEN}") or None
snapshot_download(repo_id=mid, local_dir="${OUTDIR}", token=token)
print("model downloaded to ${OUTDIR}")
PY
