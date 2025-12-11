ARG BASE_IMAGE=nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV LANG=C.UTF-8

ARG INDEX_TTS_VERSION=1.5
ARG MODEL_ID=""
ARG HF_TOKEN=""
ARG TORCH_WHEEL_URL=""         # Optional: direct wheel URL for torch matching desired CUDA (recommended)
ARG TORCH_EXTRA_INDEX_URL=""   # Optional extra index (if using pip --extra-index-url)

RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget ca-certificates build-essential ffmpeg curl unzip \
    python3 python3-dev python3-venv python3-pip && \
    rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip

WORKDIR /app
COPY . /app

# Install runtime dependencies (except torch/torchaudio by default)
RUN pip install --no-cache-dir runpod huggingface-hub safetensors soundfile numpy==1.26.0 librosa ffmpeg-python

# Install the index-tts package in editable mode (expects repository files to be in /app)
RUN pip install --no-cache-dir -e .

# Optional: install a specific torch wheel (pass TORCH_WHEEL_URL at build time).
# Example build usage (replace URL with the correct wheel for your CUDA/torch version):
#   docker build --build-arg TORCH_WHEEL_URL="https://download.pytorch.org/whl/cu128/torch-2.2.0+cu128-cp310-cp310-linux_x86_64.whl" ...
RUN if [ -n "$TORCH_WHEEL_URL" ] ; then \
      echo "[build] installing torch from wheel URL"; \
      pip install --no-cache-dir "$TORCH_WHEEL_URL" ${TORCH_EXTRA_INDEX_URL:+--extra-index-url $TORCH_EXTRA_INDEX_URL}; \
    else \
      echo "[build] TORCH_WHEEL_URL empty; skipping torch install. You MUST install a torch+cu wheel compatible with your target GPU/CUDA at runtime or build time."; \
    fi

# Optional: bake model snapshot during image build (pass MODEL_ID and HF_TOKEN as build-args).
RUN if [ -n "$MODEL_ID" ] ; then \
      python3 - <<PY
from huggingface_hub import snapshot_download
import os
mid = os.environ.get("MODEL_ID")
token = os.environ.get("HF_TOKEN") or None
print("[build] downloading model snapshot:", mid)
snapshot_download(repo_id=mid, local_dir="checkpoints", token=token)
print("[build] model downloaded to checkpoints/")
PY
    else \
      echo "[build] MODEL_ID empty; no model will be baked into image."; \
    fi

EXPOSE 8080
CMD ["python3", "runpod_serverless.py"]
