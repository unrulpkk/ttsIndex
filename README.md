# IndexTTS → RunPod Serverless (template)

Files included:
- Dockerfile
- runpod_serverless.py
- download_models.sh
- requirements.txt

## Quick notes / assumptions
- This template installs most Python dependencies but **does not** force-install a specific torch wheel by default.
  You MUST install a torch+CUDA wheel that matches the GPU/CUDA on RunPod. The Dockerfile accepts a build-arg `TORCH_WHEEL_URL`.
- You can bake the HF model during image build by passing `MODEL_ID` as a build-arg, or use RunPod Cached Models and set `USE_CACHED_MODEL=1`.
- Select the right GPU on RunPod (A10 / A100 / H100) based on desired throughput and VRAM. IndexTTS v2 typically needs more VRAM than v1.

## Example docker build (with torch wheel)
# replace <URL> with the proper wheel matching your target CUDA and python version
docker build \
  --build-arg INDEX_TTS_VERSION=1.5 \
  --build-arg MODEL_ID=IndexTeam/IndexTTS-1.5 \
  --build-arg HF_TOKEN=${HF_TOKEN:-""} \
  --build-arg TORCH_WHEEL_URL="<TORCH_WHEEL_URL>" \
  -t indextts-runpod:1.5 .

## Example RunPod deploy (push to registry then import on RunPod)
1. push image to Docker Hub / GHCR
2. Create Serverless Endpoint on RunPod, choose `Import from Docker Registry` and set environment variables:
   - INDEX_TTS_VERSION=1.5
   - MODEL_ID=IndexTeam/IndexTTS-1.5
   - USE_CACHED_MODEL=0 (or 1 if using RunPod cached models)
   - HF_TOKEN=<your HF token if needed>
3. Set health-check to call the endpoint root or use custom probe.

## Example request (curl)
curl -X POST "https://<your-runpod-endpoint-url>" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello from IndexTTS on RunPod!"
  }'

## Important caveats (read before deploying)
1. Torch/CUDA mismatch is the most common runtime failure. Provide an explicit TORCH_WHEEL_URL that matches your RunPod GPU's CUDA version.
2. IndexTTS repo APIs (class names, infer signatures) may change — if you see import errors, open a shell in the container and inspect `indextts` package to adapt constructors/args.
3. I cannot run the container build or test the runtime environment for you from here. This repository template aims to minimize friction, but you will need to build the image on your environment or RunPod CI and ensure the torch wheel and GPU selection match.

If you want, I can also:
- include pre-filled wheel URL examples for common CUDA releases (cu118/cu128) in Dockerfile,
- or produce two ready-to-build Docker commands for specific RunPod GPU choices (tell me which GPU/CUDA you will pick).
