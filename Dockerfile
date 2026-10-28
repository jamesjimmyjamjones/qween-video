# Wan 2.1 I2V-14B-720P + NSFW umT5 text encoder, as a Serverless Container.
# Twin of the Qwen image stack (same harness, fresh inference core):
# same torch base, same async-job server shape, same sealed-envelope
# crypto (identical params, so the app reuses its seal code untouched),
# same CORS/health patterns. Only the model + encoder + mp4 output differ.
#
# Base is torch 2.9.1 + CUDA 12.8 (proven on RTX PRO 6000 / Blackwell
# sm_100; deliberately not CUDA 13 — see image Dockerfile for rationale).
FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    HF_HUB_CACHE=/data/hf \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && \
    apt-get install -y --no-install-recommends git curl && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip
# brotli==1.2.0 is REQUIRED, not optional (see image Dockerfile: the old
# conda-bundled brotli crashes huggingface_hub's downloader and the
# replica crash-loops forever).
# imageio-ffmpeg ships a static ffmpeg binary on PyPI — no apt ffmpeg needed.
RUN pip install git+https://github.com/huggingface/diffusers transformers accelerate safetensors Pillow fastapi uvicorn pydantic hf_transfer huggingface_hub brotli==1.2.0 peft cryptography imageio imageio-ffmpeg

COPY server.py /app/server.py
WORKDIR /app
EXPOSE 8000

# First boot pulls ~70GB (base + NSFW encoder) into /data; later boots
# reuse. /health always answers 200 (loading/ready in the body), so the
# platform never kills a slow boot.
HEALTHCHECK --interval=30s --timeout=10s --start-period=20m --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

CMD ["python3", "-m", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000", "--timeout-graceful-shutdown", "25"]
