FROM ghcr.io/pytorch/pytorch-nightly:2.4.0.dev20240513-cuda12.4-cudnn8-devel

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# 🔒 CRITICAL SAFETY STEP
# Remove any CPU torch that pip may pull
RUN pip uninstall -y torch torchvision torchaudio || true

# Python ML deps (NO torch here)
RUN pip install --no-cache-dir \
    diffusers \
    transformers \
    accelerate \
    safetensors \
    timm \
    einops \
    moviepy==1.0.3 \
    imageio \
    imageio-ffmpeg \
    opencv-python \
    Pillow \
    tqdm

# HuggingFace cache
ENV HF_HOME=/models/hf
ENV TRANSFORMERS_CACHE=/models/hf
ENV DIFFUSERS_CACHE=/models/hf

COPY . /app
RUN chmod -R 755 /app

CMD ["bash", "-lc", "python3 generate_gpu_job.py --job-id ${JOB_ID}"]
