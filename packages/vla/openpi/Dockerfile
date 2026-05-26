# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

WORKDIR /ryzers

# Install deps
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    ffmpeg \
    git \
    libavcodec-dev \
    libavdevice-dev \
    libavfilter-dev \
    libavformat-dev \
    libavutil-dev \
    libswresample-dev \
    libswscale-dev \
    pkg-config \
    python3-dev \
    speech-dispatcher \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Setup LeRobot (skip if already provided by a base image)
RUN if [ ! -d lerobot ]; then \
        git clone https://github.com/huggingface/lerobot.git \
     && cd lerobot \
     && git checkout 1396b9fa \
     && sed -i '/"torch[>=<~ ]*/d' pyproject.toml \
     && sed -i '/"torchvision[>=<~ ]*/d' pyproject.toml \
     && pip install -e ".[all]"; \
    fi

RUN pip install "numpy<2" pytest huggingface_hub num2words ninja transformers

# Setup flash-attention with ROCm Triton-AMD backend
ENV FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE \
    FLASH_ATTENTION_SKIP_CK_BUILD=FALSE \
    TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 \
    GPU_ARCHS=gfx1151
RUN git clone https://github.com/Dao-AILab/flash-attention.git \
    && cd flash-attention \
    && python setup.py install


# Pre-bake HF cache into the image at a path the runtime bind-mount won't shadow.
ENV HF_HOME=/opt/hf-cache \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

ARG HF_TOKEN
RUN test -n "$HF_TOKEN" || { echo "ERROR: HF_TOKEN build arg is empty."; exit 1; } \
 && HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 python - <<PY
import os, shutil
from huggingface_hub import snapshot_download

tok = os.environ["HF_TOKEN"]

# Models (HF Hub cache layout — fine for transformers/from_pretrained)
snapshot_download("lerobot/pi0_base", repo_type="model", token=tok)
snapshot_download(
    "google/paligemma-3b-pt-224", repo_type="model", token=tok,
    allow_patterns=[
        "config.json", "generation_config.json",
        "preprocessor_config.json", "processor_config.json",
        "special_tokens_map.json", "added_tokens.json",
        "tokenizer.json", "tokenizer.model", "tokenizer_config.json",
        "*.txt",
    ],
)

# Dataset: stage at the path LeRobotDataset looks at locally.
lerobot_home = os.environ.get(
    "HF_LEROBOT_HOME",
    os.path.join(os.environ["HF_HOME"], "lerobot"),
)
local_libero = os.path.join(lerobot_home, "lerobot", "libero")
os.makedirs(local_libero, exist_ok=True)
snapshot_download(
    "lerobot/libero",
    repo_type="dataset",
    token=tok,
    local_dir=local_libero,
    allow_patterns=[
        "meta/**",
        "data/chunk-000/file-000.parquet",
        "videos/*/chunk-000/file-000.mp4",
    ]
)

# CRITICAL: remove the local_dir bookkeeping dir; otherwise lerobot's
# has_legacy_hub_download_metadata() returns True and forces a re-fetch.
shutil.rmtree(os.path.join(local_libero, ".cache"), ignore_errors=True)
PY

COPY test_inference.py /ryzers/test_inference.py
COPY test_inference_with_so101.py /ryzers/test_inference_with_so101.py
COPY test_flash_attention.py /ryzers/test_flash_attention.py

CMD ["python3", "/ryzers/test_inference.py"]