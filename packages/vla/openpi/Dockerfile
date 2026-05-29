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
RUN if [ ! -d lerobot ]; then git clone https://github.com/huggingface/lerobot.git \
     && cd lerobot \
     && git checkout 1396b9fa \
     && sed -i '/"torch[>=<~ ]*/d' pyproject.toml \
     && sed -i '/"torchvision[>=<~ ]*/d' pyproject.toml \
     && pip install -e ".[all]"; \
    fi

RUN if [ ! -d lerobot ]; then pip install "numpy<2" pytest huggingface_hub num2words ninja transformers; fi

# Setup flash-attention with ROCm Triton-AMD backend
ENV FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE \
    FLASH_ATTENTION_SKIP_CK_BUILD=FALSE \
    TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 \
    GPU_ARCHS=gfx1151
RUN git clone https://github.com/Dao-AILab/flash-attention.git \
    && cd flash-attention \
    && python setup.py install


ENV HF_HOME=/opt/hf-cache \
    HF_LEROBOT_HOME=/opt/hf-cache/lerobot
RUN mkdir -p "$HF_HOME" "$HF_LEROBOT_HOME" && chmod -R 777 /opt/hf-cache


COPY test_inference.py /ryzers/test_inference.py
COPY test_inference_with_so101.py /ryzers/test_inference_with_so101.py

CMD ["python3", "/ryzers/test_inference.py"]