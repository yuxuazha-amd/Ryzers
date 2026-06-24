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

# Setup LeRobot
RUN git clone https://github.com/huggingface/lerobot.git \
    && cd lerobot \
    && git checkout v0.5.1 \
    && sed -i '/"torch[>=<~ ]*/d' pyproject.toml \
    && sed -i '/"torchvision[>=<~ ]*/d' pyproject.toml \
    && pip install -e ".[all]"

RUN pip install "numpy<2" pytest huggingface_hub num2words ninja

COPY test.sh /ryzers/test_lerobot.sh
CMD ["bash", "/ryzers/test_lerobot.sh"]