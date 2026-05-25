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
    && git checkout 1396b9fa \
    && sed -i '/"torch[>=<~ ]*/d' pyproject.toml \
    && sed -i '/"torchvision[>=<~ ]*/d' pyproject.toml \
    && pip install -e ".[all]"

RUN pip install "numpy<2" pytest huggingface_hub num2words ninja

# # Create an entrypoint to autmatically login to Hugging Face on docker start
# RUN echo -e '#!/bin/bash\n\
# if [ -n "$HF_TOKEN" ]; then\n\
#   hf auth login --token "$HF_TOKEN" --add-to-git-credential\n\
# fi\n\
# export HF_USER=$(hf auth whoami | head -n 1 | cut -d" " -f3)\n\
# echo "🚀 Logged in as $HF_USER"\n\
# exec "$@"' > /ryzers/entrypoint.sh && chmod +x /ryzers/entrypoint.sh

# ENTRYPOINT ["/ryzers/entrypoint.sh"]

COPY test.sh /ryzers/test_lerobot.sh
RUN chmod +x /ryzers/test_lerobot.sh
CMD /ryzers/test_lerobot.sh