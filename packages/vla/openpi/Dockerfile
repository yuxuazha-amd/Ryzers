# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

WORKDIR /ryzers

RUN if python3 -c "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('lerobot') else 1)"; then \
        echo "lerobot found; skipping Physical-Intelligence openpi install"; \
    else \
        apt-get update && apt-get install -y \
        cmake ca-certificates wget patchelf git vim htop \
        libgl1-mesa-dev libosmesa6-dev libglfw3 libglew-dev libxrandr-dev libxinerama-dev libxcursor-dev libxi-dev \
        libavdevice-dev libavfilter-dev \
        && rm -rf /var/lib/apt/lists/* \
        && pip install uv chex \
        && git clone https://github.com/Physical-Intelligence/openpi.git \
        && cd openpi \
        && git checkout 95aadc6b16d170e4b13ab2e0ac64fbf2d1bb8e31 \
        && git submodule update --init --recursive \
        && sed -i '/"lerobot"/a\    "av==14.2.0",' pyproject.toml \
        && sed -i '/"torch[>=<~ ]*/d' pyproject.toml \
        && GIT_LFS_SKIP_SMUDGE=1 uv pip install --python /opt/venv/bin/python --break-system-packages -e .; \
    fi

ENV HF_HOME=/opt/hf-cache \
    HF_LEROBOT_HOME=/opt/hf-cache/lerobot
RUN mkdir -p "$HF_HOME" "$HF_LEROBOT_HOME" && chmod -R 777 /opt/hf-cache

COPY lerobot_openpi.py /ryzers/lerobot_openpi.py
COPY lerobot_so101_openpi.py /ryzers/lerobot_so101_openpi.py
COPY pi_openpi.py /ryzers/pi_openpi.py

CMD ["bash", "-lc", "\
    if python3 -c \"import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('openpi') else 1)\"; then \
        exec python3 /ryzers/pi_openpi.py; \
    else \
        exec python3 /ryzers/lerobot_openpi.py; \
    fi"]