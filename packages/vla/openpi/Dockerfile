ARG BASE_IMAGE
FROM ${BASE_IMAGE}

WORKDIR /ryzers
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN if python3 -c "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('lerobot') else 1)"; then \
        echo "lerobot" > /ryzers/openpi_mode; \
        echo "Detected LeRobot base image; using LeRobot PI0 demo"; \
    else \
        echo "pi" > /ryzers/openpi_mode; \
        echo "No LeRobot base image; installing upstream OpenPI demo"; \
    fi

ENV HF_HOME=/opt/hf-cache \
    HF_LEROBOT_HOME=/opt/hf-cache/lerobot

RUN mkdir -p "$HF_HOME" "$HF_LEROBOT_HOME" && chmod -R 777 /opt/hf-cache

COPY test_inference.py /ryzers/test_inference.py
COPY test_inference_with_so101.py /ryzers/test_inference_with_so101.py

COPY openpi.patch /ryzers/openpi-assets/openpi.patch
COPY test.py /ryzers/openpi-assets/test.py
COPY test.sh /ryzers/openpi-assets/test.sh

RUN if [ "$(cat /ryzers/openpi_mode)" = "pi" ]; then \
        apt-get update && apt-get install -y \
            git vim wget htop cmake ca-certificates patchelf \
            libgl1-mesa-dev libosmesa6-dev libglfw3 libglew-dev \
            libxrandr-dev libxinerama-dev libxcursor-dev libxi-dev \
        && rm -rf /var/lib/apt/lists/*; \
    fi

ARG MUJOCO_VERSION=2.3.7
ENV MUJOCO_PATH=/root/.mujoco/mujoco-${MUJOCO_VERSION}
ENV MUJOCO_PLUGIN_PATH=/root/.mujoco/plugins-empty
ENV LD_LIBRARY_PATH=${MUJOCO_PATH}/bin:${LD_LIBRARY_PATH}

RUN if [ "$(cat /ryzers/openpi_mode)" = "pi" ]; then \
        mkdir -p /root/.mujoco \
        && wget -q https://github.com/google-deepmind/mujoco/releases/download/${MUJOCO_VERSION}/mujoco-${MUJOCO_VERSION}-linux-x86_64.tar.gz -O /root/.mujoco/mujoco.tar.gz \
        && tar -xzf /root/.mujoco/mujoco.tar.gz -C /root/.mujoco \
        && rm /root/.mujoco/mujoco.tar.gz \
        && mkdir -p "${MUJOCO_PLUGIN_PATH}" \
        && pip install --no-cache-dir --break-system-packages mujoco==${MUJOCO_VERSION}; \
    fi

RUN if [ "$(cat /ryzers/openpi_mode)" = "pi" ]; then \
        pip install git+https://github.com/huggingface/lerobot.git@0cf864870cf29f4738d3ade893e6fd13fbd7cdb5#0cf864870cf29f4738d3ade893e6fd13fbd7cdb5 \
        && pip install jax[rocm] \
        && pip install uv; \
    fi

RUN if [ "$(cat /ryzers/openpi_mode)" = "pi" ]; then \
        git clone https://github.com/Physical-Intelligence/openpi /ryzers/openpi \
        && cd /ryzers/openpi \
        && git checkout 95aadc6b16d170e4b13ab2e0ac64fbf2d1bb8e31 \
        && git submodule update --init --recursive \
        && cp /ryzers/openpi-assets/openpi.patch ./openpi.patch \
        && git apply --whitespace=fix ./openpi.patch || true \
        && sed -i '/^override-dependencies = \[/ s/\]/, "av==13.1.0"]/' pyproject.toml \
        && GIT_LFS_SKIP_SMUDGE=1 uv sync \
        && GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .; \
    fi

RUN if [ "$(cat /ryzers/openpi_mode)" = "pi" ]; then \
        cd /ryzers/openpi \
    && uv run python -c "from openpi.shared import download; download.maybe_download('gs://openpi-assets/checkpoints/pi0_droid')" \
        && cp -r ./src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/ \
        && uv run examples/convert_jax_model_to_pytorch.py \
            --checkpoint_dir /root/.cache/openpi/openpi-assets/checkpoints/pi0_droid \
            --config_name pi0_droid \
            --output_path /root/.cache/openpi/openpi-assets/checkpoints/pi0_droid \
        && cp /ryzers/openpi-assets/test.py /ryzers/openpi/test.py \
        && cp /ryzers/openpi-assets/test.sh /ryzers/test.sh \
        && chmod +x /ryzers/test.sh; \
    fi

RUN echo "export LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm-6.4.4/lib:\$LD_LIBRARY_PATH" >> /root/.bashrc

CMD ["/bin/bash", "-lc", "if [ \"$(cat /ryzers/openpi_mode)\" = \"lerobot\" ]; then python3 /ryzers/test_inference.py; else /ryzers/test.sh; fi"]