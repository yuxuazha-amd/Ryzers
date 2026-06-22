# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

WORKDIR /ryzers

ENV HF_HOME=/opt/hf-cache \
    HF_LEROBOT_HOME=/opt/hf-cache/lerobot
RUN mkdir -p "$HF_HOME" "$HF_LEROBOT_HOME" && chmod -R 777 /opt/hf-cache

COPY test_inference.py /ryzers/test_inference.py
COPY test_inference_with_so101.py /ryzers/test_inference_with_so101.py

CMD ["python3", "/ryzers/test_inference.py"]