### openpi

This directory contains the docker configuration files to run [openpi from Physical Intelligence](https://github.com/Physical-Intelligence/openpi).

---

### Openpi from Physical-Intelligence

#### Build and run the Docker Image

```sh
ryzers build openpi
ryzers run
```

#### Run other models

For example, to download and convert pi0.5 use the following commands:

```sh
cd /ryzers/openpi

# Download
uv run scripts/serve_policy.py --download-only policy:checkpoint --policy.config=pi05_droid --policy.dir=gs://openpi-assets/checkpoints/pi05_droid

# Convert JAX model to PyTorch
uv run examples/convert_jax_model_to_pytorch.py \
    --checkpoint_dir /root/.cache/openpi/openpi-assets/checkpoints/pi05_droid \
    --config_name pi05_droid \
    --output_path /root/.cache/openpi/openpi-assets/checkpoints/pi05_droid
```

---

### Openpi from LeRobot

#### Build and run the Docker Image

```sh
sudo chmod 666 /dev/ttyACM*
sudo chmod 666 /dev/video*

ryzers build lerobot openpi

HF_TOKEN=<token> ryzers run <args>

"""e.g. for inference only"""
HF_TOKEN=<token> ryzers run
"""e.g. for inference on lerobot so101"""
HF_TOKEN=<token> ryzers run "python /ryzers/lerobot_so101_openpi.py"
```

Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
