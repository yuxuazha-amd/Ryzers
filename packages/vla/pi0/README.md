### openpi

This directory contains the docker configuration files to run [openpi from Physical Intelligence](https://github.com/Physical-Intelligence/openpi).

### Build and run the Docker Image

1. To run pure software inference from dataset
```sh
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxx   # token that has accepted PaliGemma license
sudo chmod 666 /dev/ttyACM*
sudo chmod 666 /dev/video*

cd Ryzers/
ryzers build lerobot pi0

"""for inference only"""
ryzers run
"""for inference on lerobot so101"""
ryzers run "python /ryzers/test_inference_with_so101.py"
```

Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
