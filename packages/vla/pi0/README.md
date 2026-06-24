### openpi

This directory contains the docker configuration files to run [openpi from Physical Intelligence](https://github.com/Physical-Intelligence/openpi).

### Build and run the Docker Image

```sh
sudo chmod 666 /dev/ttyACM*
sudo chmod 666 /dev/video*

ryzers build lerobot pi0

HF_TOKEN=<token> ryzers run <args>

"""e.g. for inference only"""
HF_TOKEN=<token> ryzers run
"""e.g. for inference on lerobot so101"""
HF_TOKEN=<token> ryzers run "python /ryzers/test_inference_with_so101.py"
```

Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
