#!/usr/bin/env python3
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
from huggingface_hub import snapshot_download
import os
import shutil
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.pi0 import PI0Policy


"""Download policy and dataset from HuggingFace"""
tok = os.environ.get("HF_TOKEN")
if not tok:
    raise RuntimeError("Error: HF_TOKEN is required, please provide an HF_TOKEN with permission to download PaliGemma. Set it with: HF_TOKEN=<token> ryzers run <args>")

# Pin pi0_base to the last revision compatible with lerobot v0.5.1
# (the commit BEFORE "Add relative action processor steps", #6, Jun 3 2026).
PI0_REV = "26b99b9439acb1e352439e34ee9c67af0d76efa3"
local_pi0 = snapshot_download("lerobot/pi0_base", repo_type="model", token=tok, revision=PI0_REV)

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


"""Load policy and dataset"""
# load a policy
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
policy = PI0Policy.from_pretrained(local_pi0).to(device).eval()
preprocess, postprocess = make_pre_post_processors(
    policy.config,
    local_pi0,
    preprocessor_overrides={
        "device_processor": {"device": str(device)},
        "rename_observations_processor": {
            "rename_map": {
                "observation.images.image": "observation.images.base_0_rgb",
                "observation.images.image2": "observation.images.left_wrist_0_rgb",
            }
        },
    },
)
# load a lerobotdataset
dataset = LeRobotDataset("lerobot/libero", episodes=[0])


"""Test inference"""
# pick an episode
episode_index = 0
# each episode corresponds to a contiguous range of frame indices
from_idx = dataset.meta.episodes["dataset_from_index"][episode_index]
# get a single frame from that episode (e.g. the first frame)
frame = dict(dataset[from_idx])

batch = preprocess(frame)
with torch.inference_mode():
    pred_action = policy.select_action(batch)
    # use your policy postprocess, this post process the action
    # for instance unnormalize the actions, detokenize it etc..
    pred_action = postprocess(pred_action)
    print("Predicted action:", pred_action)