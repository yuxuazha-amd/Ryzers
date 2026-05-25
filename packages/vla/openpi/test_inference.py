#!/usr/bin/env python3
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.pi0 import PI0Policy

# load a policy
model_id = "lerobot/pi0_base"  # <- swap checkpoint
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

policy = PI0Policy.from_pretrained(model_id).to(device).eval()

preprocess, postprocess = make_pre_post_processors(
    policy.config,
    model_id,
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

# pick an episode
episode_index = 0

# each episode corresponds to a contiguous range of frame indices
from_idx = dataset.meta.episodes["dataset_from_index"][episode_index]
to_idx   = dataset.meta.episodes["dataset_to_index"][episode_index]

# get a single frame from that episode (e.g. the first frame)
frame_index = from_idx
frame = dict(dataset[frame_index])

batch = preprocess(frame)
with torch.inference_mode():
    pred_action = policy.select_action(batch)
    # use your policy postprocess, this post process the action
    # for instance unnormalize the actions, detokenize it etc..
    pred_action = postprocess(pred_action)
    print("Predicted action:", pred_action)