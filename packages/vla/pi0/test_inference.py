from huggingface_hub import snapshot_download
import os
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors
# Swap this import per-policy
from lerobot.policies.pi0 import PI0Policy


tok = os.environ.get("HF_TOKEN")
if not tok:
    raise RuntimeError("Error: HF_TOKEN is required, please provide an HF_TOKEN with permission to download PaliGemma. Set it with: HF_TOKEN=<token> ryzers run <args>")

# Pin pi0_base to the last revision compatible with lerobot v0.5.1
PI0_REV = "26b99b9439acb1e352439e34ee9c67af0d76efa3"
local_pi0 = snapshot_download("lerobot/pi0_base", repo_type="model", token=tok, revision=PI0_REV)

# load a policy
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

policy = PI0Policy.from_pretrained(
    local_pi0, 
    cli_overrides=["--dtype", "bfloat16"], 
).to(device).eval()

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
dataset = LeRobotDataset("lerobot/libero")

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
    print(f"Predicted action: {pred_action}")