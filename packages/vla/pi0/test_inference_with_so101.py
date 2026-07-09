from huggingface_hub import snapshot_download
import os
import torch

from lerobot.cameras.configs import Cv2Backends
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.datasets.feature_utils import hw_to_dataset_features
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors
# Swap this import per-policy
from lerobot.policies.pi0 import PI0Policy
from lerobot.policies.utils import build_inference_frame, make_robot_action
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

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

# load lerobot so101
camera_devices = {
    "side": "/dev/video0",
    "wrist": "/dev/video2",
}
cameras = { 
    name: OpenCVCameraConfig(
        index_or_path=port,
        width=320,
        height=240,
        fps=30,
        backend=Cv2Backends.V4L2,
        fourcc="MJPG",
    ) for name, port in camera_devices.items() 
}
robot = SO101Follower(SO101FollowerConfig(
    port="/dev/ttyACM0",
    id="follower_arm",
    cameras=cameras,
))
robot.connect()

# Build dataset features from robot's action and observation features
action_features = hw_to_dataset_features(robot.action_features, "action")
obs_features = hw_to_dataset_features(robot.observation_features, "observation")
dataset_features = {**action_features, **obs_features}

task = ""           # e.g. "pick the red block" — must be set for pi0
robot_type = ""     # e.g. "so100_follower"

observation = robot.get_observation()
obs_frame = build_inference_frame(
    observation=observation,
    ds_features=dataset_features,
    device=device,
    task=task,
    robot_type=robot_type,
)

batch = preprocess(obs_frame)

with torch.inference_mode():
    pred_action = policy.select_action(batch)
    # use your policy postprocess, this post process the action
    # for instance unnormalize the actions, detokenize it etc..
    pred_action = postprocess(pred_action)

    # convert the predicted action to a robot action and send it to the robot
    pred_action = make_robot_action(pred_action, dataset_features)
    print(f"Predicted action: {pred_action}")

    robot.send_action(pred_action)

robot.disconnect()