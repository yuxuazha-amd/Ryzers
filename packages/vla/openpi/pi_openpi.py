from openpi.policies import droid_policy, libero_policy
from openpi.policies import policy_config as _policy_config
from openpi.shared import download
from openpi.training import config as _config
from huggingface_hub import snapshot_download

config = _config.get_config("pi0_fast_libero")
# Download the checkpoint from Hugging Face instead of gs://openpi-assets.
checkpoint_dir = snapshot_download(repo_id="doremifasolatido/pi0-fast-libero-lora")

# # Download the checkpoint from gs://openpi-assets instead of Hugging Face.
# config = _config.get_config("pi05_libero")
# checkpoint_dir = download.maybe_download("gs://openpi-assets/checkpoints/pi05_libero")

# Create a trained policy.
policy = _policy_config.create_trained_policy(config, checkpoint_dir)

# Run inference on a dummy LIBERO observation
# (8-dim state, base image, wrist image, prompt).
example = libero_policy.make_libero_example()
result = policy.infer(example)

del policy
print("Actions shape:", result["actions"].shape)
