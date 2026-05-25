#!/bin/bash
# Auto-generated script to run Docker with combined flags

# Enable X11 forwarding
xhost +local:docker

docker run -it --rm --shm-size 16G --cap-add=SYS_PTRACE  --network=host --ipc=host -v ~/.cache/huggingface/lerobot/calibration/:/opt/hf-cache/lerobot/calibration -v $PWD/packages/robotics/lerobot/mounted:/ryzers/lerobot_mounted -v ~/.cache/huggingface/:/root/.cache/huggingface/ -v $PWD/packages/vla/openpi/mounted:/ryzers/mounted --device=/dev/kfd --device=/dev/dri --security-opt seccomp=unconfined --group-add video --group-add render  -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix --device /dev/video0 --device /dev/video1 --device /dev/video2 --device /dev/video3 --device /dev/video4 --device /dev/video5 $(for d in /dev/ttyACM* /dev/video*; do [ -e "$d" ] && printf -- '--device=%s ' "$d"; done) -e HF_TOKEN ryzerdocker $1
