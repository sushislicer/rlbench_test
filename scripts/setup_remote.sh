#!/bin/bash
set -e

echo "Setting up environment on remote machine..."

# 1. Install System Dependencies
echo "Installing system packages (Xvfb, OpenGL, Qt deps, ffmpeg)..."
sudo apt-get update

# Keep this list inline so the repo stays minimal.
# If your base image already contains some of these, apt will skip them.
PACKAGES=(
  git wget unzip
  xvfb x11-xserver-utils xserver-xorg mesa-utils
  libgl1-mesa-dev libgl1-mesa-glx libglew-dev libosmesa6-dev
  libx11-xcb1 libxcb-icccm4 libxcb-image0 libxcb-keysyms1 libxcb-randr0 libxcb-render-util0
  libxcb-shape0 libxcb-xfixes0 libxcb-xinerama0 libxkbcommon-x11-0
  ffmpeg
)
sudo apt-get install -y "${PACKAGES[@]}"

# 2. Install Python Dependencies
echo "Installing Python dependencies..."
pip install --upgrade pip
pip install gymnasium>=0.29.0
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
fi

# 3. Install RLBench and PyRep if not present (Optional, based on pyproject.toml)
# echo "Checking RLBench installation..."
# pip install git+https://github.com/stepjam/PyRep.git
# pip install git+https://github.com/stepjam/RLBench.git

echo "Environment setup complete."
echo "Please ensure you have synced the latest version of 'rl_vec_env.py' and 'mp_rl_env_test.py' to this machine."
