#!/bin/bash
set -e

echo "Setting up environment on remote machine..."

# 1. Install System Dependencies
if [ -f "apt-packages.txt" ]; then
    echo "Installing system packages from apt-packages.txt..."
    # Filter out comments and empty lines
    PACKAGES=$(grep -vE '^\s*($|#)' apt-packages.txt)
    if [ -n "$PACKAGES" ]; then
        sudo apt-get update
        sudo apt-get install -y $PACKAGES
    else
        echo "No packages found in apt-packages.txt"
    fi
else
    echo "apt-packages.txt not found, skipping system package installation."
fi

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
echo "Please ensure you have synced the latest version of 'rlbench_vec_env.py' and 'mp_rlbench_env_test.py' to this machine."
