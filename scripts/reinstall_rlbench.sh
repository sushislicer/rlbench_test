#!/bin/bash
set -e

echo "Reinstalling PyRep and RLBench..."

# Run this from the repository root:
#   bash scripts/reinstall_rlbench.sh

# Check if COPPELIASIM_ROOT is set
if [ -z "$COPPELIASIM_ROOT" ]; then
    echo "Error: COPPELIASIM_ROOT is not set."
    echo "Please run 'source ~/.bashrc' or export the variable pointing to CoppeliaSim 4.1.0"
    exit 1
fi

echo "Using CoppeliaSim at: $COPPELIASIM_ROOT"

# Uninstall existing
pip uninstall -y pyrep rlbench

# Install PyRep
echo "Installing PyRep..."
pip install git+https://github.com/stepjam/PyRep.git

# Install RLBench
echo "Installing RLBench..."
pip install git+https://github.com/stepjam/RLBench.git

echo "Done. PyRep has been rebuilt against the current CoppeliaSim."
