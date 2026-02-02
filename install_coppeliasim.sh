#!/bin/bash
set -e

echo "Installing CoppeliaSim Edu V4.1.0 (Required for RLBench)..."

# Define installation path
INSTALL_DIR="$HOME/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04"
TAR_FILE="CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz"
URL="https://downloads.coppeliarobotics.com/V4_1_0/$TAR_FILE"

if [ -d "$INSTALL_DIR" ]; then
    echo "CoppeliaSim 4.1.0 already exists at $INSTALL_DIR"
else
    echo "Downloading from $URL..."
    wget -q --show-progress "$URL"
    
    echo "Extracting..."
    tar -xf "$TAR_FILE" -C "$HOME"
    rm "$TAR_FILE"
fi

echo "Setting up environment variables..."
export COPPELIASIM_ROOT="$INSTALL_DIR"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$COPPELIASIM_ROOT"
export QT_QPA_PLATFORM_PLUGIN_PATH="$COPPELIASIM_ROOT"

echo "Done."
echo "Please run the following command to update your current shell:"
echo "export COPPELIASIM_ROOT=$INSTALL_DIR"
echo "export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:\$COPPELIASIM_ROOT"
echo "export QT_QPA_PLATFORM_PLUGIN_PATH=\$COPPELIASIM_ROOT"
