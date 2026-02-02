# RLBench Vector Environment for Headless Multi-GPU Training

This repository provides an optimized, vectorized environment wrapper for RLBench, designed for high-performance headless training using `torchrun`.

## Features
- **Zero-Copy Shared Memory**: Efficient observation transfer between worker processes and the main training process.
- **Robust Headless Support**: Automatically manages Xvfb instances for each environment worker, preventing display conflicts.
- **Torchrun Compatible**: Designed to work seamlessly with `torchrun` for multi-GPU distributed training.

## Prerequisites

1.  **Install System Dependencies**:
    Run the provided setup script to install necessary system packages (Xvfb, Qt dependencies, etc.) and Python requirements.
    ```bash
    chmod +x setup_remote.sh
    ./setup_remote.sh
    ```

2.  **CoppeliaSim**:
    **CRITICAL**: You MUST use **CoppeliaSim V4.1.0**. Newer versions (like 4.10.0) are NOT compatible with RLBench/PyRep and will cause "Handle ... does not exist" errors.
    
    Use the provided script to install the correct version:
    ```bash
    chmod +x install_coppeliasim.sh
    ./install_coppeliasim.sh
    source ~/.bashrc  # Or manually export the variables printed by the script
    ```
    
    Ensure `COPPELIASIM_ROOT` environment variable is set to your CoppeliaSim 4.1.0 installation directory.

3.  **Reinstall PyRep & RLBench**:
    If you changed your CoppeliaSim version (e.g. downgraded from 4.10 to 4.1), you **MUST** reinstall PyRep so it links against the correct library.
    
    ```bash
    chmod +x reinstall_rlbench.sh
    ./reinstall_rlbench.sh
    ```

## Usage with Torchrun

To run the vectorized environment across multiple GPUs (e.g., 4 GPUs), use `torchrun`.

**Note**: The `--num_envs` argument specifies the number of environments **per process**.
If you run 4 processes with `--num_envs 8`, you will have a total of 32 environments (8 per GPU).

```bash
torchrun --nproc_per_node=4 rlbench_vec_env.py \
    --num_envs 8 \
    --steps 100 \
    --task OpenDrawer \
    --robot_setup panda
```

### Arguments
- `--num_envs`: Number of parallel environments per process (default: 4).
- `--task`: RLBench task name (e.g., `OpenDrawer`, `ReachTarget`).
- `--steps`: Number of steps to run in the benchmark loop.
- `--image_size`: Camera resolution (default: 256 256).
- `--arm_mode`: Arm control mode (`planning` or `ik`).
- `--record`: Enable video recording (saved to `rlbench_optimized_videos`).

## Troubleshooting

- **Xvfb Errors**: If you see "Xvfb failed", ensure `xvfb` is installed (`sudo apt-get install xvfb`). The script automatically retries and allocates unique displays.
- **Qt/CoppeliaSim Crashes**: Ensure you have installed the dependencies listed in `apt-packages.txt` (handled by `setup_remote.sh`).
- **Shared Memory Leaks**: If the script crashes hard, shared memory segments might leak. They are usually cleaned up automatically, but you can manually clean `/dev/shm/psm_*` if needed.
