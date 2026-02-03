# RLBench Vector Environment (Headless, Torchrun/Multi-GPU)

This repository provides a multi-process vectorized environment wrapper for RLBench, designed for high-throughput headless rollouts and `torchrun`-based multi-process launches.

## Features
- **Zero-Copy Shared Memory**: Efficient observation transfer between worker processes and the main training process.
- **Robust Headless Support**: Automatically manages Xvfb instances for each environment worker, preventing display conflicts.
- **Torchrun Compatible**: Designed to work seamlessly with `torchrun` for multi-GPU distributed training.
- **Throughput Presets**: Flags like `--fast` / `--no_rgb` reduce rendering/planning overhead for benchmarking.
- **Experimental EGL Backend Support**: `--render_backend egl` tries GPU-backed headless rendering (requires system EGL/NVIDIA runtime).

## Prerequisites

1.  **Install System Dependencies**:
    Run the provided setup script to install necessary system packages (Xvfb, Qt dependencies, etc.) and Python requirements.
    ```bash
    chmod +x scripts/setup_remote.sh
    ./scripts/setup_remote.sh
    ```

2.  **CoppeliaSim**:
    **CRITICAL**: You MUST use **CoppeliaSim V4.1.0**. Newer versions (like 4.10.0) are NOT compatible with RLBench/PyRep and will cause "Handle ... does not exist" errors.
    
    Use the provided script to install the correct version:
    ```bash
    chmod +x scripts/install_coppeliasim.sh
    ./scripts/install_coppeliasim.sh
    source ~/.bashrc  # Or manually export the variables printed by the script
    ```
    
    Ensure `COPPELIASIM_ROOT` environment variable is set to your CoppeliaSim 4.1.0 installation directory.

3.  **Reinstall PyRep & RLBench**:
    If you changed your CoppeliaSim version (e.g. downgraded from 4.10 to 4.1), you **MUST** reinstall PyRep so it links against the correct library.
    
    ```bash
    chmod +x scripts/reinstall_rlbench.sh
    ./scripts/reinstall_rlbench.sh
    ```

## Usage with Torchrun

To run across multiple processes (e.g., 4 GPUs / 4 ranks), use `torchrun`.

**Important**:
- `--num_envs` is **per-rank**.
- `--num_envs_total` is **total across all ranks** (recommended), and the script will split it across ranks.

```bash
torchrun --nproc_per_node=4 rl_vec_env.py \
    --num_envs_total 64 \
    --max_steps 100 \
    --task_class OpenDrawer \
    --action_chunk 54 \
    --fast \
    --ddp_bind_gpu
```

### Single-process (one Python process, N env worker processes)

```bash
python rl_vec_env.py \
  --num_envs 64 \
  --max_steps 100 \
  --task_class sweep_to_dustpan \
  --action_chunk 54 \
  --fast
```

### Arguments
- `--num_envs`: Env count per rank (one worker process per env).
- `--num_envs_total`: Total env count across all ranks when using `torchrun`.
- `--task_class`: RLBench task (e.g., `OpenDrawer`, `sweep_to_dustpan`).
- `--max_steps`: Total steps per rank (benchmark loop).
- `--action_chunk`: Chunk length for `step_chunk`.
- `--arm_mode`: `ik` (default, faster) or `planning` (slower).
- `--fast`: Preset for throughput benchmarking (`--arm_mode ik --no_rgb --no_video --idle_fps 0`).
- `--no_rgb`: Disable camera RGB (major speedup).
- `--no_video`: Disable video recording.
- `--render_backend`: `xvfb` (default), `egl` (no X server), `external` (use an existing DISPLAY).

## Troubleshooting

- **Xvfb Errors**: If you see "Xvfb failed", ensure `xvfb` is installed (`sudo apt-get install xvfb`). The script automatically retries and allocates unique displays.
- **Qt/CoppeliaSim Crashes**: Ensure you have installed the system dependencies via [`scripts/setup_remote.sh`](scripts/setup_remote.sh:1).
- **Shared Memory Leaks**: If the script crashes hard, shared memory segments might leak. They are usually cleaned up automatically, but you can manually clean `/dev/shm/psm_*` if needed.

### EGL rendering notes
`--render_backend egl` attempts GPU-backed headless rendering (no Xorg), but it depends on system/container runtime support:
- NVIDIA drivers available inside the runtime
- GLVND/EGL libs available (`libEGL.so`, vendor libs)

If EGL is not available, the run may fall back to software rendering or crash during context creation.
