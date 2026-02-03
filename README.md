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

When `--num_envs_total` is set, [`rl_vec_env.py`](rl_vec_env.py:1) prints an explicit distribution like:
- `envs_per_rank=[16, 16, 16, 16]`

If you do **not** use `--num_envs_total`, then `--num_envs` is **per rank** (so total envs = `--num_envs * WORLD_SIZE`).

### Modes
[`rl_vec_env.py`](rl_vec_env.py:1) supports three modes:

- `--mode bench` (default): throughput benchmark loop.
- `--mode debug`: enables a video-friendly configuration (defaults `--output_dir` and increases `--video_stride` to reduce encode load).
- `--mode train`: minimal end-to-end loop that runs env interaction + a small PyTorch model forward/backward (for pipeline debugging, not a full RL algorithm).

Example (debug w/ videos, still trying to keep runtime reasonable):

```bash
torchrun --nproc_per_node=4 rl_vec_env.py \
  --mode debug \
  --num_envs_total 64 \
  --max_steps 100 \
  --task_class OpenDrawer \
  --action_chunk 54 \
  --arm_mode ik \
  --fps 20 \
  --output_dir rlbench_vec_env_videos \
  --video_stride 4 \
  --ddp_bind_gpu
```

Notes:
- By default, only env0 per rank records video; add `--video_all` to record all envs (much slower).
- `--fps` controls the encoded video FPS (default: 20).
- `--video_stride N` writes every N-th frame (reduces encode overhead). If you keep `--fps 20` with `--video_stride N`, the resulting video plays ~N× faster (time-lapse).

### Video recording
The `--fast` preset disables rendering (`--no_rgb`) and video recording (`--no_video`) on purpose.

To save videos in [`rl_vec_env.py`](rl_vec_env.py:1):
- pass `--output_dir <DIR>`
- do **not** pass `--no_video`
- do **not** pass `--no_rgb`

By default, videos are recorded only for env 0; use `--video_all` to record all envs.

### Success rate
[`rl_vec_env.py`](rl_vec_env.py:1) prints a simple proxy metric at the end:
`Success rate (done==True)` (fraction of envs that terminated early). In RLBench this typically corresponds to task success.

[`mp_rlbench_env_test.py`](mp_rlbench_env_test.py:1) prints per-chunk progress (`done=...`) and average return, and always writes videos for all envs.

### Profiling / time breakdown
For optimization work, you can enable a more detailed timing breakdown with `--profile_timing` in [`rl_vec_env.py`](rl_vec_env.py:1).

This adds small overhead (extra timers and extra stats sent back from workers), but it breaks down where time is spent:
- **Main process**:
  - `build`: action construction time
  - `env_call`: wall time spent inside `env.step(...)` / `env.step_chunk(...)`
- **Worker process (critical path env)**:
  - `sim`: time inside RLBench `task_env.step(...)`
  - `obs`: time copying observations into shared memory
  - `vid`: video encode/write time (when enabled)
  - `toggle`: time toggling camera RGB flags for `--video_stride`
  - `other`: any remaining worker-side overhead
  - `ipc`: computed as `total - exec` on the main side (Pipe + scheduling + copies outside worker timing)

Example:

```bash
python rl_vec_env.py --num_envs 64 --max_steps 100 --task_class sweep_to_dustpan --action_chunk 54 --profile_timing
```

### End-to-end training loop (pipeline debug)
If you want a minimal end-to-end loop (env + torch forward/backward) to validate distributed launch and throughput:

```bash
torchrun --nproc_per_node=4 rl_vec_env.py \
  --mode train \
  --num_envs_total 64 \
  --task_class OpenDrawer \
  --train_updates 20 \
  --train_lr 3e-4 \
  --train_hidden 256 \
  --ddp_bind_gpu
```

`--mode train` disables RGB/video by default for speed.

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
