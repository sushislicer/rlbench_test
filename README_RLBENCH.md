# RLBench Optimized Vector Environment

此项目提供了一个基于 Shared Memory 和 Multiprocessing 的高性能 RLBench 并行环境封装，专为 `torchrun` 分布式训练和 Headless 服务器环境设计。

本仓库目前包含两种并行交互/基准测试方式：

1) [`mp_rlbench_env_test.py`](mp_rlbench_env_test.py):
   - 主进程 <-> N 个 worker 进程，每个 worker 维护 1 个 RLBench `TaskEnvironment`。
   - 通过 `step_chunk` 一次发送 K 个动作，worker 内部顺序执行并写视频（默认每 env 两路视频：front+wrist）。
   - 适合做“端到端跑通 + 诊断瓶颈”（把视频/渲染/规划都包含进去）。

2) [`rlbench_vec_env.py`](rlbench_vec_env.py):
   - 共享内存（SharedMemory）承载图像与低维状态，主进程 zero-copy 读取（可选返回拷贝）。
   - 新增 `step_chunk` 以减少 IPC 往返开销，更接近你想要的「vectorized env」吞吐测试。
   - 适合用于训练侧（例如重建 reward/视频编码器 reward）的大规模采样。

## 核心特性

*   **高性能**：使用 Shared Memory (共享内存) 传输图像和状态，避免了 Python Pickling 的巨大开销。
*   **Torchrun 兼容**：使用 `spawn` 启动模式，完美适配 PyTorch 分布式训练流程。
*   **Headless 优化**：自动为每个 Worker 进程管理独立的 Xvfb 显示服务（或根据配置共享），无需手动启动 `xvfb-run`。
*   **灵活配置**：支持可选的视频录制（避免 I/O 瓶颈）。

## 环境依赖安装指南

本指南基于 Linux (Ubuntu 20.04/22.04) 环境。

### 0. 安装本仓库脚本的 Python 依赖（推荐）

本仓库的 `rlbench_vec_env.py` / `mp_rlbench_env_test.py` 额外依赖一些纯 Python 包（例如 `gymnasium`、`opencv`、`numpy`）。

建议在独立虚拟环境里安装（避免与系统里已有的 numpy/scipy/matplotlib 等发生版本冲突）。

使用 pip：

```bash
python3 -m pip install -r requirements.txt
```

如果你使用 `uv`：

```bash
uv pip install -r requirements.txt
```

依赖版本说明：

* `requirements.txt` 里将 `opencv-python-headless` pin 到 `<4.12`，以兼容 `numpy<2`（OpenCV 4.12+ 在 Python>=3.9 上声明需要 numpy>=2）。

### 1. 系统依赖

首先安装必要的系统库和工具，特别是 Xvfb（用于 Headless 渲染）和 OpenGL 库：

```bash
sudo apt-get update
sudo apt-get install -y \
    git \
    wget \
    unzip \
    vim \
    libgl1-mesa-dev \
    libgl1-mesa-glx \
    libglew-dev \
    libosmesa6-dev \
    software-properties-common \
    net-tools \
    xvfb \
    x11-xserver-utils
```

### 2. 安装 CoppeliaSim

RLBench 上游 README 指明 RLBench 围绕 **CoppeliaSim v4.1.0** 与 **PyRep** 构建。建议优先按上游版本安装以避免兼容性问题。

```bash
# set env variables
export COPPELIASIM_ROOT=${HOME}/CoppeliaSim
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT

# download + extract (Ubuntu 20.04 build shown in upstream README)
wget https://downloads.coppeliarobotics.com/V4_1_0/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz
mkdir -p $COPPELIASIM_ROOT && tar -xf CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz -C $COPPELIASIM_ROOT --strip-components 1
rm -rf CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz
```

> 建议把上述 `export ...` 写入你的 `~/.bashrc` / `~/.zshrc`，并在新 shell 中 `source` 生效。

### 3. 安装 PyRep

PyRep 是 CoppeliaSim 的 Python 包装器。

```bash
git clone https://github.com/stepjam/PyRep.git
cd PyRep

# 安装 Python 依赖
python3 -m pip install -r requirements.txt

# 安装 PyRep（会根据 COPPELIASIM_ROOT 构建）
python3 -m pip install .
```

### 4. 安装 RLBench

上游 README 推荐直接用 pip 从 GitHub 安装：

```bash
python3 -m pip install git+https://github.com/stepjam/RLBench.git
```

如果你需要修改 RLBench 源码进行开发，可以改为 clone + editable：

```bash
git clone https://github.com/stepjam/RLBench.git
cd RLBench
python3 -m pip install -e .
```

### 5. 验证安装

运行以下 Python 代码验证环境是否配置正确：

```python
import os
from pyrep import PyRep
from rlbench.environment import Environment
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.observation_config import ObservationConfig
from rlbench.tasks import ReachTarget

print("CoppeliaSim Root:", os.environ.get("COPPELIASIM_ROOT"))
# 如果没有报错，说明安装成功
```

---

## Headless / 服务器渲染（对齐 RLBench 上游 README）

如果你在 **无显示器的服务器** 上需要 **GPU headless 渲染**（例如多卡服务器、云主机），RLBench 上游 README 推荐启动一个 X server（例如 `X :99`），并用 `DISPLAY=:99.<gpu_id>` 选择 GPU。

### 1) 一次性配置 X config（只需做一次）

```bash
sudo nvidia-xconfig -a --use-display-device=None --virtual=1280x1024

# increase max clients
echo -e 'Section "ServerFlags"\n\tOption "MaxClients" "2048"\nEndSection\n' | sudo tee /etc/X11/xorg.conf.d/99-maxclients.conf
```

> 如果你的 GPU 本身就是 headless（没有 display outputs），上游说明可去掉 `--use-display-device=None`。

### 2) 运行时启动 X server

```bash
# nohup + disown is important for the X server to keep running in the background
sudo nohup X :99 & disown
```

测试：

```bash
DISPLAY=:99 glxgears
```

多 GPU 选择（上游示例）：

```bash
DISPLAY=:99.<gpu_id> glxgears
```

---

## 运行指南

### 1. 直接运行基准测试 (Benchmark)

使用 Python 直接运行脚本进行性能测试：

```bash
# 运行 64 个环境，测试 100 步
python rlbench_vec_env.py --num_envs 64 --steps 100 --task OpenDrawer
```

建议在 headless SSH 上显式使用 `python3`：

```bash
python3 rlbench_vec_env.py --num_envs 64 --steps 100 --task OpenDrawer
```

为了提升吞吐，推荐：

```bash
# 1) 使用 IK（通常比 planning 快很多）
# 2) 使用 step_chunk 降低 IPC 往返（例如每次 K=16 步）
# 3) 返回 zero-copy 共享内存视图（copy_obs=0），训练侧自行小心不要原地修改
python3 rlbench_vec_env.py \
  --num_envs 64 --steps 100 --task OpenDrawer \
  --arm_mode ik --action_chunk 16 --copy_obs 0
```

如果你外部已经使用了 `xvfb-run` 或者你希望由外部统一管理 DISPLAY，可禁用每 worker 的 Xvfb：

```bash
export RLBENCH_PER_PROCESS_XVFB=0
xvfb-run -a python3 rlbench_vec_env.py --num_envs 16 --steps 100 --task OpenDrawer
```

如果需要录制视频（会显著降低速度，仅用于调试）：

```bash
python rlbench_vec_env.py --num_envs 4 --steps 50 --record
```

也可以使用 `mp_rlbench_env_test.py` 跑端到端（含双路视频写入 + chunk 执行时延统计）：

```bash
python3 mp_rlbench_env_test.py \
  --task_class OpenDrawer \
  --num_envs 8 \
  --max_steps 100 \
  --action_chunk 16 \
  --image_size 256 256
```

### 2. 使用 Torchrun 运行

在分布式训练场景下，通常使用 `torchrun` 启动。虽然本脚本主要演示环境封装，但可以直接作为 Worker 脚本运行：

#### (A) 4-GPU + headless（使用上游 X server 方法，推荐）

1) 先按上面的“Headless / 服务器渲染”启动 `X :99`。

2) 禁用每个 worker 自启 Xvfb，并让每个 `torchrun` rank 自动选择 `DISPLAY=:99.<LOCAL_RANK>`：

```bash
export RLBENCH_PER_PROCESS_XVFB=0
export RLBENCH_BASE_DISPLAY=:99
```

3) 4 GPU 并行：

```bash
# 总环境数 = nproc_per_node * num_envs
# 例如：4 GPUs * 16 env/rank = 64 env total
torchrun --nproc_per_node=4 rlbench_vec_env.py \
  --num_envs 16 \
  --steps 100 \
  --task OpenDrawer \
  --arm_mode ik \
  --action_chunk 16 \
  --copy_obs 0
```

#### (B) 4-GPU + headless（每个 worker 使用 Xvfb，偏兼容性优先）

如果你不想配置上游的 `X :99`，本仓库默认会给每个 worker 启动独立 Xvfb（见 [`_worker_entry()`](rlbench_vec_env.py:129)）。

```bash
torchrun --nproc_per_node=4 rlbench_vec_env.py --num_envs 8 --steps 100 --task OpenDrawer
```

**注意**：
*   `rlbench_vec_env.py` 内部使用了 `multiprocessing.set_start_method("spawn")`，这是 PyTorch 分布式要求的。
*   每个 `torchrun` 的 rank 进程会启动 `num_envs` 个子进程。如果你有 8 张卡，每张卡跑 8 个环境，则总共会有 64 个环境。请根据显存和 CPU 核心数调整 `num_envs`。

关于 task 名称：

* [`rlbench_vec_env.py`](rlbench_vec_env.py:1) / [`mp_rlbench_env_test.py`](mp_rlbench_env_test.py:1) 支持传入 `OpenDrawer` 这种 RLBench 的 **Task Class 名称**。
* 同时也支持传入 `open_drawer` / `sweep_to_dustpan` 这种 `snake_case`（脚本会自动转换成 CamelCase 再调用 RLBench 的 `name_to_task_class`）。

### mp_rlbench_env_test（你给出的命令格式）

```bash
python3 mp_rlbench_env_test.py \
  --task_class sweep_to_dustpan \
  --num_envs 64 \
  --max_steps 100 \
  --action_chunk 54 \
  --image_size 256 256 \
  --fps 20 \
  --output_dir /mnt/zezhong/Genie-Envisioner/RLBench_env_test/rlbench_env_test_videos \
  --pos_noise_std 0.01
```

## 性能优化建议

1.  **关闭视频录制**：除非必要，不要开启 `--record`。64 路视频编码和磁盘写入是巨大的 I/O 瓶颈。
2.  **CPU 核心绑定**：CoppeliaSim 非常消耗 CPU。确保机器有足够的物理核心。64 个环境建议至少有 64-128 个逻辑核心。
3.  **渲染频率**：RLBench 默认每步都渲染。如果不需要每步都获取图像，可以修改代码降低渲染频率。

额外建议（在大规模并行时常见提速手段）：

4. **避免 planning**：默认 arm 控制使用 planning（慢）。优先尝试 `--arm_mode ik`（见 [`EndEffectorPoseViaIK`](rlbench_vec_env.py:142) 的选择逻辑）。
5. **降低 IPC 往返**：使用 `--action_chunk K` 让每次通信执行 K 步（见 [`RLBenchVectorEnv.step_chunk()`](rlbench_vec_env.py:374)）。
6. **限制线程数避免 oversubscription**（尤其 32/64 env）：

```bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
```

## 代码结构

*   `rlbench_vec_env.py`: 核心实现文件。
    *   `_worker_entry`: 子进程工作函数，负责运行 RLBench 和 Xvfb。
    *   `RLBenchVectorEnv`: 主进程接口类，管理共享内存和子进程。
    *   `main`: 测试入口。

*   `mp_rlbench_env_test.py`: 多进程端到端交互测试脚本。
    *   适合验证 headless + 多进程稳定性，并拆分 `exec_s`/`ipc_s` 时延。
