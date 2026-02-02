# RLBench Optimized Vector Environment

此项目提供了一个基于 Shared Memory 和 Multiprocessing 的高性能 RLBench 并行环境封装，专为 `torchrun` 分布式训练和 Headless 服务器环境设计。

## 核心特性

*   **高性能**：使用 Shared Memory (共享内存) 传输图像和状态，避免了 Python Pickling 的巨大开销。
*   **Torchrun 兼容**：使用 `spawn` 启动模式，完美适配 PyTorch 分布式训练流程。
*   **Headless 优化**：自动为每个 Worker 进程管理独立的 Xvfb 显示服务（或根据配置共享），无需手动启动 `xvfb-run`。
*   **灵活配置**：支持可选的视频录制（避免 I/O 瓶颈）。

## 环境依赖安装指南

本指南基于 Linux (Ubuntu 20.04/22.04) 环境。

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

RLBench 依赖特定版本的 CoppeliaSim (通常推荐 V4.1.0 或 V4.2.0，请参考 PyRep 官方文档确认最新兼容版本)。

```bash
# 下载 CoppeliaSim Edu V4.1.0 (示例)
wget https://www.coppeliarobotics.com/files/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz

# 解压
tar -xf CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz

# 移动到合适的位置 (例如用户主目录)
mv CoppeliaSim_Edu_V4_1_0_Ubuntu20_04 $HOME/CoppeliaSim

# 配置环境变量 (建议写入 ~/.bashrc)
echo 'export COPPELIASIM_ROOT=$HOME/CoppeliaSim' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT' >> ~/.bashrc
echo 'export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT' >> ~/.bashrc
source ~/.bashrc
```

### 3. 安装 PyRep

PyRep 是 CoppeliaSim 的 Python 包装器。

```bash
git clone https://github.com/stepjam/PyRep.git
cd PyRep

# 安装 Python 依赖
pip install -r requirements.txt

# 安装 PyRep
pip install .
```

### 4. 安装 RLBench

```bash
cd ..
git clone https://github.com/stepjam/RLBench.git
cd RLBench

# 安装 Python 依赖
pip install -r requirements.txt

# 安装 RLBench
pip install .
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

## 运行指南

### 1. 直接运行基准测试 (Benchmark)

使用 Python 直接运行脚本进行性能测试：

```bash
# 运行 64 个环境，测试 100 步
python rlbench_vec_env.py --num_envs 64 --steps 100 --task OpenDrawer
```

如果需要录制视频（会显著降低速度，仅用于调试）：

```bash
python rlbench_vec_env.py --num_envs 4 --steps 50 --record
```

### 2. 使用 Torchrun 运行

在分布式训练场景下，通常使用 `torchrun` 启动。虽然本脚本主要演示环境封装，但可以直接作为 Worker 脚本运行：

```bash
# 单机单卡/多卡启动 (示例：使用 1 个节点，1 个 Master 进程)
torchrun --nproc_per_node=1 rlbench_vec_env.py --num_envs 64
```

**注意**：
*   `rlbench_vec_env.py` 内部使用了 `multiprocessing.set_start_method("spawn")`，这是 PyTorch 分布式要求的。
*   每个 `torchrun` 的 rank 进程会启动 `num_envs` 个子进程。如果你有 8 张卡，每张卡跑 8 个环境，则总共会有 64 个环境。请根据显存和 CPU 核心数调整 `num_envs`。

## 性能优化建议

1.  **关闭视频录制**：除非必要，不要开启 `--record`。64 路视频编码和磁盘写入是巨大的 I/O 瓶颈。
2.  **CPU 核心绑定**：CoppeliaSim 非常消耗 CPU。确保机器有足够的物理核心。64 个环境建议至少有 64-128 个逻辑核心。
3.  **渲染频率**：RLBench 默认每步都渲染。如果不需要每步都获取图像，可以修改代码降低渲染频率。

## 代码结构

*   `rlbench_vec_env.py`: 核心实现文件。
    *   `_worker_entry`: 子进程工作函数，负责运行 RLBench 和 Xvfb。
    *   `RLBenchVectorEnv`: 主进程接口类，管理共享内存和子进程。
    *   `main`: 测试入口。
