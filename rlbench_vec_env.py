#!/usr/bin/env python3
"""
Optimized RLBench Vectorized Environment with Shared Memory
适配 torchrun + headless，旨在解决大规模并行下的性能瓶颈。

主要优化点：
1. Shared Memory (共享内存)：避免图像数据的 Pickling 开销 (Zero-copy)。
2. 进程管理：使用 spawn 启动，兼容 PyTorch 分布式环境。
3. 视频录制优化：可选开启，避免 I/O 阻塞。
"""

import os
import time
import argparse
import multiprocessing as mp
from multiprocessing import shared_memory
import numpy as np
import cv2


def _get_int_env(keys, default: int) -> int:
    for k in keys:
        v = os.environ.get(k)
        if v is None:
            continue
        try:
            return int(str(v).strip())
        except Exception:
            continue
    return int(default)


def _get_local_rank() -> int:
    # torchrun sets LOCAL_RANK; SLURM uses SLURM_LOCALID.
    return _get_int_env(["LOCAL_RANK", "SLURM_LOCALID"], 0)


def _get_global_rank() -> int:
    return _get_int_env(["RANK", "SLURM_PROCID"], 0)


def _maybe_set_display_from_base() -> None:
    """Optionally set DISPLAY for multi-GPU headless X setup.

    RLBench upstream suggests starting an X server on :99 and then selecting GPU by
    using DISPLAY=:99.<gpu_id>. For torchrun, <gpu_id> typically matches LOCAL_RANK.

    Behavior:
    - If RLBENCH_BASE_DISPLAY is set and DISPLAY is empty, set DISPLAY.
    - Format is controlled by RLBENCH_DISPLAY_FMT (default: "{base}.{gpu}").
      Example: base=":99", gpu=0 -> ":99.0".
    """

    if str(os.environ.get("DISPLAY", "")).strip() != "":
        return
    base = str(os.environ.get("RLBENCH_BASE_DISPLAY", "")).strip()
    if base == "":
        return
    fmt = str(os.environ.get("RLBENCH_DISPLAY_FMT", "{base}.{gpu}")).strip()
    lr = _get_local_rank()
    os.environ["DISPLAY"] = fmt.format(base=base, gpu=lr, local_rank=lr, rank=_get_global_rank())


def _require_env(var: str) -> None:
    if var not in os.environ or str(os.environ[var]).strip() == "":
        raise RuntimeError(f"Missing required environment variable: {var}")


def _start_xvfb(display_str: str, max_tries: int = 64):
    """Start a dedicated Xvfb server for the worker and return (proc, actual_display)."""
    import subprocess

    if display_str is None or str(display_str).strip() == "":
        raise ValueError("display_str must be non-empty")

    # Qt may need XDG_RUNTIME_DIR on headless servers
    if "XDG_RUNTIME_DIR" not in os.environ or str(os.environ["XDG_RUNTIME_DIR"]).strip() == "":
        os.environ["XDG_RUNTIME_DIR"] = "/tmp/runtime-root"
        try:
            os.makedirs(os.environ["XDG_RUNTIME_DIR"], exist_ok=True)
            try:
                os.chmod(os.environ["XDG_RUNTIME_DIR"], 0o700)
            except Exception:
                pass
        except Exception:
            pass

    try:
        start_disp = int(str(display_str).lstrip(":"))
    except Exception as e:
        raise ValueError(f"Invalid display_str: {display_str}") from e

    last_err = None
    for off in range(int(max_tries)):
        disp = int(start_disp + off)
        disp_str = f":{disp}"
        sock_path = f"/tmp/.X11-unix/X{disp}"
        if os.path.exists(sock_path):
            continue

        os.environ["DISPLAY"] = disp_str
        p = subprocess.Popen(
            ["Xvfb", disp_str, "-screen", "0", "1024x768x24", "-ac", "-nolisten", "tcp"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

        ok = False
        for _ in range(50):  # ~2.5s
            if p.poll() is not None:
                err_txt = ""
                try:
                    if p.stderr is not None:
                        err_txt = p.stderr.read().decode("utf-8", errors="ignore")
                except Exception:
                    err_txt = ""
                last_err = f"Xvfb failed: display={disp_str} exit_code={p.returncode} stderr={err_txt.strip()}"
                break
            if os.path.exists(sock_path):
                ok = True
                break
            time.sleep(0.05)

        if ok:
            try:
                if p.stderr is not None:
                    p.stderr.close()
            except Exception:
                pass
            return p, disp_str

        try:
            if p.poll() is None:
                p.terminate()
                try:
                    p.wait(timeout=1.0)
                except Exception:
                    p.kill()
        except Exception:
            pass

    raise RuntimeError(last_err if last_err is not None else f"Xvfb failed: no available DISPLAY starting at {display_str}")

# =============================================================================
# Worker Function
# =============================================================================

def _worker_entry(
    rank: int,
    pipe: mp.connection.Connection,
    shm_info: dict,
    env_config: dict
):
    """
    Worker 进程入口。
    负责维护一个 RLBench 环境，并将 Observation 写入共享内存。
    """
    # 1. 配置 Headless Xvfb
    # -------------------------------------------------------------------------
    # 尝试为每个 Worker 分配独立的 Display，避免冲突
    # 注意：如果系统资源不足，可能需要考虑多个 Worker 共享 Display（需测试稳定性）
    xvfb_proc = None
    try:
        per_process_xvfb = True
        if "RLBENCH_PER_PROCESS_XVFB" in os.environ:
            per_process_xvfb = str(os.environ["RLBENCH_PER_PROCESS_XVFB"]).strip() == "1"

        # If the user pre-started an X server (RLBench upstream headless instructions),
        # allow selecting a GPU-specific DISPLAY like ":99.<LOCAL_RANK>".
        _maybe_set_display_from_base()

        if env_config.get("headless", True) and per_process_xvfb:
            # Make DISPLAY allocation stable across torchrun ranks to reduce collisions.
            # Each torchrun rank gets its own window of display numbers.
            local_rank = _get_local_rank()
            xvfb_base = _get_int_env(["RLBENCH_XVFB_BASE"], 10000)
            xvfb_stride = _get_int_env(["RLBENCH_XVFB_STRIDE"], 2000)
            start_display = f":{int(xvfb_base + local_rank * xvfb_stride + int(rank))}"
            xvfb_proc, actual_display = _start_xvfb(start_display)
            os.environ["DISPLAY"] = str(actual_display)

    except Exception as e:
        pipe.send(("error", f"Xvfb setup failed: {e}"))
        return

    # 2. 初始化 RLBench 环境
    # -------------------------------------------------------------------------
    try:
        from rlbench.environment import Environment
        from rlbench.action_modes.action_mode import MoveArmThenGripper
        from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaPlanning
        from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaIK
        from rlbench.action_modes.gripper_action_modes import Discrete
        from rlbench.observation_config import ObservationConfig, CameraConfig
        from rlbench.utils import name_to_task_class

        def _normalize_task_name(name: str) -> str:
            # Accept either CamelCase ("OpenDrawer") or snake_case ("open_drawer").
            n = str(name).strip()
            if "_" in n or "-" in n:
                parts = [p for p in n.replace("-", "_").split("_") if p]
                return "".join([p[:1].upper() + p[1:] for p in parts])
            return n

        # 配置 Observation
        img_size = env_config["image_size"] # (H, W)
        obs_config = ObservationConfig()
        obs_config.set_all(False)
        # 开启需要的相机
        obs_config.front_camera = CameraConfig(rgb=True, image_size=img_size)
        obs_config.wrist_camera = CameraConfig(rgb=True, image_size=img_size)
        obs_config.gripper_pose = True
        obs_config.gripper_open = True

        arm_mode = str(env_config.get("arm_mode", "planning")).lower().strip()
        if arm_mode == "planning":
            arm_action_mode = EndEffectorPoseViaPlanning()
        elif arm_mode == "ik":
            arm_action_mode = EndEffectorPoseViaIK()
        else:
            raise ValueError(f"Unsupported arm_mode: {arm_mode} (supported: planning, ik)")

        action_mode = MoveArmThenGripper(arm_action_mode=arm_action_mode, gripper_action_mode=Discrete())

        env = Environment(
            action_mode=action_mode,
            obs_config=obs_config,
            headless=True, # RLBench 内部的 headless 标志
            robot_setup=env_config["robot_setup"]
        )
        env.launch()

        task_class = name_to_task_class(_normalize_task_name(env_config["task_name"]))
        task_env = env.get_task(task_class)

    except Exception as e:
        pipe.send(("error", f"RLBench init failed: {e}"))
        if xvfb_proc: xvfb_proc.kill()
        return

    # 3. 连接共享内存
    # -------------------------------------------------------------------------
    shm_objs = {}
    arrays = {}
    try:
        for name, info in shm_info.items():
            shm = shared_memory.SharedMemory(name=info["name"])
            shm_objs[name] = shm
            # 创建 numpy 数组视图
            # 注意：这里创建的是整个 batch 的视图，我们需要切片出当前 rank 的部分
            full_arr = np.ndarray(info["shape"], dtype=np.dtype(info["dtype"]), buffer=shm.buf)
            arrays[name] = full_arr[rank] # Slice: [rank, ...]
    except Exception as e:
        pipe.send(("error", f"Shared memory connect failed: {e}"))
        return

    # 视频录制相关
    video_writer_front = None
    video_writer_wrist = None
    recording = env_config.get("record_video", False)
    output_dir = env_config.get("output_dir", "./videos")
    fps = env_config.get("fps", 20.0)
    
    if recording:
        os.makedirs(output_dir, exist_ok=True)

    # 辅助函数：写入共享内存
    def write_obs_to_shm(obs):
        # Front RGB
        if hasattr(obs, "front_rgb"):
            arrays["front_rgb"][:] = np.asarray(obs.front_rgb, dtype=np.uint8)
        # Wrist RGB
        if hasattr(obs, "wrist_rgb"):
            arrays["wrist_rgb"][:] = np.asarray(obs.wrist_rgb, dtype=np.uint8)
        # Gripper Pose
        if hasattr(obs, "gripper_pose"):
            arrays["gripper_pose"][:] = np.asarray(obs.gripper_pose, dtype=np.float32)
        # Gripper Open
        if hasattr(obs, "gripper_open"):
            # RLBench may expose scalar or 1-element array
            go = np.asarray(obs.gripper_open, dtype=np.float32).reshape(-1)
            arrays["gripper_open"][0] = float(go[0])

    # 4. 主循环
    # -------------------------------------------------------------------------
    pipe.send("ready")

    done_flag = False
    try:
        while True:
            cmd, data = pipe.recv()

            if cmd == "reset":
                desc, obs = task_env.reset()
                done_flag = False
                write_obs_to_shm(obs)
                
                # 重置视频录制
                if recording:
                    if video_writer_front: video_writer_front.release()
                    if video_writer_wrist: video_writer_wrist.release()
                    vf_path = os.path.join(output_dir, f"env{rank}_front.mp4")
                    vw_path = os.path.join(output_dir, f"env{rank}_wrist.mp4")
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    video_writer_front = cv2.VideoWriter(vf_path, fourcc, fps, (img_size[1], img_size[0]))
                    video_writer_wrist = cv2.VideoWriter(vw_path, fourcc, fps, (img_size[1], img_size[0]))
                    
                    # 写入第一帧
                    if hasattr(obs, "front_rgb"):
                        video_writer_front.write(cv2.cvtColor(obs.front_rgb, cv2.COLOR_RGB2BGR))
                    if hasattr(obs, "wrist_rgb"):
                        video_writer_wrist.write(cv2.cvtColor(obs.wrist_rgb, cv2.COLOR_RGB2BGR))

                pipe.send("done")

            elif cmd == "step":
                if done_flag:
                    arrays["reward"][0] = 0.0
                    arrays["done"][0] = True
                    pipe.send("done")
                    continue
                action = data # (8,)
                obs, reward, terminate = task_env.step(action)
                
                write_obs_to_shm(obs)
                arrays["reward"][0] = reward
                arrays["done"][0] = terminate
                done_flag = bool(terminate)

                if recording:
                    if hasattr(obs, "front_rgb"):
                        video_writer_front.write(cv2.cvtColor(obs.front_rgb, cv2.COLOR_RGB2BGR))
                    if hasattr(obs, "wrist_rgb"):
                        video_writer_wrist.write(cv2.cvtColor(obs.wrist_rgb, cv2.COLOR_RGB2BGR))

                pipe.send("done")

            elif cmd == "step_chunk":
                # actions: (K, 8)
                if done_flag:
                    arrays["reward_sum"][0] = 0.0
                    arrays["n_steps"][0] = 0
                    arrays["reward"][0] = 0.0
                    arrays["done"][0] = True
                    pipe.send("done")
                    continue

                actions = np.asarray(data, dtype=np.float32)
                if actions.ndim != 2 or actions.shape[1] != 8:
                    raise ValueError(f"step_chunk expects actions (K, 8), got {actions.shape}")

                reward_sum = 0.0
                n_steps = 0
                last_reward = 0.0
                for j in range(int(actions.shape[0])):
                    obs, reward, terminate = task_env.step(actions[j])
                    last_reward = float(reward)
                    reward_sum += float(reward)
                    n_steps += 1
                    write_obs_to_shm(obs)
                    if bool(terminate):
                        done_flag = True
                        break

                arrays["reward_sum"][0] = float(reward_sum)
                arrays["n_steps"][0] = int(n_steps)
                arrays["reward"][0] = float(last_reward)
                arrays["done"][0] = bool(done_flag)
                pipe.send("done")

            elif cmd == "close":
                break
    except Exception as e:
        pipe.send(("error", str(e)))
    finally:
        # Cleanup
        if recording:
            if video_writer_front: video_writer_front.release()
            if video_writer_wrist: video_writer_wrist.release()
        
        try:
            env.shutdown()
        except:
            pass
            
        if xvfb_proc:
            xvfb_proc.terminate()
            
        for shm in shm_objs.values():
            shm.close()
        
        pipe.close()


# =============================================================================
# Vector Environment Class
# =============================================================================

class RLBenchVectorEnv:
    def __init__(
        self,
        num_envs,
        task_name,
        image_size=(256, 256),
        robot_setup="panda",
        record_video=False,
        output_dir="",
        fps=20.0,
        arm_mode="planning",
        copy_obs=True,
    ):
        self.num_envs = num_envs
        self.image_size = image_size
        self.copy_obs = bool(copy_obs)

        _require_env("COPPELIASIM_ROOT")
        _maybe_set_display_from_base()
        
        # 1. 定义共享内存结构
        # ---------------------------------------------------------------------
        # 格式: {name: (shape, dtype)}
        # 注意：Shape 的第一维是 num_envs
        self.shm_specs = {
            "front_rgb":    ((num_envs, image_size[0], image_size[1], 3), np.uint8),
            "wrist_rgb":    ((num_envs, image_size[0], image_size[1], 3), np.uint8),
            "gripper_pose": ((num_envs, 7), np.float32),
            "gripper_open": ((num_envs, 1), np.float32),
            "reward":       ((num_envs, 1), np.float32),
            "reward_sum":   ((num_envs, 1), np.float32),
            "n_steps":      ((num_envs, 1), np.int32),
            "done":         ((num_envs, 1), np.bool_),
        }

        self.shm_objs = []
        self.shm_info = {} # 传递给 Worker 的信息
        self.arrays = {}   # 主进程访问的 Numpy 视图

        for name, (shape, dtype) in self.shm_specs.items():
            # 计算字节大小
            size = int(np.prod(shape)) * np.dtype(dtype).itemsize
            shm = shared_memory.SharedMemory(create=True, size=size)
            self.shm_objs.append(shm)
            
            self.shm_info[name] = {
                "name": shm.name,
                "shape": shape,
                "dtype": np.dtype(dtype).str,
            }
            
            # 创建主进程视图
            self.arrays[name] = np.ndarray(shape, dtype=dtype, buffer=shm.buf)

        # 2. 启动 Workers
        # ---------------------------------------------------------------------
        self.ctx = mp.get_context("spawn")
        self.procs = []
        self.pipes = []
        
        env_config = {
            "task_name": task_name,
            "image_size": image_size,
            "robot_setup": robot_setup,
            "record_video": record_video,
            "output_dir": output_dir,
            "fps": float(fps),
            "arm_mode": str(arm_mode),
            "headless": True,
        }

        print(f"[RLBenchVecEnv] Starting {num_envs} workers...")
        for i in range(num_envs):
            parent_conn, child_conn = self.ctx.Pipe()
            p = self.ctx.Process(
                target=_worker_entry,
                args=(i, child_conn, self.shm_info, env_config)
            )
            p.start()
            self.procs.append(p)
            self.pipes.append(parent_conn)
            child_conn.close() # Close child handle in parent

        # 等待所有 Worker Ready
        for i, pipe in enumerate(self.pipes):
            msg = pipe.recv()
            if msg != "ready":
                if isinstance(msg, tuple) and msg[0] == "error":
                    raise RuntimeError(f"Worker {i} failed: {msg[1]}")
                raise RuntimeError(f"Worker {i} failed to start: {msg}")
        
        print(f"[RLBenchVecEnv] All {num_envs} workers ready.")

    def reset(self):
        """
        重置所有环境。
        """
        for pipe in self.pipes:
            pipe.send(("reset", None))
        
        # 等待完成
        for pipe in self.pipes:
            self._recv_ack(pipe)
            
        return self._get_obs_dict()

    def step(self, actions):
        """
        执行一步。
        actions: (num_envs, 8) numpy array
        """
        # 发送 Action
        for i, pipe in enumerate(self.pipes):
            pipe.send(("step", actions[i]))
            
        # 等待完成
        for pipe in self.pipes:
            self._recv_ack(pipe)
            
        obs = self._get_obs_dict()
        reward = self.arrays["reward"].copy()
        done = self.arrays["done"].copy()
        
        return obs, reward, done

    def step_chunk(self, actions_chunk):
        """
        Execute a chunk of K steps per env.

        actions_chunk: (num_envs, K, 8) float32
        Returns: obs, reward_sum, done, n_steps
        """
        actions_chunk = np.asarray(actions_chunk, dtype=np.float32)
        if actions_chunk.ndim != 3 or actions_chunk.shape[0] != self.num_envs or actions_chunk.shape[2] != 8:
            raise ValueError(f"actions_chunk must be (num_envs, K, 8), got {actions_chunk.shape}")

        for i, pipe in enumerate(self.pipes):
            pipe.send(("step_chunk", actions_chunk[i]))
        for pipe in self.pipes:
            self._recv_ack(pipe)

        obs = self._get_obs_dict()
        reward_sum = self.arrays["reward_sum"].copy()
        done = self.arrays["done"].copy()
        n_steps = self.arrays["n_steps"].copy()
        return obs, reward_sum, done, n_steps

    def _recv_ack(self, pipe):
        msg = pipe.recv()
        if msg != "done":
            if isinstance(msg, tuple) and msg[0] == "error":
                raise RuntimeError(f"Worker error: {msg[1]}")
            raise RuntimeError(f"Unexpected worker msg: {msg}")

    def _get_obs_dict(self):
        if self.copy_obs:
            return {
                "front_rgb": self.arrays["front_rgb"].copy(),
                "wrist_rgb": self.arrays["wrist_rgb"].copy(),
                "gripper_pose": self.arrays["gripper_pose"].copy(),
                "gripper_open": self.arrays["gripper_open"].copy(),
            }
        # zero-copy views (do not mutate)
        return {
            "front_rgb": self.arrays["front_rgb"],
            "wrist_rgb": self.arrays["wrist_rgb"],
            "gripper_pose": self.arrays["gripper_pose"],
            "gripper_open": self.arrays["gripper_open"],
        }

    def close(self):
        print("[RLBenchVecEnv] Closing...")
        for pipe in self.pipes:
            try:
                pipe.send(("close", None))
            except:
                pass
        
        for p in self.procs:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()
                
        # 释放共享内存
        for shm in self.shm_objs:
            shm.close()
            shm.unlink()
        print("[RLBenchVecEnv] Closed.")

# =============================================================================
# Benchmark / Test Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_envs", type=int, default=4)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--task", type=str, default="OpenDrawer")
    parser.add_argument("--record", action="store_true", help="Enable video recording (slows down)")
    parser.add_argument("--output_dir", type=str, default="rlbench_optimized_videos", help="Video output dir (if --record)")
    parser.add_argument("--fps", type=float, default=20.0, help="Video FPS (if --record)")
    parser.add_argument("--robot_setup", type=str, default="panda", help="RLBench robot_setup (e.g., panda)")
    parser.add_argument("--image_size", type=int, nargs=2, default=[256, 256], help="Camera resolution [H W]")
    parser.add_argument("--arm_mode", type=str, default="planning", choices=["planning", "ik"], help="Arm control mode")
    parser.add_argument("--action_chunk", type=int, default=1, help="If >1, run step_chunk with this K")
    parser.add_argument("--copy_obs", type=int, default=1, help="1: return copied obs; 0: return zero-copy views")
    args = parser.parse_args()

    # 模拟 Torchrun 环境（如果直接运行此脚本）
    # 在实际 Torchrun 中，LOCAL_RANK 等变量会被设置
    
    print(f"Initializing {args.num_envs} environments for task {args.task}...")
    
    env = RLBenchVectorEnv(
        num_envs=args.num_envs,
        task_name=args.task,
        image_size=(int(args.image_size[0]), int(args.image_size[1])),
        record_video=args.record,
        output_dir=str(args.output_dir),
        fps=float(args.fps),
        arm_mode=str(args.arm_mode),
        robot_setup=str(args.robot_setup),
        copy_obs=bool(int(args.copy_obs)),
    )

    try:
        print("Resetting...")
        t0 = time.time()
        obs = env.reset()
        t_reset = time.time() - t0
        print(f"Reset done in {t_reset:.2f}s")

        if int(args.action_chunk) <= 0:
            raise ValueError("--action_chunk must be >= 1")
        print(f"Running {args.steps} steps... (action_chunk={int(args.action_chunk)})")
        
        # 构造随机 Action
        # RLBench Action: [x, y, z, qx, qy, qz, qw, gripper]
        # 这里简单使用 gripper_pose 加噪声
        
        total_time = 0.0
        rng = np.random.default_rng(0)
        s = 0
        while s < int(args.steps):
            k = int(min(int(args.action_chunk), int(args.steps) - s))
            current_pose = obs["gripper_pose"]  # (N, 7)
            # If copy_obs=0, current_pose is a view backed by shared memory; copy minimal data needed.
            base_pose = np.asarray(current_pose, dtype=np.float32)

            if k == 1:
                noise = rng.normal(0, 0.001, size=(args.num_envs, 3)).astype(np.float32)
                actions = np.zeros((args.num_envs, 8), dtype=np.float32)
                actions[:, :3] = base_pose[:, :3] + noise
                actions[:, 3:7] = base_pose[:, 3:7]
                actions[:, 7] = 1.0

                t_start = time.time()
                obs, reward, done = env.step(actions)
                t_step = time.time() - t_start
                total_time += t_step
            else:
                # Build chunk: (N, K, 8)
                pos_noise = rng.normal(0, 0.001, size=(args.num_envs, k, 3)).astype(np.float32)
                actions_chunk = np.zeros((args.num_envs, k, 8), dtype=np.float32)
                actions_chunk[:, :, :3] = base_pose[:, None, :3] + pos_noise
                actions_chunk[:, :, 3:7] = base_pose[:, None, 3:7]
                actions_chunk[:, :, 7] = 1.0

                t_start = time.time()
                obs, reward_sum, done, n_steps = env.step_chunk(actions_chunk)
                t_step = time.time() - t_start
                total_time += t_step

            if s % 10 == 0:
                fps = (args.num_envs * k) / max(t_step, 1e-9)
                print(f"Step {s}/{args.steps}: {t_step:.4f}s (ChunkFPS: {fps:.1f})")

            s += k

        print(f"Total time for {args.steps} steps: {total_time:.2f}s")
        print(f"Average FPS: {(args.num_envs * args.steps) / total_time:.2f}")
        print(f"Average time per step (batch): {total_time / args.steps:.4f}s")

    finally:
        env.close()

if __name__ == "__main__":
    # 必须使用 spawn
    mp.set_start_method("spawn", force=True)
    main()
