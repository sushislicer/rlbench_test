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
import ctypes

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
        if env_config.get("headless", True):
            # 简单的 Display 分配策略：基数 + rank
            display_id = 10000 + (os.getpid() % 10000) + rank
            display_str = f":{display_id}"
            os.environ["DISPLAY"] = display_str
            
            # 启动 Xvfb
            # -screen 0 1024x768x24 是常见配置
            xvfb_cmd = [
                "Xvfb", display_str, "-screen", "0", "1024x768x24", "-ac", "-nolisten", "tcp"
            ]
            import subprocess
            xvfb_proc = subprocess.Popen(
                xvfb_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            # 等待 Xvfb 启动
            time.sleep(1.0) 

    except Exception as e:
        pipe.send(("error", f"Xvfb setup failed: {e}"))
        return

    # 2. 初始化 RLBench 环境
    # -------------------------------------------------------------------------
    try:
        from rlbench import Environment
        from rlbench.action_modes.action_mode import MoveArmThenGripper
        from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaPlanning
        from rlbench.action_modes.gripper_action_modes import Discrete
        from rlbench.observation_config import ObservationConfig, CameraConfig
        from rlbench.utils import name_to_task_class

        # 配置 Observation
        img_size = env_config["image_size"] # (H, W)
        obs_config = ObservationConfig()
        obs_config.set_all(False)
        # 开启需要的相机
        obs_config.front_camera = CameraConfig(rgb=True, image_size=img_size)
        obs_config.wrist_camera = CameraConfig(rgb=True, image_size=img_size)
        obs_config.gripper_pose = True
        obs_config.gripper_open = True

        action_mode = MoveArmThenGripper(
            arm_action_mode=EndEffectorPoseViaPlanning(),
            gripper_action_mode=Discrete(),
        )

        env = Environment(
            action_mode=action_mode,
            obs_config=obs_config,
            headless=True, # RLBench 内部的 headless 标志
            robot_setup=env_config["robot_setup"]
        )
        env.launch()

        task_class = name_to_task_class(env_config["task_name"])
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
            full_arr = np.ndarray(info["shape"], dtype=info["dtype"], buffer=shm.buf)
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
            arrays["front_rgb"][:] = obs.front_rgb
        # Wrist RGB
        if hasattr(obs, "wrist_rgb"):
            arrays["wrist_rgb"][:] = obs.wrist_rgb
        # Gripper Pose
        if hasattr(obs, "gripper_pose"):
            arrays["gripper_pose"][:] = obs.gripper_pose
        # Gripper Open
        if hasattr(obs, "gripper_open"):
            arrays["gripper_open"][:] = obs.gripper_open

    # 4. 主循环
    # -------------------------------------------------------------------------
    pipe.send("ready")

    try:
        while True:
            cmd, data = pipe.recv()

            if cmd == "reset":
                desc, obs = task_env.reset()
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
                action = data # (8,)
                obs, reward, terminate = task_env.step(action)
                
                write_obs_to_shm(obs)
                arrays["reward"][0] = reward
                arrays["done"][0] = terminate

                if recording:
                    if hasattr(obs, "front_rgb"):
                        video_writer_front.write(cv2.cvtColor(obs.front_rgb, cv2.COLOR_RGB2BGR))
                    if hasattr(obs, "wrist_rgb"):
                        video_writer_wrist.write(cv2.cvtColor(obs.wrist_rgb, cv2.COLOR_RGB2BGR))

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
    def __init__(self, num_envs, task_name, image_size=(256, 256), robot_setup="panda", record_video=False, output_dir=""):
        self.num_envs = num_envs
        self.image_size = image_size
        
        # 1. 定义共享内存结构
        # ---------------------------------------------------------------------
        # 格式: {name: (shape, dtype)}
        # 注意：Shape 的第一维是 num_envs
        self.shm_specs = {
            "front_rgb":    ((num_envs, image_size[0], image_size[1], 3), np.uint8),
            "wrist_rgb":    ((num_envs, image_size[0], image_size[1], 3), np.uint8),
            "gripper_pose": ((num_envs, 7), np.float32),
            "gripper_open": ((num_envs,), np.float32),
            "reward":       ((num_envs, 1), np.float32),
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
                "dtype": dtype
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
            "headless": True
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

    def _recv_ack(self, pipe):
        msg = pipe.recv()
        if msg != "done":
            if isinstance(msg, tuple) and msg[0] == "error":
                raise RuntimeError(f"Worker error: {msg[1]}")
            raise RuntimeError(f"Unexpected worker msg: {msg}")

    def _get_obs_dict(self):
        # 返回当前共享内存的视图（或拷贝，如果需要防止修改）
        # 这里返回拷贝以安全使用，如果追求极致速度可返回视图但需小心
        return {
            "front_rgb": self.arrays["front_rgb"].copy(),
            "wrist_rgb": self.arrays["wrist_rgb"].copy(),
            "gripper_pose": self.arrays["gripper_pose"].copy(),
            "gripper_open": self.arrays["gripper_open"].copy(),
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
    args = parser.parse_args()

    # 模拟 Torchrun 环境（如果直接运行此脚本）
    # 在实际 Torchrun 中，LOCAL_RANK 等变量会被设置
    
    print(f"Initializing {args.num_envs} environments for task {args.task}...")
    
    env = RLBenchVectorEnv(
        num_envs=args.num_envs,
        task_name=args.task,
        record_video=args.record,
        output_dir="rlbench_optimized_videos"
    )

    try:
        print("Resetting...")
        t0 = time.time()
        obs = env.reset()
        t_reset = time.time() - t0
        print(f"Reset done in {t_reset:.2f}s")

        print(f"Running {args.steps} steps...")
        
        # 构造随机 Action
        # RLBench Action: [x, y, z, qx, qy, qz, qw, gripper]
        # 这里简单使用 gripper_pose 加噪声
        
        total_time = 0
        for s in range(args.steps):
            # 简单的 Mock Policy: 保持当前 Pose，加微小噪声
            current_pose = obs["gripper_pose"] # (N, 7)
            noise = np.random.normal(0, 0.001, size=(args.num_envs, 3))
            
            actions = np.zeros((args.num_envs, 8), dtype=np.float32)
            actions[:, :3] = current_pose[:, :3] + noise
            actions[:, 3:7] = current_pose[:, 3:7]
            actions[:, 7] = 1.0 # Open gripper
            
            t_start = time.time()
            obs, reward, done = env.step(actions)
            t_step = time.time() - t_start
            total_time += t_step
            
            if s % 10 == 0:
                print(f"Step {s}/{args.steps}: {t_step:.4f}s (FPS: {args.num_envs/t_step:.1f})")

        print(f"Total time for {args.steps} steps: {total_time:.2f}s")
        print(f"Average FPS: {(args.num_envs * args.steps) / total_time:.2f}")
        print(f"Average time per step (batch): {total_time / args.steps:.4f}s")

    finally:
        env.close()

if __name__ == "__main__":
    # 必须使用 spawn
    mp.set_start_method("spawn", force=True)
    main()
