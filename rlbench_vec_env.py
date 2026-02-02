#!/usr/bin/env python3
"""
Optimized RLBench Vectorized Environment for Multi-GPU Headless Training
"""

from __future__ import annotations

import os
import sys
import time
import argparse
import multiprocessing as mp
from multiprocessing import shared_memory
from multiprocessing.connection import Connection
import numpy as np
import cv2
import subprocess
import random
import atexit
import traceback

# =============================================================================
# Utils
# =============================================================================

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
    return _get_int_env(["LOCAL_RANK", "SLURM_LOCALID"], 0)

def _get_global_rank() -> int:
    return _get_int_env(["RANK", "SLURM_PROCID"], 0)

def _start_xvfb(display_str: str, max_tries: int = 20):
    """Start Xvfb on a specific display or find a free one."""
    if not display_str:
        raise ValueError("display_str required")

    # Ensure XDG_RUNTIME_DIR is set and unique per process to avoid Qt conflicts
    if "XDG_RUNTIME_DIR" not in os.environ:
        runtime_dir = f"/tmp/runtime-root-{os.getpid()}"
        os.environ["XDG_RUNTIME_DIR"] = runtime_dir
        os.makedirs(runtime_dir, exist_ok=True, mode=0o700)

    try:
        start_disp = int(str(display_str).lstrip(":"))
    except ValueError:
        start_disp = 10000

    last_err = None
    # Try a range of displays
    for off in range(max_tries):
        disp = start_disp + off
        disp_str = f":{disp}"
        sock_path = f"/tmp/.X11-unix/X{disp}"
        
        if os.path.exists(sock_path):
            continue

        # Start Xvfb
        cmd = ["Xvfb", disp_str, "-screen", "0", "1024x768x24", "-ac", "-nolisten", "tcp"]
        p = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        
        # Wait for it to start
        start_time = time.time()
        ok = False
        while time.time() - start_time < 10.0: # Wait up to 10s
            if p.poll() is not None:
                # Crashed
                err = p.stderr.read().decode("utf-8", errors="ignore") if p.stderr else "unknown"
                last_err = f"Xvfb crashed on {disp_str}: {err}"
                break
            if os.path.exists(sock_path):
                ok = True
                break
            time.sleep(0.1)
        
        if ok:
            return p, disp_str
        
        # Cleanup if failed
        try:
            p.terminate()
            p.wait(1.0)
        except:
            p.kill()

    raise RuntimeError(f"Could not start Xvfb after {max_tries} tries. Last error: {last_err}")

# =============================================================================
# Worker
# =============================================================================

def _worker_entry(rank: int, pipe: Connection, shm_info: dict, env_config: dict):
    xvfb_proc = None
    env = None
    try:
        # 1. Setup Display
        # Stagger start to reduce system load spikes (32 workers total)
        time.sleep(random.uniform(0.5, 5.0))
        
        local_rank = _get_local_rank()
        # Unique display base: 20000 + local_rank*1000 + worker_rank*10
        # This gives plenty of space between workers
        base_disp = 20000 + (local_rank * 1000) + (rank * 10)
        xvfb_proc, display = _start_xvfb(f":{base_disp}")
        os.environ["DISPLAY"] = display
        
        # Isolate CoppeliaSim home to avoid config corruption
        # /tmp/coppelia_home/rank_X/env_Y
        home_dir = f"/tmp/coppelia_home/proc_{os.getpid()}/env_{rank}"
        os.makedirs(home_dir, exist_ok=True)
        os.environ["HOME"] = home_dir

        print(f"[Worker {rank}] Starting Xvfb on {display}...", flush=True)

        # 2. Import RLBench (must be after DISPLAY set)
        from rlbench.environment import Environment
        from rlbench.action_modes.action_mode import MoveArmThenGripper
        from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaPlanning, EndEffectorPoseViaIK
        from rlbench.action_modes.gripper_action_modes import Discrete
        from rlbench.observation_config import ObservationConfig, CameraConfig
        from rlbench.utils import name_to_task_class

        # 3. Setup Environment
        img_size = env_config["image_size"]
        obs_config = ObservationConfig()
        obs_config.set_all(False)
        # 开启需要的相机
        obs_config.front_camera = CameraConfig(rgb=True, image_size=img_size)
        obs_config.wrist_camera = CameraConfig(rgb=True, image_size=img_size)
        obs_config.gripper_pose = True
        obs_config.gripper_open = True

        arm_mode_str = env_config.get("arm_mode", "planning")
        if arm_mode_str == "planning":
            arm_mode = EndEffectorPoseViaPlanning()
        else:
            arm_mode = EndEffectorPoseViaIK()

        action_mode = MoveArmThenGripper(arm_mode, Discrete())
        
        env = Environment(
            action_mode=action_mode,
            obs_config=obs_config,
            headless=True,
            robot_setup=env_config["robot_setup"]
        )
        print(f"[Worker {rank}] Launching RLBench...", flush=True)
        env.launch()
        
        task_class = name_to_task_class(env_config["task_name"])
        task_env = env.get_task(task_class)

        # 4. Connect Shared Memory
        shm_objs = {}
        arrays = {}
        for name, info in shm_info.items():
            shm = shared_memory.SharedMemory(name=info["name"])
            shm_objs[name] = shm
            full_arr = np.ndarray(info["shape"], dtype=np.dtype(info["dtype"]), buffer=shm.buf)
            arrays[name] = full_arr[rank]

        # 5. Loop
        print(f"[Worker {rank}] Ready.", flush=True)
        pipe.send("ready")
        
        done_flag = False
        
        while True:
            cmd, data = pipe.recv()
            
            if cmd == "reset":
                _, obs = task_env.reset()
                done_flag = False
                _write_obs(arrays, obs)
                pipe.send("done")
                
            elif cmd == "step":
                if done_flag:
                    arrays["done"][0] = True
                    arrays["reward"][0] = 0.0
                else:
                    obs, reward, term = task_env.step(data)
                    _write_obs(arrays, obs)
                    arrays["reward"][0] = reward
                    arrays["done"][0] = term
                    done_flag = bool(term)
                pipe.send("done")
                
            elif cmd == "close":
                break
                
    except Exception:
        # Print traceback immediately to stderr so it's captured in logs
        sys.stderr.write(f"\n[Worker {rank}] CRASHED:\n")
        traceback.print_exc(file=sys.stderr)
        
        # Add hints for common errors
        err_msg = traceback.format_exc()
        if "Handle" in err_msg and "does not exist" in err_msg:
            sys.stderr.write(
                "\n[Worker Hint] 'Handle ... does not exist' usually means the RLBench scene failed to load.\n"
                "Possible causes:\n"
                "1. COPPELIASIM_ROOT points to an incompatible version.\n"
                "2. Multiple CoppeliaSim instances are conflicting on the ZMQ Remote API port (default 23000).\n"
                "   Solution: Disable the 'ZMQ remote API' add-on in CoppeliaSim's installation folder (addOns.txt or remove the lua file).\n"
                "3. Headless rendering (Xvfb) failed to initialize OpenGL properly.\n"
            )
        
        sys.stderr.flush()
        try:
            pipe.send(("error", err_msg))
        except:
            pass
    finally:
        try:
            if env: env.shutdown()
        except:
            pass
        if xvfb_proc:
            xvfb_proc.terminate()

def _write_obs(arrays, obs):
    if hasattr(obs, "front_rgb"):
        arrays["front_rgb"][:] = obs.front_rgb
    if hasattr(obs, "wrist_rgb"):
        arrays["wrist_rgb"][:] = obs.wrist_rgb
    if hasattr(obs, "gripper_pose"):
        arrays["gripper_pose"][:] = obs.gripper_pose
    if hasattr(obs, "gripper_open"):
        arrays["gripper_open"][0] = float(obs.gripper_open)

# =============================================================================
# Main Vector Env
# =============================================================================

class RLBenchVectorEnv:
    def __init__(self, num_envs, task_name, image_size=(256, 256), robot_setup="panda", arm_mode="planning"):
        self.num_envs = num_envs
        self.shm_specs = {
            "front_rgb":    ((num_envs, image_size[0], image_size[1], 3), np.uint8),
            "wrist_rgb":    ((num_envs, image_size[0], image_size[1], 3), np.uint8),
            "gripper_pose": ((num_envs, 7), np.float32),
            "gripper_open": ((num_envs, 1), np.float32),
            "reward":       ((num_envs, 1), np.float32),
            "done":         ((num_envs, 1), np.bool_),
        }
        
        self.shm_objs = []
        self.shm_info = {}
        self.arrays = {}
        self.procs = []
        self.pipes = []
        
        # Register cleanup
        atexit.register(self.close)
        
        # Create Shared Memory
        for name, (shape, dtype) in self.shm_specs.items():
            size = int(np.prod(shape)) * np.dtype(dtype).itemsize
            shm = shared_memory.SharedMemory(create=True, size=size)
            self.shm_objs.append(shm)
            self.shm_info[name] = {"name": shm.name, "shape": shape, "dtype": np.dtype(dtype).str}
            self.arrays[name] = np.ndarray(shape, dtype=dtype, buffer=shm.buf)

        # Start Workers
        ctx = mp.get_context("spawn")
        env_config = {
            "task_name": task_name,
            "image_size": image_size,
            "robot_setup": robot_setup,
            "arm_mode": arm_mode
        }
        
        print(f"Starting {num_envs} workers...", flush=True)
        for i in range(num_envs):
            parent, child = ctx.Pipe()
            p = ctx.Process(target=_worker_entry, args=(i, child, self.shm_info, env_config))
            p.start()
            self.procs.append(p)
            self.pipes.append(parent)
            
        # Wait for ready
        for i, pipe in enumerate(self.pipes):
            msg = pipe.recv()
            if msg != "ready":
                if isinstance(msg, tuple) and msg[0] == "error":
                    raise RuntimeError(f"Worker {i} failed:\n{msg[1]}")
                raise RuntimeError(f"Worker {i} failed: {msg}")
        print("All workers ready.", flush=True)

    def reset(self):
        for pipe in self.pipes:
            pipe.send(("reset", None))
        for pipe in self.pipes:
            self._recv_ack(pipe)
        return self._get_obs()

    def step(self, actions):
        for i, pipe in enumerate(self.pipes):
            pipe.send(("step", actions[i]))
        for pipe in self.pipes:
            self._recv_ack(pipe)
        return self._get_obs(), self.arrays["reward"].copy(), self.arrays["done"].copy()

    def _recv_ack(self, pipe):
        msg = pipe.recv()
        if msg != "done":
            if isinstance(msg, tuple) and msg[0] == "error":
                raise RuntimeError(f"Worker error:\n{msg[1]}")
            raise RuntimeError(f"Unexpected worker msg: {msg}")

    def _get_obs(self):
        return {k: self.arrays[k].copy() for k in ["front_rgb", "wrist_rgb", "gripper_pose", "gripper_open"]}

    def close(self):
        # Unregister to avoid double call
        atexit.unregister(self.close)
        
        print("[RLBenchVecEnv] Closing...", flush=True)
        for pipe in self.pipes:
            try: pipe.send(("close", None))
            except: pass
        for pipe in self.pipes:
            try: pipe.close()
            except: pass
            
        for p in self.procs:
            p.join(timeout=5)
            if p.is_alive(): p.terminate()
            
        for shm in self.shm_objs:
            try: shm.close(); shm.unlink()
            except: pass
        print("[RLBenchVecEnv] Closed.", flush=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_envs", type=int, default=2)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--task", type=str, default="OpenDrawer")
    parser.add_argument("--robot_setup", type=str, default="panda")
    parser.add_argument("--image_size", type=int, nargs=2, default=[256, 256])
    parser.add_argument("--arm_mode", type=str, default="planning")
    # Ignore other args to prevent errors if user adds them back
    args, _ = parser.parse_known_args()
    
    print(f"COPPELIASIM_ROOT: {os.environ.get('COPPELIASIM_ROOT', 'NOT SET')}", flush=True)
    print(f"Initializing {args.num_envs} environments for task {args.task}...", flush=True)
    
    try:
        env = RLBenchVectorEnv(
            num_envs=args.num_envs, 
            task_name=args.task,
            image_size=tuple(args.image_size),
            robot_setup=args.robot_setup,
            arm_mode=args.arm_mode
        )

        print("Resetting...", flush=True)
        t0 = time.time()
        obs = env.reset()
        t_reset = time.time() - t0
        print(f"Reset done in {t_reset:.2f}s", flush=True)

        print(f"Running {args.steps} steps...", flush=True)
        
        total_time = 0.0
        for s in range(args.steps):
            actions = np.zeros((args.num_envs, 8), dtype=np.float32)
            actions[:, 7] = 1.0 # Open gripper
            
            t_start = time.time()
            env.step(actions)
            t_step = time.time() - t_start
            total_time += t_step
            
            if s % 10 == 0:
                print(f"Step {s}/{args.steps}: {t_step:.4f}s", flush=True)

        print(f"Total time: {total_time:.2f}s", flush=True)

    except Exception:
        print("\n" + "="*60, file=sys.stderr)
        print("ERROR IN MAIN PROCESS", file=sys.stderr)
        print("="*60, file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        print("="*60 + "\n", file=sys.stderr)
        sys.stderr.flush()
        raise
    finally:
        if 'env' in locals():
            env.close()

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
