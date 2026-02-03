#!/usr/bin/env python3
"""
RLBench Vectorized Environment for Headless Multi-Process / Torchrun
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

# Reduce CPU oversubscription when launching many env processes.
# This is especially important when using planning / linear algebra backends.
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

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
    vw_front = None
    vw_wrist = None
    try:
        # 1. Setup Display / Rendering backend
        # Stagger start to reduce system load spikes
        time.sleep(random.uniform(0.2, 2.0))

        render_backend = str(env_config.get("render_backend", "xvfb")).lower().strip()
        if render_backend not in {"xvfb", "egl", "external"}:
            render_backend = "xvfb"

        # Allow using external GPU-backed rendering by disabling per-process Xvfb.
        # Convention compatible with mp_rlbench_env_test.py: RLBENCH_PER_PROCESS_XVFB=0 disables Xvfb.
        per_process_xvfb = True
        if "RLBENCH_PER_PROCESS_XVFB" in os.environ:
            per_process_xvfb = str(os.environ["RLBENCH_PER_PROCESS_XVFB"]).strip() == "1"
        per_process_xvfb = bool(env_config.get("per_process_xvfb", per_process_xvfb))

        def _norm_opt_str(v):
            if v is None:
                return None
            s = str(v).strip()
            if s == "":
                return None
            if s.lower() in {"none", "null"}:
                return None
            return s

        # EGL mode: attempt GPU-backed headless rendering (no X server).
        # This only works if your container/host has NVIDIA drivers + EGL/GLVND correctly installed.
        if render_backend == "egl":
            per_process_xvfb = False
            # Qt/EGL hints.
            # IMPORTANT: do NOT use setdefault() here. If the parent environment sets e.g.
            # QT_QPA_PLATFORM=offscreen, Qt will load the wrong plugin and you may get:
            #   "This plugin does not support createPlatformOpenGLContext!"
            # Force a sane EGL-capable default.
            qpa = _norm_opt_str(env_config.get("qt_qpa_platform")) or "minimalegl"
            qt_opengl = _norm_opt_str(env_config.get("qt_opengl")) or "egl"
            qt_xcb = _norm_opt_str(env_config.get("qt_xcb_gl_integration")) or "none"
            pyopengl = _norm_opt_str(env_config.get("pyopengl_platform")) or "egl"

            os.environ["QT_QPA_PLATFORM"] = qpa
            os.environ["QT_OPENGL"] = qt_opengl
            # Avoid XCB integration when running without X
            os.environ["QT_XCB_GL_INTEGRATION"] = qt_xcb
            # Some OpenGL stacks respect this for EGL selection
            os.environ["PYOPENGL_PLATFORM"] = pyopengl
            # Ensure we don't accidentally bind to an X display
            os.environ.pop("DISPLAY", None)

        local_rank = _get_local_rank()

        # Optional GL/Qt hints (best-effort; actual GPU-backed rendering depends on system setup)
        qt_qpa = _norm_opt_str(env_config.get("qt_qpa_platform"))
        if qt_qpa:
            os.environ.setdefault("QT_QPA_PLATFORM", qt_qpa)
        glx_vendor = env_config.get("glx_vendor")
        if glx_vendor:
            os.environ.setdefault("__GLX_VENDOR_LIBRARY_NAME", str(glx_vendor))
        if "libgl_always_software" in env_config:
            os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", str(int(env_config.get("libgl_always_software") or 0)))

        if per_process_xvfb and render_backend == "xvfb":
            # Unique display base: 20000 + local_rank*1000 + worker_rank*10
            # Plenty of space between workers to avoid socket collisions.
            base_disp = 20000 + (local_rank * 1000) + (rank * 10)
            xvfb_proc, display = _start_xvfb(f":{base_disp}")
            os.environ["DISPLAY"] = display
        else:
            # External DISPLAY assignment. This does NOT start X.
            # Expected usage:
            # - render_backend=external: user starts GPU-backed Xorg servers (e.g. :0..:3) and we route workers.
            # - render_backend=egl: DISPLAY is typically unset; we keep this block for compatibility.
            display = None
            displays = env_config.get("displays")
            if isinstance(displays, (list, tuple)) and len(displays) > 0:
                display = str(displays[rank % len(displays)])
            else:
                display_base = int(env_config.get("display_base", 0))
                display_per_rank = bool(env_config.get("display_per_rank", True))
                display_per_worker = bool(env_config.get("display_per_worker", False))
                disp_idx = display_base
                if display_per_rank:
                    disp_idx += int(local_rank)
                if display_per_worker:
                    disp_idx += int(rank)
                display = f":{disp_idx}"
            if display and render_backend != "egl":
                os.environ.setdefault("DISPLAY", str(display))
        
        # Isolate CoppeliaSim home to avoid config corruption
        # /tmp/coppelia_home/rank_X/env_Y
        home_dir = f"/tmp/coppelia_home/proc_{os.getpid()}/env_{rank}"
        os.makedirs(home_dir, exist_ok=True)
        os.environ["HOME"] = home_dir

        if env_config.get("worker_log", False):
            if per_process_xvfb:
                print(f"[Worker {rank}] Starting Xvfb on {os.environ.get('DISPLAY')}...", flush=True)
            else:
                print(f"[Worker {rank}] render_backend={render_backend} DISPLAY={os.environ.get('DISPLAY', '')}", flush=True)

        # 2. Import RLBench (must be after DISPLAY set)
        from rlbench.environment import Environment
        from rlbench.action_modes.action_mode import MoveArmThenGripper
        from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaPlanning, EndEffectorPoseViaIK
        from rlbench.action_modes.gripper_action_modes import Discrete
        from rlbench.observation_config import ObservationConfig, CameraConfig
        from rlbench.utils import name_to_task_class
        from rlbench.backend.exceptions import InvalidActionError

        def _construct_action_mode(cls, **kwargs):
            """Construct an RLBench action-mode class with best-effort kwargs filtering.

            RLBench/pyrep versions differ slightly in constructor signatures.
            """
            try:
                import inspect

                sig = inspect.signature(cls)
                filtered = {k: v for k, v in kwargs.items() if k in sig.parameters}
                return cls(**filtered)
            except Exception:
                return cls()

        def _camel_to_snake(name):
            import re
            name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
            return re.sub('([a-z0-9])([A-Z])', r'\1_\2', name).lower()

        def _write_video(vw, obs_attr):
            if vw is not None and obs_attr is not None:
                bgr = cv2.cvtColor(obs_attr, cv2.COLOR_RGB2BGR)
                vw.write(bgr)

        # 3. Setup Environment
        img_size = env_config["image_size"]
        fps = env_config.get("fps", 20.0)
        output_dir = env_config.get("output_dir", None)
        video_all = env_config.get("video_all", False)
        no_video = env_config.get("no_video", False)
        enable_rgb_cfg = bool(env_config.get("enable_rgb", True))
        video_stride = max(1, int(env_config.get("video_stride", 1)))
        profile_timing = bool(env_config.get("profile_timing", False))
        rgb_video_only = bool(env_config.get("rgb_video_only", False))

        # Torchrun ranks (for controlling which process writes videos)
        global_rank = int(env_config.get("global_rank", 0))
        video_rank0_only = bool(env_config.get("video_rank0_only", False))

        # Decide whether this worker will record video.
        # We use this both to decide whether to open a VideoWriter, and (optionally)
        # to disable RGB cameras for non-video workers to reduce sim/render load.
        want_video = bool(
            output_dir
            and (not no_video)
            and enable_rgb_cfg
            and ((not video_rank0_only) or (global_rank == 0))
            and (rank == 0 or video_all)
        )

        # If enabled, only keep RGB cameras on for the workers that actually record video.
        # This can drastically reduce `task_env.step()` time when running many envs.
        enable_rgb = bool(enable_rgb_cfg and ((not rgb_video_only) or want_video))
        
        # Video recording requires RGB observations.
        if want_video:
            os.makedirs(output_dir, exist_ok=True)
            f_front = os.path.join(output_dir, f"env{rank}_front.mp4")
            f_wrist = os.path.join(output_dir, f"env{rank}_wrist.mp4")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            # cv2 uses (width, height)
            vw_front = cv2.VideoWriter(f_front, fourcc, fps, (img_size[1], img_size[0]))
            vw_wrist = cv2.VideoWriter(f_wrist, fourcc, fps, (img_size[1], img_size[0]))

        obs_config = ObservationConfig()
        obs_config.set_all(False)

        # Cameras are expensive. Keep them optional.
        if enable_rgb:
            obs_config.front_camera = CameraConfig(rgb=True, image_size=img_size)
            obs_config.wrist_camera = CameraConfig(rgb=True, image_size=img_size)
        else:
            obs_config.front_camera = CameraConfig(rgb=False, image_size=img_size)
            obs_config.wrist_camera = CameraConfig(rgb=False, image_size=img_size)

        obs_config.gripper_pose = True
        obs_config.gripper_open = True

        arm_mode_str = env_config.get("arm_mode", "ik")
        collision_checking = bool(env_config.get("collision_checking", False))
        if arm_mode_str == "planning":
            # Planning is significantly slower; prefer IK unless you truly need planning.
            arm_mode = _construct_action_mode(EndEffectorPoseViaPlanning, collision_checking=collision_checking)
        else:
            # IK is much faster and typically required for high-throughput rollouts.
            arm_mode = _construct_action_mode(EndEffectorPoseViaIK, collision_checking=collision_checking)

        action_mode = MoveArmThenGripper(arm_mode, Discrete())
        
        env = Environment(
            action_mode=action_mode,
            obs_config=obs_config,
            headless=True,
            robot_setup=env_config["robot_setup"]
        )
        if env_config.get("worker_log", False):
            print(f"[Worker {rank}] Launching RLBench...", flush=True)
        env.launch()

        # Disable real-time throttling so simulation runs as fast as possible.
        # In some setups CoppeliaSim defaults to real-time, which can cap throughput severely.
        try:
            from pyrep.backend import sim

            realtime = bool(env_config.get("realtime", False))
            if hasattr(sim, "simSetBoolParameter") and hasattr(sim, "sim_boolparam_realtime_simulation"):
                sim.simSetBoolParameter(sim.sim_boolparam_realtime_simulation, bool(realtime))

            idle_fps = int(env_config.get("idle_fps", 0))
            if hasattr(sim, "simSetInt32Parameter") and hasattr(sim, "sim_intparam_idle_fps"):
                sim.simSetInt32Parameter(sim.sim_intparam_idle_fps, int(idle_fps))
        except Exception:
            pass
        
        task_name = env_config["task_name"]
        if not task_name.islower():
            task_name = _camel_to_snake(task_name)
            
        task_class = name_to_task_class(task_name)
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
        if env_config.get("worker_log", False):
            print(f"[Worker {rank}] Ready.", flush=True)
        pipe.send("ready")
        
        done_flag = False
        # Used for video frame decimation (write every N env-steps)
        step_idx = 0
        
        while True:
            cmd, data = pipe.recv()
            
            if cmd == "reset":
                _, obs = task_env.reset()
                done_flag = False
                step_idx = 0
                _write_obs(arrays, obs)
                if vw_front and hasattr(obs, "front_rgb"): _write_video(vw_front, obs.front_rgb)
                if vw_wrist and hasattr(obs, "wrist_rgb"): _write_video(vw_wrist, obs.wrist_rgb)
                pipe.send("done")
                
            elif cmd == "step":
                t0 = time.perf_counter()
                t_toggle = 0.0
                t_sim = 0.0
                t_obs_write = 0.0
                t_video = 0.0
                cpu0 = time.process_time() if profile_timing else 0.0
                n_steps = 0
                if done_flag:
                    arrays["done"][0] = True
                    arrays["reward"][0] = 0.0
                else:
                    try:
                        if enable_rgb and (vw_front is not None or vw_wrist is not None):
                            want_frame = (step_idx % video_stride) == 0
                            try:
                                if profile_timing:
                                    _t = time.perf_counter()
                                obs_config.front_camera.rgb = want_frame
                                obs_config.wrist_camera.rgb = want_frame
                                if profile_timing:
                                    t_toggle += time.perf_counter() - _t
                            except Exception:
                                pass

                        _t = time.perf_counter()
                        obs, reward, term = task_env.step(data)
                        if profile_timing:
                            t_sim += time.perf_counter() - _t
                        n_steps = 1
                        step_idx += 1

                        if profile_timing:
                            _t = time.perf_counter()
                            _write_obs(arrays, obs)
                            t_obs_write += time.perf_counter() - _t
                        else:
                            _write_obs(arrays, obs)

                        if vw_front and hasattr(obs, "front_rgb"):
                            if profile_timing:
                                _t = time.perf_counter()
                                _write_video(vw_front, obs.front_rgb)
                                t_video += time.perf_counter() - _t
                            else:
                                _write_video(vw_front, obs.front_rgb)
                        if vw_wrist and hasattr(obs, "wrist_rgb"):
                            if profile_timing:
                                _t = time.perf_counter()
                                _write_video(vw_wrist, obs.wrist_rgb)
                                t_video += time.perf_counter() - _t
                            else:
                                _write_video(vw_wrist, obs.wrist_rgb)
                        arrays["reward"][0] = reward
                        arrays["done"][0] = term
                        done_flag = bool(term)
                    except InvalidActionError:
                        arrays["reward"][0] = 0.0
                        arrays["done"][0] = False
                t_exec = time.perf_counter() - t0
                if profile_timing:
                    t_cpu = float(time.process_time() - cpu0)
                    cpu_util = float(t_cpu / t_exec) if t_exec > 0 else 0.0
                    pipe.send(
                        (
                            "done",
                            {
                                "t_exec": float(t_exec),
                                "n_steps": int(n_steps),
                                "t_toggle": float(t_toggle),
                                "t_sim": float(t_sim),
                                "t_obs_write": float(t_obs_write),
                                "t_video": float(t_video),
                                "t_cpu": float(t_cpu),
                                "cpu_util": float(cpu_util),
                            },
                        )
                    )
                else:
                    pipe.send(("done", {"t_exec": t_exec}))
            
            elif cmd == "step_chunk":
                t0 = time.perf_counter()
                t_toggle = 0.0
                t_sim = 0.0
                t_obs_write = 0.0
                t_video = 0.0
                cpu0 = time.process_time() if profile_timing else 0.0
                # data is actions (K, 8)
                if done_flag:
                    arrays["reward_sum"][0] = 0.0
                    arrays["n_steps"][0] = 0
                    arrays["reward"][0] = 0.0
                    arrays["done"][0] = True
                    if profile_timing:
                        pipe.send(
                            (
                                "done",
                                {
                                    "t_exec": 0.0,
                                    "n_steps": 0,
                                    "t_toggle": 0.0,
                                    "t_sim": 0.0,
                                    "t_obs_write": 0.0,
                                    "t_video": 0.0,
                                    "t_cpu": 0.0,
                                    "cpu_util": 0.0,
                                },
                            )
                        )
                    else:
                        pipe.send(("done", {"t_exec": 0.0}))
                    continue
                
                actions = data
                reward_sum = 0.0
                n_steps = 0
                last_reward = 0.0
                
                obs = None
                for j in range(len(actions)):
                    try:
                        # Optional optimization: if RGB is enabled, render only when we need a frame.
                        # Note: dynamic toggling is version-dependent; keep it best-effort.
                        if enable_rgb and (vw_front is not None or vw_wrist is not None):
                            want_frame = (step_idx % video_stride) == 0
                            try:
                                if profile_timing:
                                    _t = time.perf_counter()
                                obs_config.front_camera.rgb = want_frame
                                obs_config.wrist_camera.rgb = want_frame
                                if profile_timing:
                                    t_toggle += time.perf_counter() - _t
                            except Exception:
                                pass

                        _t = time.perf_counter()
                        obs, reward, term = task_env.step(actions[j])
                        if profile_timing:
                            t_sim += time.perf_counter() - _t
                        step_idx += 1
                        last_reward = float(reward)
                        reward_sum += float(reward)
                        n_steps += 1
                        
                        if vw_front and hasattr(obs, "front_rgb"):
                            if profile_timing:
                                _t = time.perf_counter()
                                _write_video(vw_front, obs.front_rgb)
                                t_video += time.perf_counter() - _t
                            else:
                                _write_video(vw_front, obs.front_rgb)
                        if vw_wrist and hasattr(obs, "wrist_rgb"):
                            if profile_timing:
                                _t = time.perf_counter()
                                _write_video(vw_wrist, obs.wrist_rgb)
                                t_video += time.perf_counter() - _t
                            else:
                                _write_video(vw_wrist, obs.wrist_rgb)
                        if bool(term):
                            done_flag = True
                            break
                    except InvalidActionError:
                        last_reward = 0.0
                        n_steps += 1
                        pass
                
                if obs is not None:
                    if profile_timing:
                        _t = time.perf_counter()
                        _write_obs(arrays, obs)
                        t_obs_write += time.perf_counter() - _t
                    else:
                        _write_obs(arrays, obs)
                
                arrays["reward_sum"][0] = float(reward_sum)
                arrays["n_steps"][0] = int(n_steps)
                arrays["reward"][0] = float(last_reward)
                arrays["done"][0] = bool(done_flag)
                t_exec = time.perf_counter() - t0
                if profile_timing:
                    t_cpu = float(time.process_time() - cpu0)
                    cpu_util = float(t_cpu / t_exec) if t_exec > 0 else 0.0
                    pipe.send(
                        (
                            "done",
                            {
                                "t_exec": float(t_exec),
                                "n_steps": int(n_steps),
                                "t_toggle": float(t_toggle),
                                "t_sim": float(t_sim),
                                "t_obs_write": float(t_obs_write),
                                "t_video": float(t_video),
                                "t_cpu": float(t_cpu),
                                "cpu_util": float(cpu_util),
                            },
                        )
                    )
                else:
                    pipe.send(("done", {"t_exec": t_exec}))
                
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
                "1. COPPELIASIM_ROOT points to an incompatible version (Must be 4.1.0).\n"
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
        if vw_front: vw_front.release()
        if vw_wrist: vw_wrist.release()
        if xvfb_proc:
            xvfb_proc.terminate()

def _write_obs(arrays, obs):
    # RLBench may return RGB images as None when the corresponding camera is disabled.
    # Guard against assigning None into numpy arrays.
    if hasattr(obs, "front_rgb") and (obs.front_rgb is not None):
        arrays["front_rgb"][:] = obs.front_rgb
    if hasattr(obs, "wrist_rgb") and (obs.wrist_rgb is not None):
        arrays["wrist_rgb"][:] = obs.wrist_rgb
    if hasattr(obs, "gripper_pose"):
        arrays["gripper_pose"][:] = obs.gripper_pose
    if hasattr(obs, "gripper_open"):
        arrays["gripper_open"][0] = float(obs.gripper_open)

# =============================================================================
# Main Vector Env
# =============================================================================

class RLBenchVectorEnv:
    def __init__(
        self,
        num_envs,
        task_name,
        image_size=(256, 256),
        robot_setup="panda",
        arm_mode="ik",
        fps=20.0,
        output_dir=None,
        video_all=False,
        no_video=False,
        enable_rgb=True,
        collision_checking=False,
        realtime=False,
        idle_fps=0,
        verbose=True,
        worker_log=False,
        render_backend="xvfb",
        per_process_xvfb=True,
        displays=None,
        display_base=0,
        display_per_rank=True,
        display_per_worker=False,
        qt_qpa_platform=None,
        qt_opengl=None,
        qt_xcb_gl_integration=None,
        pyopengl_platform=None,
        glx_vendor=None,
        libgl_always_software=None,
        video_stride=1,
        global_rank=0,
        video_rank0_only=False,
        profile_timing: bool = False,
        rgb_video_only: bool = False,
    ):
        self.num_envs = num_envs
        self.verbose = bool(verbose)
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
            "arm_mode": arm_mode,
            "fps": fps,
            "output_dir": output_dir,
            "video_all": video_all,
            "no_video": no_video,
            "enable_rgb": enable_rgb,
            "collision_checking": collision_checking,
            "realtime": realtime,
            "idle_fps": idle_fps,
            "worker_log": bool(worker_log),
            "render_backend": str(render_backend),
            "per_process_xvfb": bool(per_process_xvfb),
            "displays": displays,
            "display_base": int(display_base),
            "display_per_rank": bool(display_per_rank),
            "display_per_worker": bool(display_per_worker),
            "qt_qpa_platform": qt_qpa_platform,
            "qt_opengl": qt_opengl,
            "qt_xcb_gl_integration": qt_xcb_gl_integration,
            "pyopengl_platform": pyopengl_platform,
            "glx_vendor": glx_vendor,
            "libgl_always_software": libgl_always_software,
            "video_stride": int(video_stride),
            "global_rank": int(global_rank),
            "video_rank0_only": bool(video_rank0_only),
            "profile_timing": bool(profile_timing),
            "rgb_video_only": bool(rgb_video_only),
        }
        
        if self.verbose:
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
        if self.verbose:
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
        stats = []
        for pipe in self.pipes:
            stats.append(self._recv_ack(pipe))
        return self._get_obs(), self.arrays["reward"].copy(), self.arrays["done"].copy(), stats

    def step_chunk(self, actions_chunk):
        for i, pipe in enumerate(self.pipes):
            pipe.send(("step_chunk", actions_chunk[i]))
        stats = []
        for pipe in self.pipes:
            stats.append(self._recv_ack(pipe))
        return self._get_obs(), self.arrays["reward_sum"].copy(), self.arrays["done"].copy(), self.arrays["n_steps"].copy(), stats

    def _recv_ack(self, pipe):
        msg = pipe.recv()
        if isinstance(msg, tuple) and msg[0] == "done":
            # msg[1] is stats dict
            return msg[1] if len(msg) > 1 else {}
        elif msg == "done":
            return {}
        else:
            if isinstance(msg, tuple) and msg[0] == "error":
                raise RuntimeError(f"Worker error:\n{msg[1]}")
            raise RuntimeError(f"Unexpected worker msg: {msg}")

    def _get_obs(self):
        return {k: self.arrays[k].copy() for k in ["front_rgb", "wrist_rgb", "gripper_pose", "gripper_open"]}

    def close(self):
        # Unregister to avoid double call
        atexit.unregister(self.close)
        
        if self.verbose:
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
        if self.verbose:
            print("[RLBenchVecEnv] Closed.", flush=True)

def _build_action_from_obs(obs_dict, idx, rng, pos_noise_std=0.01, gripper_close_prob=0.05):
    gp = obs_dict["gripper_pose"][idx] # (7,)
    pos = gp[:3]
    quat = gp[3:7]
    
    pos_noisy = pos + rng.normal(loc=0.0, scale=pos_noise_std, size=(3,)).astype(np.float32)
    
    gripper = 0.0 if rng.random() < gripper_close_prob else 1.0
    
    return np.concatenate([pos_noisy, quat, [gripper]]).astype(np.float32)

def main():
    parser = argparse.ArgumentParser()
    # Support both old and new args
    parser.add_argument("--num_envs", type=int, default=2)
    parser.add_argument(
        "--mode",
        type=str,
        default="bench",
        choices=["bench", "debug", "train"],
        help="bench: throughput benchmark; debug: save videos; train: minimal end-to-end torch training loop",
    )
    parser.add_argument(
        "--num_envs_total",
        type=int,
        default=None,
        help="Total envs across all ranks when using torchrun (overrides --num_envs per rank)",
    )
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--max_steps", type=int, default=100)
    parser.add_argument("--task", type=str, default="OpenDrawer", dest="task_class")
    parser.add_argument("--task_class", type=str, default="OpenDrawer")
    parser.add_argument("--robot_setup", type=str, default="panda")
    parser.add_argument("--image_size", type=int, nargs=2, default=[256, 256])
    # Default to IK for throughput. Planning is dramatically slower.
    parser.add_argument("--arm_mode", type=str, default="ik", choices=["ik", "planning"])
    parser.add_argument("--action_chunk", type=int, default=1)
    parser.add_argument("--fps", type=float, default=20.0)
    # NOTE: default None to avoid accidental video/IO overhead during benchmarking.
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--pos_noise_std", type=float, default=0.01)
    parser.add_argument("--gripper_close_prob", type=float, default=0.05)
    parser.add_argument("--video_all", action="store_true", help="Record video for all environments (default: only env 0)")
    parser.add_argument("--no_video", action="store_true", help="Disable video recording completely")
    parser.add_argument(
        "--video_rank0_only",
        action="store_true",
        help="When using torchrun, record videos only on global rank 0 (reduces overhead)",
    )
    parser.add_argument(
        "--video_stride",
        type=int,
        default=1,
        help="Write every Nth frame to the video (reduces encode load). Only applies when video is enabled.",
    )
    parser.add_argument(
        "--rgb_video_only",
        action="store_true",
        help=(
            "If video is enabled, only the workers that actually record videos keep RGB cameras enabled. "
            "Non-video workers run with RGB disabled to reduce sim/render time. "
            "(Has no effect if --no_rgb is set.)"
        ),
    )
    parser.add_argument("--no_rgb", action="store_true", help="Disable RGB observations (major speedup)")
    parser.add_argument("--collision_checking", action="store_true", help="Enable collision checking in action mode (slower)")
    parser.add_argument("--realtime", action="store_true", help="Run CoppeliaSim in real-time (slower). Default: as-fast-as-possible")
    parser.add_argument("--idle_fps", type=int, default=0, help="CoppeliaSim idle FPS (0 is fastest)")
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Throughput preset: --arm_mode ik --no_rgb --no_video --idle_fps 0 (overrides some flags)",
    )
    parser.add_argument(
        "--ddp_bind_gpu",
        action="store_true",
        help="If launched with torchrun, set CUDA_VISIBLE_DEVICES=LOCAL_RANK for each rank (helps GPU isolation)",
    )
    parser.add_argument(
        "--render_backend",
        type=str,
        default="xvfb",
        choices=["xvfb", "egl", "external"],
        help="Rendering backend: xvfb (default), egl (no X), external (use existing DISPLAY)",
    )
    parser.add_argument(
        "--per_process_xvfb",
        type=int,
        default=1,
        help="1=spawn per-worker Xvfb (default). 0=do not spawn Xvfb (use EGL or external DISPLAY)",
    )
    parser.add_argument(
        "--display_base",
        type=int,
        default=0,
        help="When --render_backend external: base DISPLAY index (e.g. 0 -> :0)",
    )
    parser.add_argument(
        "--display_per_rank",
        type=int,
        default=1,
        help="When --render_backend external: add LOCAL_RANK to DISPLAY index (default 1)",
    )
    parser.add_argument(
        "--display_per_worker",
        type=int,
        default=0,
        help="When --render_backend external: also add worker id to DISPLAY index (default 0)",
    )
    parser.add_argument("--qt_qpa_platform", type=str, default=None, help="Set QT_QPA_PLATFORM in workers")
    parser.add_argument("--qt_opengl", type=str, default=None, help="Set QT_OPENGL in workers (e.g. egl)")
    parser.add_argument("--qt_xcb_gl_integration", type=str, default=None, help="Set QT_XCB_GL_INTEGRATION in workers")
    parser.add_argument("--pyopengl_platform", type=str, default=None, help="Set PYOPENGL_PLATFORM in workers (e.g. egl)")
    parser.add_argument(
        "--log_all_ranks",
        action="store_true",
        help="Print benchmark logs from all torchrun ranks (default: only rank 0)",
    )
    parser.add_argument(
        "--worker_log",
        action="store_true",
        help="Print per-worker startup logs (very noisy for many envs)",
    )

    parser.add_argument(
        "--profile_timing",
        action="store_true",
        help=(
            "Collect a more detailed timing breakdown per env-step/chunk (worker-side: sim/obs-copy/video/toggle/cpu_util; "
            "main-side: action-build vs env-call). Adds small overhead; intended for profiling/optimization."
        ),
    )

    # Minimal training loop settings (mode=train)
    parser.add_argument("--train_updates", type=int, default=10, help="Number of optimizer updates")
    parser.add_argument("--train_lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--train_hidden", type=int, default=256, help="MLP hidden size")
    parser.add_argument("--train_log_every", type=int, default=1, help="Log every N updates")
    parser.add_argument("--train_ckpt", type=str, default=None, help="Optional checkpoint path (rank0 only)")
    
    args, _ = parser.parse_known_args()

    # Torchrun / DDP info (used for multi-GPU launches).
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = _get_local_rank()

    # Mode presets (applied before --fast so they can take precedence).
    if args.mode == "debug":
        # Debug mode is intended for visual inspection; default to saving videos.
        if args.output_dir is None:
            args.output_dir = "rlbench_vec_env_videos"
        # Keep encoding load reasonable by default.
        if int(args.video_stride) == 1:
            args.video_stride = 4
        # In multi-rank runs, write videos only from global rank 0 by default.
        if world_size > 1 and (not bool(getattr(args, "video_rank0_only", False))):
            args.video_rank0_only = True
        # If someone accidentally passes --fast with debug, keep visuals.
        args.fast = False

    elif args.mode == "train":
        # Training mode is intended to be end-to-end (env + torch forward/backward), not visuals.
        args.output_dir = None
        args.no_video = True
        args.no_rgb = True
        args.video_all = False
        # Prefer throughput settings.
        args.arm_mode = "ik"
        args.collision_checking = False
        args.realtime = False
        args.idle_fps = 0
        # If someone passes --fast with train, it's redundant.
        args.fast = False

    # Throughput preset (aimed at fast rollouts, not visuals).
    if getattr(args, "fast", False):
        args.arm_mode = "ik"
        args.no_rgb = True
        args.no_video = True
        args.collision_checking = False
        args.realtime = False
        args.idle_fps = 0
    verbose = (world_size == 1) or bool(getattr(args, "log_all_ranks", False)) or (rank == 0)
    if world_size > 1 and verbose:
        print(f"[DDP] world_size={world_size} rank={rank} local_rank={local_rank}", flush=True)

    # Optionally bind each rank to a single GPU.
    # CoppeliaSim rendering (OpenGL) is not always affected by CUDA_VISIBLE_DEVICES,
    # but isolating ranks is still a good practice for GPU-backed pipelines.
    if world_size > 1 and getattr(args, "ddp_bind_gpu", False):
        # Force binding; setdefault() is too weak when the parent env already exported CUDA_VISIBLE_DEVICES.
        os.environ["CUDA_VISIBLE_DEVICES"] = str(local_rank)

    # If user specified total envs (e.g. 64) and launched with torchrun,
    # split them across ranks so total env count stays constant.
    if args.num_envs_total is not None and world_size > 1:
        total = int(args.num_envs_total)
        base = total // world_size
        extra = total % world_size
        args.num_envs = base + (1 if rank < extra else 0)
        if rank == 0:
            dist = [base + (1 if r < extra else 0) for r in range(world_size)]
            print(f"[DDP] num_envs_total={total} -> envs_per_rank={dist} (sum={sum(dist)})", flush=True)
        if verbose:
            print(f"[DDP] num_envs_this_rank={args.num_envs}", flush=True)
    elif world_size > 1 and rank == 0:
        # Common pitfall: --num_envs is PER RANK; total envs = num_envs * world_size
        print(f"[DDP] WARNING: --num_envs is per-rank. Total envs = {int(args.num_envs) * int(world_size)}", flush=True)
    
    # Handle aliases.
    # NOTE: Some older copies of this script referenced args.steps without
    # defining it in argparse (causing AttributeError). Keep this robust by
    # using getattr().
    steps_arg = int(getattr(args, "steps", 100))
    if args.max_steps != 100 and steps_arg == 100:
        pass  # max_steps set
    elif steps_arg != 100:
        args.max_steps = steps_arg
        
    # Auto-configure CoppeliaSim environment
    coppelia_root = os.environ.get("COPPELIASIM_ROOT")
    if not coppelia_root:
        # Try default location
        default_root = os.path.expanduser("~/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04")
        if os.path.isdir(default_root):
            coppelia_root = default_root
            os.environ["COPPELIASIM_ROOT"] = coppelia_root
            print(f"Auto-detected COPPELIASIM_ROOT: {coppelia_root}", flush=True)

    if coppelia_root and os.path.isdir(coppelia_root):
        ld_path = os.environ.get("LD_LIBRARY_PATH", "")
        if coppelia_root not in ld_path:
            print(f"Auto-adding {coppelia_root} to LD_LIBRARY_PATH and restarting...", flush=True)
            
            new_ld_path = f"{ld_path}:{coppelia_root}" if ld_path else coppelia_root
            os.environ["LD_LIBRARY_PATH"] = new_ld_path
            
            if "QT_QPA_PLATFORM_PLUGIN_PATH" not in os.environ:
                 os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = coppelia_root
            
            try:
                os.execv(sys.executable, [sys.executable] + sys.argv)
            except Exception as e:
                print(f"Failed to restart process: {e}", flush=True)

    if verbose:
        print(f"COPPELIASIM_ROOT: {os.environ.get('COPPELIASIM_ROOT', 'NOT SET')}", flush=True)
        print(f"Initializing {args.num_envs} environments for task {args.task_class}...", flush=True)
    
    try:
        # Mode defaults
        if args.mode == "debug":
            # Debug mode is meant to save videos; if user didn't pass an output_dir,
            # use a sensible default.
            if args.output_dir is None:
                args.output_dir = "rlbench_vec_env_videos"
            # Debug mode should render unless user explicitly disabled RGB/video.
        elif args.mode == "train":
            # Training mode defaults to throughput unless user explicitly enables visuals.
            # (Users can still override by passing --output_dir / removing --no_rgb etc.)
            pass

        env = RLBenchVectorEnv(
            num_envs=args.num_envs, 
            task_name=args.task_class,
            image_size=tuple(args.image_size),
            robot_setup=args.robot_setup,
            arm_mode=args.arm_mode,
            fps=args.fps,
            output_dir=args.output_dir,
            video_all=args.video_all,
            no_video=args.no_video,
            video_stride=int(args.video_stride),
            enable_rgb=(not args.no_rgb),
            collision_checking=args.collision_checking,
            realtime=args.realtime,
            idle_fps=args.idle_fps,
            verbose=verbose,
            worker_log=bool(getattr(args, "worker_log", False)),
            render_backend=str(getattr(args, "render_backend", "xvfb")),
            per_process_xvfb=bool(int(getattr(args, "per_process_xvfb", 1))),
            display_base=int(getattr(args, "display_base", 0)),
            display_per_rank=bool(int(getattr(args, "display_per_rank", 1))),
            display_per_worker=bool(int(getattr(args, "display_per_worker", 0))),
            qt_qpa_platform=getattr(args, "qt_qpa_platform", None),
            qt_opengl=getattr(args, "qt_opengl", None),
            qt_xcb_gl_integration=getattr(args, "qt_xcb_gl_integration", None),
            pyopengl_platform=getattr(args, "pyopengl_platform", None),
            global_rank=int(rank),
            video_rank0_only=bool(getattr(args, "video_rank0_only", False)),
            profile_timing=bool(getattr(args, "profile_timing", False)),
            rgb_video_only=bool(getattr(args, "rgb_video_only", False)),
        )

        def _run_env_loop(*, tag: str):
            if verbose:
                print("Resetting...", flush=True)
            t0 = time.time()
            obs = env.reset()
            t_reset = time.time() - t0
            if verbose:
                print(f"Reset done in {t_reset:.2f}s", flush=True)

            if verbose:
                print(f"Running {args.max_steps} steps (chunk={args.action_chunk})...", flush=True)

            rng = np.random.default_rng(seed=0 + rank)
            total_time = 0.0
            s = 0

            chunk_latencies = []
            chunk_exec_latencies = []
            chunk_ipc_latencies = []

            # Optional profiling breakdown
            chunk_build_latencies = []
            chunk_envcall_latencies = []
            chunk_sim_latencies = []
            chunk_obs_latencies = []
            chunk_video_latencies = []
            chunk_toggle_latencies = []
            chunk_other_exec_latencies = []
            chunk_sim_step_latencies = []
            chunk_cpu_util_latencies = []

            done = None
            while s < args.max_steps:
                k = min(args.action_chunk, args.max_steps - s)

                t_start = time.perf_counter()
                t_build0 = time.perf_counter()

                if k == 1:
                    actions = np.zeros((args.num_envs, 8), dtype=np.float32)
                    for i in range(args.num_envs):
                        actions[i] = _build_action_from_obs(obs, i, rng, args.pos_noise_std, args.gripper_close_prob)

                    t_build = time.perf_counter() - t_build0
                    t_env0 = time.perf_counter()
                    obs, reward, done, stats = env.step(actions)
                    t_envcall = time.perf_counter() - t_env0
                else:
                    actions_chunk = np.zeros((args.num_envs, k, 8), dtype=np.float32)
                    for i in range(args.num_envs):
                        gp = obs["gripper_pose"][i]
                        pos = gp[:3]
                        quat = gp[3:7]

                        pos_seq = pos + rng.normal(0, args.pos_noise_std, size=(k, 3)).astype(np.float32)
                        quat_seq = np.tile(quat, (k, 1))
                        grip_seq = (rng.random(size=(k, 1)) >= args.gripper_close_prob).astype(np.float32)

                        actions_chunk[i] = np.concatenate([pos_seq, quat_seq, grip_seq], axis=1)

                    t_build = time.perf_counter() - t_build0
                    t_env0 = time.perf_counter()
                    obs, reward, done, n_steps, stats = env.step_chunk(actions_chunk)
                    t_envcall = time.perf_counter() - t_env0

                t_step = time.perf_counter() - t_start
                total_time += t_step

                # Stats
                exec_times = [st.get("t_exec", 0.0) for st in stats]
                max_exec = max(exec_times) if exec_times else 0.0
                ipc_time = t_step - max_exec

                if bool(getattr(args, "profile_timing", False)):
                    # Choose the env on the critical path (max worker exec time) and read its breakdown.
                    if len(exec_times) > 0:
                        idx_max = int(np.argmax(np.asarray(exec_times, dtype=np.float64)))
                        st_max = stats[idx_max] if idx_max < len(stats) else {}
                    else:
                        idx_max = -1
                        st_max = {}

                    exec_arr = np.asarray(exec_times, dtype=np.float64) if len(exec_times) > 0 else np.asarray([], dtype=np.float64)
                    sim_arr = np.asarray([st.get("t_sim", 0.0) for st in stats], dtype=np.float64) if len(stats) > 0 else np.asarray([], dtype=np.float64)
                    cpu_util_arr = np.asarray([st.get("cpu_util", 0.0) for st in stats], dtype=np.float64) if len(stats) > 0 else np.asarray([], dtype=np.float64)

                    if exec_arr.size > 0:
                        exec_p50, exec_p90, exec_p99 = np.percentile(exec_arr, [50, 90, 99]).tolist()
                    else:
                        exec_p50 = exec_p90 = exec_p99 = 0.0
                    if sim_arr.size > 0:
                        sim_p50, sim_p90, sim_p99 = np.percentile(sim_arr, [50, 90, 99]).tolist()
                    else:
                        sim_p50 = sim_p90 = sim_p99 = 0.0
                    if cpu_util_arr.size > 0:
                        cu_p50, cu_p90, cu_p99 = np.percentile(cpu_util_arr, [50, 90, 99]).tolist()
                    else:
                        cu_p50 = cu_p90 = cu_p99 = 0.0

                    t_sim = float(st_max.get("t_sim", 0.0))
                    t_obs = float(st_max.get("t_obs_write", 0.0))
                    t_vid = float(st_max.get("t_video", 0.0))
                    t_tog = float(st_max.get("t_toggle", 0.0))
                    t_cpu = float(st_max.get("t_cpu", 0.0))
                    cpu_util = float(st_max.get("cpu_util", 0.0))
                    t_other = max(0.0, float(max_exec) - (t_sim + t_obs + t_vid + t_tog))

                    n_steps_crit = int(st_max.get("n_steps", 0))
                    if n_steps_crit <= 0:
                        # Fallback (shouldn't normally be needed since workers now return n_steps)
                        n_steps_crit = int(k)
                    sim_per_step = float(t_sim / n_steps_crit) if n_steps_crit > 0 else 0.0

                    chunk_build_latencies.append(float(t_build))
                    chunk_envcall_latencies.append(float(t_envcall))
                    chunk_sim_latencies.append(float(t_sim))
                    chunk_obs_latencies.append(float(t_obs))
                    chunk_video_latencies.append(float(t_vid))
                    chunk_toggle_latencies.append(float(t_tog))
                    chunk_other_exec_latencies.append(float(t_other))
                    chunk_sim_step_latencies.append(float(sim_per_step))
                    chunk_cpu_util_latencies.append(float(cpu_util))

                chunk_latencies.append(t_step)
                chunk_exec_latencies.append(max_exec)
                chunk_ipc_latencies.append(ipc_time)

                if verbose:
                    pfx = f"[rank {rank}] " if world_size > 1 else ""
                    if bool(getattr(args, "profile_timing", False)):
                        print(
                            f"{pfx}{tag} Chunk {s}/{args.max_steps}: total={t_step:.3f}s "
                            f"build={t_build:.3f}s env_call={t_envcall:.3f}s "
                            f"exec={max_exec:.3f}s(sim={t_sim:.3f}s obs={t_obs:.3f}s vid={t_vid:.3f}s toggle={t_tog:.3f}s other={t_other:.3f}s "
                            f"cpu={t_cpu:.3f}s util={cpu_util*100:.0f}% sim/step={sim_per_step:.3f}s) "
                            f"dist_exec(p50/p90/p99)={exec_p50:.3f}/{exec_p90:.3f}/{exec_p99:.3f}s "
                            f"dist_sim(p50/p90/p99)={sim_p50:.3f}/{sim_p90:.3f}/{sim_p99:.3f}s "
                            f"dist_cpu_util(p50/p90/p99)={cu_p50*100:.0f}/{cu_p90*100:.0f}/{cu_p99*100:.0f}% "
                            f"ipc={ipc_time:.3f}s",
                            flush=True,
                        )
                    else:
                        print(
                            f"{pfx}{tag} Chunk {s}/{args.max_steps}: total={t_step:.3f}s exec={max_exec:.3f}s ipc={ipc_time:.3f}s",
                            flush=True,
                        )

                s += k

            if verbose:
                pfx = f"[rank {rank}] " if world_size > 1 else ""
                print(f"{pfx}{tag} Total time: {total_time:.2f}s", flush=True)
                print(f"{pfx}{tag} Avg Step: {np.mean(chunk_latencies):.3f}s", flush=True)
                print(f"{pfx}{tag} Avg Exec: {np.mean(chunk_exec_latencies):.3f}s", flush=True)
                print(f"{pfx}{tag} Avg IPC:  {np.mean(chunk_ipc_latencies):.3f}s", flush=True)

                if bool(getattr(args, "profile_timing", False)) and len(chunk_build_latencies) > 0:
                    print(
                        f"{pfx}{tag} Avg Build: {np.mean(chunk_build_latencies):.3f}s  Avg EnvCall: {np.mean(chunk_envcall_latencies):.3f}s",
                        flush=True,
                    )
                    print(
                        f"{pfx}{tag} Avg Exec Breakdown (critical env): sim={np.mean(chunk_sim_latencies):.3f}s "
                        f"obs={np.mean(chunk_obs_latencies):.3f}s vid={np.mean(chunk_video_latencies):.3f}s "
                        f"toggle={np.mean(chunk_toggle_latencies):.3f}s other={np.mean(chunk_other_exec_latencies):.3f}s",
                        flush=True,
                    )
                    print(
                        f"{pfx}{tag} Avg Sim/Step (critical env): {np.mean(chunk_sim_step_latencies):.3f}s  "
                        f"Avg CPU util (critical env): {np.mean(chunk_cpu_util_latencies)*100:.0f}%",
                        flush=True,
                    )

                # RLBench's `term` is typically True when the task succeeds.
                # We expose this as a simple "success rate" proxy here.
                try:
                    done_flat = np.asarray(done).reshape(-1)
                    success_rate = float(done_flat.mean()) if done_flat.size > 0 else 0.0
                    print(
                        f"{pfx}{tag} Success rate (done==True): {success_rate*100:.1f}% ({int(done_flat.sum())}/{int(done_flat.size)})",
                        flush=True,
                    )
                except Exception:
                    pass

                if args.output_dir and (not args.no_video) and (not args.no_rgb):
                    print(f"{pfx}{tag} Saved videos to {args.output_dir}", flush=True)

            return obs

        if args.mode in {"bench", "debug"}:
            _run_env_loop(tag="[bench]")
        elif args.mode == "train":
            # Minimal end-to-end training loop.
            # This is not meant to be a strong RL algorithm; it's a debugging harness that:
            # - runs env rollouts
            # - runs a torch model forward/backward
            # - optionally uses DDP if launched with torchrun
            try:
                import torch
                import torch.nn as nn
                import torch.optim as optim

                use_cuda = torch.cuda.is_available()
                device = torch.device("cuda", 0) if use_cuda else torch.device("cpu")

                ddp_inited = False
                if world_size > 1 and hasattr(torch, "distributed"):
                    import torch.distributed as dist

                    if not dist.is_initialized():
                        backend = "nccl" if use_cuda else "gloo"
                        dist.init_process_group(backend=backend, init_method="env://")
                        ddp_inited = True

                obs_dim = 8  # gripper_pose(7) + gripper_open(1)
                # Predict delta-pos(3) and gripper logit(1)
                out_dim = 4

                class Policy(nn.Module):
                    def __init__(self, h: int):
                        super().__init__()
                        self.net = nn.Sequential(
                            nn.Linear(obs_dim, h),
                            nn.ReLU(),
                            nn.Linear(h, h),
                            nn.ReLU(),
                            nn.Linear(h, out_dim),
                        )

                    def forward(self, x):
                        return self.net(x)

                policy = Policy(int(args.train_hidden)).to(device)

                if world_size > 1:
                    from torch.nn.parallel import DistributedDataParallel as DDP

                    policy = DDP(policy, device_ids=[0] if use_cuda else None)

                opt = optim.Adam(policy.parameters(), lr=float(args.train_lr))

                # Training loop
                rng = np.random.default_rng(seed=1234 + rank)
                obs = env.reset()

                for upd in range(int(args.train_updates)):
                    # Build a batch from current obs
                    gp = obs["gripper_pose"].astype(np.float32)  # (N,7)
                    go = obs["gripper_open"].astype(np.float32)  # (N,1)
                    obs_low = np.concatenate([gp, go], axis=1)  # (N,8)

                    # Target action (same as benchmark heuristic)
                    target_actions = np.zeros((args.num_envs, 8), dtype=np.float32)
                    for i in range(args.num_envs):
                        target_actions[i] = _build_action_from_obs(obs, i, rng, args.pos_noise_std, args.gripper_close_prob)

                    # Targets: delta position and gripper
                    delta_pos_tgt = target_actions[:, :3] - gp[:, :3]
                    gripper_tgt = target_actions[:, 7:8]

                    x = torch.from_numpy(obs_low).to(device)
                    y_delta = torch.from_numpy(delta_pos_tgt).to(device)
                    y_grip = torch.from_numpy(gripper_tgt).to(device)

                    pred = policy(x)
                    pred_delta = pred[:, :3]
                    pred_grip_logit = pred[:, 3:4]

                    loss_delta = torch.mean((pred_delta - y_delta) ** 2)
                    loss_grip = torch.nn.functional.binary_cross_entropy_with_logits(pred_grip_logit, y_grip)
                    loss = loss_delta + loss_grip

                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    opt.step()

                    # Step env using the model outputs
                    with torch.no_grad():
                        # Clamp delta for stability
                        delta = torch.tanh(pred_delta).cpu().numpy().astype(np.float32) * float(args.pos_noise_std)
                        grip = (torch.sigmoid(pred_grip_logit) > 0.5).cpu().numpy().astype(np.float32)
                        actions = np.zeros((args.num_envs, 8), dtype=np.float32)
                        actions[:, :3] = gp[:, :3] + delta
                        actions[:, 3:7] = gp[:, 3:7]
                        actions[:, 7:8] = grip

                    obs, reward, done, stats = env.step(actions)

                    if verbose and ((upd % int(args.train_log_every)) == 0):
                        pfx = f"[rank {rank}] " if world_size > 1 else ""
                        print(
                            f"{pfx}[train] update={upd} loss={float(loss.item()):.6f} loss_delta={float(loss_delta.item()):.6f} loss_grip={float(loss_grip.item()):.6f}",
                            flush=True,
                        )

                # Save ckpt (rank0 only)
                if args.train_ckpt and rank == 0:
                    to_save = policy.module.state_dict() if hasattr(policy, "module") else policy.state_dict()
                    torch.save({"state_dict": to_save, "args": vars(args)}, str(args.train_ckpt))
                    print(f"[train] saved checkpoint to {args.train_ckpt}", flush=True)

                # Clean up DDP
                if world_size > 1 and ddp_inited:
                    import torch.distributed as dist

                    dist.destroy_process_group()

                # After training, optionally run a benchmark loop if videos are enabled
                if args.output_dir and (not args.no_video) and (not args.no_rgb):
                    _run_env_loop(tag="[debug_after_train]")

            except ImportError as e:
                raise RuntimeError(
                    "mode=train requires PyTorch installed in your environment (import torch failed)."
                ) from e

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
