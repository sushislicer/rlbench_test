#!/usr/bin/env python3
"""
多进程 RLBench 环境交互测试（主进程 <-> N 个环境子进程）

目标：
- 启动 N 个子进程，每个子进程维护 1 个 RLBench TaskEnvironment（CoppeliaSim 限制：不能共享 env）
- 主进程向每个子进程发送 reset / step(action) 指令
- 子进程回传 observation / done / reward
- action 使用“当前 gripper_pose + 小噪声”的方式模拟 VLA（不引入 dummy，占位 action 也不 try/except 吞错）
- 所有进程结束后，主进程保存 N 个视频（每个环境 1 个），并打印交互耗时

注意：
- RLBench 依赖 CoppeliaSim；通常需要配置 COPPELIASIM_ROOT / LD_LIBRARY_PATH 等。
- 默认每个 worker 进程会启动独立 Xvfb，避免多进程共享 DISPLAY 崩溃。
  若你已外部使用 xvfb-run，可 export RLBENCH_PER_PROCESS_XVFB=0 禁用内部 Xvfb。
"""

from __future__ import annotations

import argparse
import os
import subprocess
import time
from typing import Any, Tuple

import numpy as np  # pyright: ignore[reportMissingImports]


def _require_env(var: str) -> None:
    if var not in os.environ or str(os.environ[var]).strip() == "":
        raise RuntimeError(
            f"缺少环境变量 {var}。RLBench/CoppeliaSim 通常需要它。\n"
            "你可以参考 `LIBERO/RL_GUIDE_rlbench_dense.md` 里的 export 示例先把环境配好。"
        )


def _start_xvfb(display_str: str, max_tries: int = 64) -> Tuple[subprocess.Popen, str]:
    """
    为每个 worker 启动独立 Xvfb，并返回 (proc, actual_display)。
    """
    if display_str is None or str(display_str).strip() == "":
        raise ValueError("display_str 不能为空")

    # Qt 在一些环境里需要 XDG_RUNTIME_DIR
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
        raise ValueError(f"display_str 非法：{display_str}") from e

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
                last_err = f"Xvfb 启动失败：display={disp_str} exit_code={p.returncode} stderr={err_txt.strip()}"
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

    raise RuntimeError(last_err if last_err is not None else f"Xvfb 启动失败：未找到可用 DISPLAY（start={display_str} tries={max_tries}）")


def _build_action_from_obs(
    obs: Any,
    *,
    rng: np.random.Generator,
    pos_noise_std: float,
    gripper_close_prob: float,
) -> np.ndarray:
    """
    用 Observation 的 gripper_pose 作为 base action，并对位置加噪声。
    action 约定为 8 维：
      [x, y, z, qx, qy, qz, qw, gripper]
    """
    if not hasattr(obs, "gripper_pose"):
        raise AttributeError("obs 缺少 gripper_pose：请确保 ObservationConfig.gripper_pose=True")
    gp = np.asarray(obs.gripper_pose, dtype=np.float32)
    if gp.shape != (7,):
        raise ValueError(f"obs.gripper_pose 期望形状为 (7,)，但得到 {gp.shape}")

    pos = gp[:3]
    quat = gp[3:7]

    if float(pos_noise_std) < 0:
        raise ValueError("pos_noise_std 必须 >= 0")
    pos_noisy = pos + rng.normal(loc=0.0, scale=float(pos_noise_std), size=(3,)).astype(np.float32)

    if not (0.0 <= float(gripper_close_prob) <= 1.0):
        raise ValueError("gripper_close_prob 必须在 [0, 1] 内")
    gripper = np.float32(0.0 if (rng.random() < float(gripper_close_prob)) else 1.0)

    act = np.concatenate(
        [pos_noisy.astype(np.float32), quat.astype(np.float32), np.asarray([gripper], dtype=np.float32)],
        axis=0,
    )
    if act.shape != (8,):
        raise RuntimeError(f"构造 action 失败：shape={act.shape}")
    return act


def _obs_to_small_dict(obs: Any) -> dict:
    """
    为了跨进程通信轻量化，只回传关键低维信息（同时视频帧单独回传）。
    """
    out = {}
    if hasattr(obs, "gripper_pose"):
        gp = np.asarray(obs.gripper_pose, dtype=np.float32)
        out["gripper_pose"] = gp
    if hasattr(obs, "gripper_open"):
        go = np.asarray(obs.gripper_open, dtype=np.float32)
        out["gripper_open"] = go
    return out


def _get_front_rgb(obs: Any) -> np.ndarray:
    return _get_rgb(obs, "front_rgb", "front_camera.rgb")


def _get_wrist_rgb(obs: Any) -> np.ndarray:
    return _get_rgb(obs, "wrist_rgb", "wrist_camera.rgb")


def _get_rgb(obs: Any, attr: str, cfg_hint: str) -> np.ndarray:
    if not hasattr(obs, attr):
        raise AttributeError(f"obs 缺少 {attr}：请确保 ObservationConfig.{cfg_hint}=True")
    frame = np.asarray(getattr(obs, attr))
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"{attr} 形状不合法：{frame.shape}")
    if frame.dtype != np.uint8:
        # RLBench 通常是 uint8；若不是也不“吞掉”，显式转并说明
        frame = frame.astype(np.uint8)
    return frame


def _rlbench_worker(conn, worker_idx: int, cfg: dict) -> None:
    """
    子进程：维护一个 RLBench TaskEnvironment，响应 reset/step/shutdown。
    """
    xvfb_proc = None
    vw_front = None
    vw_wrist = None
    frame_idx = 0
    try:
        per_process_xvfb = True
        if "RLBENCH_PER_PROCESS_XVFB" in os.environ:
            per_process_xvfb = str(os.environ["RLBENCH_PER_PROCESS_XVFB"]).strip() == "1"

        if per_process_xvfb:
            base = 10000 + (int(os.getpid()) % 1000) * 64
            start_display = f":{int(base + int(worker_idx))}"
            xvfb_proc, actual_display = _start_xvfb(start_display)
            os.environ["DISPLAY"] = str(actual_display)

        # 延迟导入：避免在 Xvfb 前触发 Qt 初始化链
        from rlbench import Environment  # pyright: ignore[reportMissingImports]
        from rlbench.action_modes.action_mode import MoveArmThenGripper  # pyright: ignore[reportMissingImports]
        from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaPlanning  # pyright: ignore[reportMissingImports]
        from rlbench.action_modes.gripper_action_modes import Discrete  # pyright: ignore[reportMissingImports]
        from rlbench.observation_config import ObservationConfig, CameraConfig  # pyright: ignore[reportMissingImports]
        from rlbench.utils import name_to_task_class  # pyright: ignore[reportMissingImports]

        task_class_name = str(cfg["task_class"])
        image_size = cfg["image_size"]
        if not (isinstance(image_size, (tuple, list)) and len(image_size) == 2):
            raise ValueError("image_size 必须是 (H, W)")
        image_size_tuple = (int(image_size[0]), int(image_size[1]))

        obs_config = ObservationConfig()
        obs_config.set_all(False)
        obs_config.front_camera = CameraConfig(rgb=True, depth=False, point_cloud=False, mask=False, image_size=image_size_tuple)
        obs_config.wrist_camera = CameraConfig(rgb=True, depth=False, point_cloud=False, mask=False, image_size=image_size_tuple)
        obs_config.gripper_pose = True
        obs_config.gripper_open = True

        action_mode = MoveArmThenGripper(
            arm_action_mode=EndEffectorPoseViaPlanning(),
            gripper_action_mode=Discrete(),
        )

        env = Environment(action_mode=action_mode, obs_config=obs_config, headless=True, robot_setup=str(cfg["robot_setup"]))
        env.launch()
        task_class = name_to_task_class(task_class_name)
        task_env = env.get_task(task_class)

        # 视频写入放在 worker，避免主进程 IPC 传大量帧（num_envs 大时不可承受）
        import cv2  # pyright: ignore[reportMissingImports]

        fps = float(cfg["fps"])
        if fps <= 0:
            raise ValueError("cfg.fps 必须 > 0")
        output_dir = str(cfg["output_dir"])
        if output_dir.strip() == "":
            raise ValueError("cfg.output_dir 不能为空")
        os.makedirs(output_dir, exist_ok=True)

        out_front = os.path.join(output_dir, f"env{int(worker_idx)}_front.mp4")
        out_wrist = os.path.join(output_dir, f"env{int(worker_idx)}_wrist.mp4")

        def _open_writer(path: str, w: int, h: int):
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            vw = cv2.VideoWriter(path, fourcc, fps, (w, h))
            if not vw.isOpened():
                raise RuntimeError(f"VideoWriter 打开失败：{path}（可能缺少 mp4 编码器）")
            return vw

        def _write_rgb(vw, rgb: np.ndarray):
            if rgb.dtype != np.uint8:
                rgb = rgb.astype(np.uint8)
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            vw.write(bgr)

        # 初始化：创建完先 reset，返回初始 observation（并写入第 0 帧）
        t0 = time.perf_counter()
        done_flag = False
        desc0, obs0 = task_env.reset()
        t1 = time.perf_counter()
        last_obs = obs0
        fr0_f = _get_front_rgb(obs0)
        fr0_w = _get_wrist_rgb(obs0)
        if fr0_f.ndim != 3 or fr0_f.shape[2] != 3:
            raise RuntimeError(f"front frame 形状非法：{fr0_f.shape}")
        if fr0_w.ndim != 3 or fr0_w.shape[2] != 3:
            raise RuntimeError(f"wrist frame 形状非法：{fr0_w.shape}")
        hf, wf = int(fr0_f.shape[0]), int(fr0_f.shape[1])
        hw, ww = int(fr0_w.shape[0]), int(fr0_w.shape[1])
        vw_front = _open_writer(out_front, wf, hf)
        vw_wrist = _open_writer(out_wrist, ww, hw)
        _write_rgb(vw_front, fr0_f)
        _write_rgb(vw_wrist, fr0_w)
        frame_idx = 1
        obs0_small = _obs_to_small_dict(obs0)
        t2 = time.perf_counter()

        conn.send(
            (
                "ready",
                {
                    "pid": int(os.getpid()),
                    "desc": desc0,
                    "obs": obs0_small,
                    "done": False,
                    "reward": 0.0,
                    "video_front": out_front,
                    "video_wrist": out_wrist,
                    "init_frame_count": 1,
                    "t_exec_s": float(t1 - t0),
                    "t_post_s": float(t2 - t1),
                    "t_total_s": float(t2 - t0),
                },
            )
        )

        while True:
            msg = conn.recv()
            if (not isinstance(msg, dict)) or ("cmd" not in msg):
                raise RuntimeError("worker 收到非法消息（需 dict 且包含 cmd）")
            cmd = str(msg["cmd"])

            if cmd == "step_chunk":
                if "actions" not in msg:
                    raise KeyError("step_chunk 缺少 actions")
                actions = np.asarray(msg["actions"], dtype=np.float32)
                if actions.ndim != 2 or actions.shape[1] != 8:
                    raise ValueError(f"actions 形状必须为 (K, 8)，但得到 {actions.shape}")
                k = int(actions.shape[0])
                if k < 0:
                    raise ValueError("K 必须 >= 0")

                start_frame = int(frame_idx)
                n_exec = 0
                reward_sum = 0.0

                # 如果已经 done，则不再推进，直接返回 0 帧
                if done_flag or k == 0:
                    obs_small = _obs_to_small_dict(last_obs)
                    conn.send(
                        (
                            "ok",
                            {
                                "obs": obs_small,
                                "done": bool(done_flag),
                                "reward_sum": float(reward_sum),
                                "n_steps": int(n_exec),
                                "frame_range": [int(start_frame), int(start_frame)],
                                # timing
                                "t_exec_s": 0.0,
                                "t_post_s": 0.0,
                                "t_total_s": 0.0,
                            },
                        )
                    )
                    continue

                t0 = time.perf_counter()
                # 连续执行 k 步，遇到 done 立即停止；每步写入 front+wrist 两路视频
                for j in range(k):
                    obs, reward, terminate = task_env.step(actions[j])
                    last_obs = obs
                    reward_sum += float(reward)
                    n_exec += 1

                    fr_f = _get_front_rgb(obs)
                    fr_w = _get_wrist_rgb(obs)
                    _write_rgb(vw_front, fr_f)
                    _write_rgb(vw_wrist, fr_w)
                    frame_idx += 1

                    if bool(terminate):
                        done_flag = True
                        break
                t1 = time.perf_counter()
                obs_small = _obs_to_small_dict(last_obs)
                t2 = time.perf_counter()

                end_frame = int(start_frame + n_exec)
                conn.send(
                    (
                        "ok",
                        {
                            "obs": obs_small,
                            "done": bool(done_flag),
                            "reward_sum": float(reward_sum),
                            "n_steps": int(n_exec),
                            "frame_range": [int(start_frame), int(end_frame)],
                            # timing
                            "t_exec_s": float(t1 - t0),   # k 次 task_env.step + 取帧 + 写视频（关键执行路径）
                            "t_post_s": float(t2 - t1),  # 打包/小字典
                            "t_total_s": float(t2 - t0),
                        },
                    )
                )
                continue

            if cmd == "shutdown":
                try:
                    env.shutdown()
                except Exception:
                    pass
                try:
                    if vw_front is not None:
                        vw_front.release()
                except Exception:
                    pass
                try:
                    if vw_wrist is not None:
                        vw_wrist.release()
                except Exception:
                    pass
                conn.send(("ok", None))
                return

            raise RuntimeError(f"未知 cmd: {cmd}")
    except Exception as e:
        import traceback

        tb = traceback.format_exc()
        try:
            conn.send(("err", {"err": repr(e), "traceback": tb}))
        except Exception:
            pass
    finally:
        try:
            conn.close()
        except Exception:
            pass
        try:
            if xvfb_proc is not None:
                xvfb_proc.terminate()
                try:
                    xvfb_proc.wait(timeout=2.0)
                except Exception:
                    xvfb_proc.kill()
        except Exception:
            pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_class", type=str, default="OpenDrawer", help="RLBench task class 名称（如 OpenDrawer）")
    parser.add_argument("--num_envs", type=int, default=4, help="环境进程数 N（每个进程 1 个 TaskEnvironment）")
    parser.add_argument("--image_size", type=int, nargs=2, default=[256, 256], help="相机分辨率 [H W]（front+wrist）")
    parser.add_argument("--max_steps", type=int, default=50, help="每个环境的最大 step 数（done 前会提前停止统计）")
    parser.add_argument("--action_chunk", type=int, default=54, help="每次发送的 action chunk 长度 K（默认 54）")
    parser.add_argument("--fps", type=float, default=20.0, help="保存视频的 fps（RLBench 内部 dt=0.05 对应 20fps）")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="主进程保存 N 个视频的输出目录（默认：RLBench_env_test/rlbench_env_test_videos）",
    )
    parser.add_argument("--pos_noise_std", type=float, default=0.01, help="位置噪声标准差（米）")
    parser.add_argument("--gripper_close_prob", type=float, default=0.05, help="每步关闭夹爪的概率（0~1）")
    parser.add_argument("--seed", type=int, default=0, help="随机种子")
    parser.add_argument("--sleep_s", type=float, default=0.0, help="每步 sleep（用于观察资源占用/并发稳定性）")
    args = parser.parse_args()

    if int(args.num_envs) <= 0:
        raise ValueError("--num_envs 必须 > 0")
    if int(args.max_steps) <= 0:
        raise ValueError("--max_steps 必须 > 0")
    if float(args.pos_noise_std) < 0:
        raise ValueError("--pos_noise_std 必须 >= 0")
    if not (0.0 <= float(args.gripper_close_prob) <= 1.0):
        raise ValueError("--gripper_close_prob 必须在 [0, 1] 内")
    if not (isinstance(args.image_size, (list, tuple)) and len(args.image_size) == 2):
        raise ValueError("--image_size 必须是两个整数 [H W]")
    if int(args.action_chunk) <= 0:
        raise ValueError("--action_chunk 必须 > 0")

    # 尽量在启动多进程和导入 RLBench 前做最小前置检查，避免“静默卡住”
    _require_env("COPPELIASIM_ROOT")

    if float(args.fps) <= 0:
        raise ValueError("--fps 必须 > 0")
    output_dir = str(args.output_dir)
    if output_dir.strip() == "":
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rlbench_env_test_videos")
    os.makedirs(output_dir, exist_ok=True)

    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    conns = []
    procs = []
    cfg = {
        "task_class": str(args.task_class),
        "image_size": (int(args.image_size[0]), int(args.image_size[1])),
        "robot_setup": "panda",
        "fps": float(args.fps),
        "output_dir": str(output_dir),
    }
    for i in range(int(args.num_envs)):
        parent_end, child_end = ctx.Pipe(duplex=True)
        p = ctx.Process(target=_rlbench_worker, args=(child_end, int(i), dict(cfg)))
        p.start()
        child_end.close()
        conns.append(parent_end)
        procs.append(p)

    # 等待 ready（包含初始 observation）
    obs_small = [None for _ in range(int(args.num_envs))]
    done = np.zeros((int(args.num_envs),), dtype=bool)
    ret = np.zeros((int(args.num_envs),), dtype=np.float32)
    steps_taken = np.zeros((int(args.num_envs),), dtype=np.int32)
    video_front = ["" for _ in range(int(args.num_envs))]
    video_wrist = ["" for _ in range(int(args.num_envs))]
    for i in range(int(args.num_envs)):
        st, payload = conns[i].recv()
        if st != "ready":
            raise RuntimeError(f"worker{i} 启动失败：{st} {payload}")
        if (not isinstance(payload, dict)) or ("obs" not in payload) or ("video_front" not in payload) or ("video_wrist" not in payload):
            raise RuntimeError(f"worker{i} ready 返回格式错误：{payload}")
        obs_small[i] = payload["obs"]
        video_front[i] = str(payload["video_front"])
        video_wrist[i] = str(payload["video_wrist"])

    rng = np.random.default_rng(seed=int(args.seed))
    print(
        f"[MP-RLBENCH-TEST] task_class={args.task_class} num_envs={int(args.num_envs)} "
        f"max_steps={int(args.max_steps)} action_chunk={int(args.action_chunk)} "
        f"output_dir={output_dir}"
    )

    # ready 阶段已经做过一次 reset（并写入初始帧）；这里把 ready 的执行时间粗略作为 reset_time
    #（不再单独发 reset 指令，避免额外开销）
    t_reset = 0.0

    # chunk loop（每轮对每个 env 发送一段 actions；worker 顺序执行直到 done 或耗尽）
    t_start = time.time()
    chunk_latencies = []
    chunk_exec_latencies = []
    chunk_worker_total_latencies = []
    chunk_ipc_latencies = []
    round_idx = 0
    while True:
        if int(done.sum()) == int(args.num_envs):
            break
        if bool(np.all(steps_taken >= int(args.max_steps))):
            break

        t_round0 = time.time()
        t_send0 = time.perf_counter()
        # send
        for i in range(int(args.num_envs)):
            remaining = int(args.max_steps) - int(steps_taken[i])
            if remaining <= 0 or bool(done[i]):
                # 不再推进
                conns[i].send({"cmd": "step_chunk", "actions": np.zeros((0, 8), dtype=np.float32)})
                continue

            k = int(min(int(args.action_chunk), remaining))
            if obs_small[i] is None or "gripper_pose" not in obs_small[i]:
                raise RuntimeError(f"env{i} 缺少 gripper_pose（ready 的 obs_config 可能未开启）")

            # 使用当前 gripper_pose 作为 base，对位置加噪声构造一段动作序列（模拟 VLA）
            base_gp = np.asarray(obs_small[i]["gripper_pose"], dtype=np.float32)
            if base_gp.shape != (7,):
                raise RuntimeError(f"env{i} gripper_pose 形状非法：{base_gp.shape}")
            base_pos = base_gp[:3]
            base_quat = base_gp[3:7]

            pos_noise = rng.normal(loc=0.0, scale=float(args.pos_noise_std), size=(k, 3)).astype(np.float32)
            pos_seq = base_pos[None, :].astype(np.float32) + pos_noise
            quat_seq = np.repeat(base_quat[None, :].astype(np.float32), repeats=k, axis=0)
            grip_seq = (rng.random(size=(k, 1)) >= float(args.gripper_close_prob)).astype(np.float32)  # True->1(open), False->0(close)
            actions = np.concatenate([pos_seq, quat_seq, grip_seq], axis=1).astype(np.float32)
            if actions.shape != (k, 8):
                raise RuntimeError(f"env{i} actions 构造失败：{actions.shape}")
            conns[i].send({"cmd": "step_chunk", "actions": actions})
        t_send1 = time.perf_counter()

        # recv
        t_recv0 = time.perf_counter()
        per_env_exec = []
        per_env_total = []
        n_exec_sum = 0
        for i in range(int(args.num_envs)):
            st, payload = conns[i].recv()
            if st != "ok":
                raise RuntimeError(f"worker{i} step_chunk 失败：{payload}")
            if not isinstance(payload, dict):
                raise RuntimeError(f"worker{i} step_chunk 返回 payload 非 dict：{type(payload)}")
            need = {"obs", "done", "reward_sum", "n_steps", "frame_range", "t_exec_s", "t_total_s"}
            if any(k not in payload for k in need):
                raise RuntimeError(f"worker{i} step_chunk 缺少字段：need={need} got={payload.keys()}")
            obs_small[i] = payload["obs"]
            done[i] = bool(done[i] or bool(payload["done"]))
            ret[i] += float(payload["reward_sum"])
            n_steps = int(payload["n_steps"])
            if n_steps < 0:
                raise RuntimeError("n_steps 不应为负")
            steps_taken[i] += n_steps
            n_exec_sum += n_steps
            per_env_exec.append(float(payload["t_exec_s"]))
            per_env_total.append(float(payload["t_total_s"]))
        t_recv1 = time.perf_counter()

        round_lat = time.time() - t_round0
        chunk_latencies.append(float(round_lat))
        exec_lat = float(np.max(np.asarray(per_env_exec, dtype=np.float64))) if len(per_env_exec) > 0 else 0.0
        worker_total_lat = float(np.max(np.asarray(per_env_total, dtype=np.float64))) if len(per_env_total) > 0 else 0.0
        ipc_lat = float(round_lat - worker_total_lat)
        chunk_exec_latencies.append(exec_lat)
        chunk_worker_total_latencies.append(worker_total_lat)
        chunk_ipc_latencies.append(ipc_lat)

        send_lat = float(t_send1 - t_send0)
        recv_lat = float(t_recv1 - t_recv0)
        print(
            f"[MP-RLBENCH-TEST] chunk={round_idx:04d} done={int(done.sum())}/{int(args.num_envs)} "
            f"avg_return={float(ret.mean()):.4f} steps_exec_sum={int(n_exec_sum)} "
            f"chunk_total_s={round_lat:.3f} exec_s={exec_lat:.3f} ipc_s={ipc_lat:.3f} "
            f"(send_s={send_lat:.3f} recv_s={recv_lat:.3f})"
        )

        round_idx += 1
        if float(args.sleep_s) > 0:
            time.sleep(float(args.sleep_s))

    # shutdown
    for i in range(int(args.num_envs)):
        try:
            conns[i].send({"cmd": "shutdown"})
        except Exception:
            pass
    for i in range(int(args.num_envs)):
        try:
            _st, _pl = conns[i].recv()
        except Exception:
            pass
    for p in procs:
        try:
            p.join(timeout=10.0)
        except Exception:
            pass

    for i in range(int(args.num_envs)):
        print(f"[MP-RLBENCH-TEST] saved videos: {video_front[i]} , {video_wrist[i]} steps={int(steps_taken[i])}")

    elapsed = time.time() - t_start
    avg_step_lat = float(np.mean(chunk_latencies)) if len(chunk_latencies) > 0 else 0.0
    avg_exec = float(np.mean(chunk_exec_latencies)) if len(chunk_exec_latencies) > 0 else 0.0
    avg_ipc = float(np.mean(chunk_ipc_latencies)) if len(chunk_ipc_latencies) > 0 else 0.0
    print(
        f"[MP-RLBENCH-TEST] reset_time_s={t_reset:.3f} "
        f"avg_step_total_s={avg_step_lat:.3f} avg_exec_s={avg_exec:.3f} avg_ipc_s={avg_ipc:.3f} "
        f"total_interaction_time_s={elapsed:.3f}"
    )
    print(f"[MP-RLBENCH-TEST] done={int(done.sum())}/{int(args.num_envs)} avg_return={float(ret.mean()):.4f}")


if __name__ == "__main__":
    main()

