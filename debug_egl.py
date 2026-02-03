#!/usr/bin/env python3
"""
Debug script for RLBench/PyRep EGL headless rendering.
Tests different QT_QPA_PLATFORM settings and checks for common issues.
"""

import os
import sys
import subprocess
import time

def check_env():
    print("="*60)
    print("ENVIRONMENT CHECK")
    print("="*60)
    
    coppelia_root = os.environ.get("COPPELIASIM_ROOT")
    print(f"COPPELIASIM_ROOT: {coppelia_root}")
    
    if not coppelia_root or not os.path.isdir(coppelia_root):
        print("ERROR: COPPELIASIM_ROOT is not set or invalid.")
        return False
        
    ld_path = os.environ.get("LD_LIBRARY_PATH", "")
    print(f"LD_LIBRARY_PATH: {ld_path}")
    if coppelia_root not in ld_path:
        print("WARNING: COPPELIASIM_ROOT not in LD_LIBRARY_PATH.")
        
    plugin_path = os.environ.get("QT_QPA_PLATFORM_PLUGIN_PATH")
    print(f"QT_QPA_PLATFORM_PLUGIN_PATH: {plugin_path}")
    
    # Check for Qt plugins
    platforms_dir = os.path.join(coppelia_root, "platforms")
    if not os.path.isdir(platforms_dir):
        # Sometimes it's in plugins/platforms?
        platforms_dir = os.path.join(coppelia_root, "plugins", "platforms")
    
    if os.path.isdir(platforms_dir):
        print(f"Found Qt platforms dir: {platforms_dir}")
        try:
            plugins = os.listdir(platforms_dir)
            print(f"Available plugins: {plugins}")
        except Exception as e:
            print(f"Error listing plugins: {e}")
    else:
        print(f"WARNING: Could not find 'platforms' directory in {coppelia_root}")

    # Check NVIDIA
    try:
        subprocess.run(["nvidia-smi"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("nvidia-smi: OK")
    except:
        print("nvidia-smi: FAILED (or not found)")
        
    return True

def run_test(backend, env_vars):
    print(f"\nTesting backend: {backend} (Env: {env_vars})")
    
    code = """
import sys
import os
import time
try:
    print("Importing PyRep...", flush=True)
    from pyrep import PyRep
    print("Creating PyRep instance...", flush=True)
    pr = PyRep()
    print("Launching PyRep (headless=True)...", flush=True)
    pr.launch(headless=True)
    print("PyRep launched successfully!", flush=True)
    for i in range(5):
        pr.step()
    print("Stepped successfully.", flush=True)
    pr.shutdown()
    print("Shutdown complete.", flush=True)
except Exception as e:
    print(f"CRASHED: {e}", flush=True)
    import traceback
    traceback.print_exc()
    sys.exit(1)
"""
    
    env = os.environ.copy()
    env.update(env_vars)
    
    # Ensure we don't use X11
    env.pop("DISPLAY", None)
    
    start = time.time()
    p = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=30
    )
    dur = time.time() - start
    
    print(f"Exit Code: {p.returncode}")
    print(f"Stdout:\n{p.stdout}")
    print(f"Stderr:\n{p.stderr}")
    
    if "This plugin does not support createPlatformOpenGLContext" in p.stderr:
        print("RESULT: FAILED (Plugin does not support OpenGL)")
    elif p.returncode == 0:
        print("RESULT: SUCCESS")
    else:
        print("RESULT: FAILED")

def main():
    if not check_env():
        return

    # 1. Test 'offscreen' (Standard headless)
    run_test("offscreen", {
        "QT_QPA_PLATFORM": "offscreen",
        "QT_QPA_PLATFORM_PLUGIN_PATH": os.environ.get("COPPELIASIM_ROOT", ""),
        "QT_OPENGL": "desktop"
    })

    # 2. Test 'minimalegl' (Often required for EGL)
    run_test("minimalegl", {
        "QT_QPA_PLATFORM": "minimalegl",
        "QT_QPA_PLATFORM_PLUGIN_PATH": os.environ.get("COPPELIASIM_ROOT", ""),
        "QT_OPENGL": "desktop",
        "EGL_PLATFORM": "device" # Sometimes helps
    })
    
    # 3. Test 'eglfs' (Another EGL option)
    run_test("eglfs", {
        "QT_QPA_PLATFORM": "eglfs",
        "QT_QPA_PLATFORM_PLUGIN_PATH": os.environ.get("COPPELIASIM_ROOT", ""),
    })

if __name__ == "__main__":
    main()
