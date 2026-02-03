#!/usr/bin/env python3
"""Compatibility wrapper.

Some setups/scripts refer to `rlbench_vec_env.py` while the actual implementation
in this repo lives in `rl_vec_env.py`.

This wrapper keeps the old entrypoint working without duplicating code.
"""

from __future__ import annotations


def main() -> None:
    from rl_vec_env import main as _main

    _main()


if __name__ == "__main__":
    main()

