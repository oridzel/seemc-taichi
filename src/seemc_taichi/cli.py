from __future__ import annotations

import argparse
from .config import BackendConfig
from .runtime import init_taichi


def main():
    p = argparse.ArgumentParser(description="Initialize and report the SEEMC Taichi backend")
    p.add_argument("--arch", default="cpu", choices=["cpu", "gpu", "cuda", "vulkan", "metal", "opengl"])
    args = p.parse_args()
    ti = init_taichi(BackendConfig(arch=args.arch))
    print("seemc-taichi initialized")
    print("Taichi version:", getattr(ti, "__version__", "unknown"))
    print("requested arch:", args.arch)
