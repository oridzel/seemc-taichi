from __future__ import annotations

from .config import BackendConfig


def _require_taichi():
    try:
        import taichi as ti
    except ImportError as exc:
        raise RuntimeError(
            "Taichi is not installed. Install seemc-taichi with `python -m pip install -e .`."
        ) from exc
    return ti


def resolve_arch(ti, name: str):
    mapping = {
        "cpu": ti.cpu,
        "gpu": ti.gpu,
        "cuda": ti.cuda,
        "vulkan": ti.vulkan,
        "metal": ti.metal,
        "opengl": ti.opengl,
    }
    return mapping[name]


def init_taichi(config: BackendConfig | None = None):
    """Initialize Taichi once using explicit numerical precision."""
    config = config or BackendConfig()
    config.validate()
    ti = _require_taichi()
    fp = ti.f64 if config.default_fp == "f64" else ti.f32
    ip = ti.i64 if config.default_ip == "i64" else ti.i32
    ti.init(
        arch=resolve_arch(ti, config.arch),
        default_fp=fp,
        default_ip=ip,
        debug=config.debug,
        offline_cache=config.offline_cache,
        random_seed=config.random_seed,
    )
    return ti
