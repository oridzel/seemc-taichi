from dataclasses import dataclass


@dataclass(frozen=True)
class BackendConfig:
    """Execution/storage controls. Physics controls stay in reference MCConfig."""

    arch: str = "cpu"
    max_particles: int = 1_000_000
    max_emissions: int = 1_000_000
    default_fp: str = "f64"
    default_ip: str = "i32"
    debug: bool = False
    offline_cache: bool = True
    random_seed: int = 0

    def validate(self) -> None:
        allowed = {"cpu", "gpu", "cuda", "vulkan", "metal", "opengl"}
        if self.arch not in allowed:
            raise ValueError(f"unsupported arch {self.arch!r}; choose from {sorted(allowed)}")
        if self.max_particles < 1 or self.max_emissions < 1:
            raise ValueError("particle/emission capacities must be positive")
        if self.default_fp not in {"f32", "f64"}:
            raise ValueError("default_fp must be 'f32' or 'f64'")
        if self.default_ip not in {"i32", "i64"}:
            raise ValueError("default_ip must be 'i32' or 'i64'")
