"""Experimental Taichi acceleration backend for SEEMC."""

from .config import BackendConfig
from .runtime import init_taichi
from .tables import HostMaterialTables, extract_reference_tables
from .validation import YieldStats, compare_yields
from .bulk import BulkPhysicsConfig, BulkTransportEngine
from .plane import PlaneTransportEngine, load_reference_plane_tables
from .surface import SurfacePhysicsConfig
from .trapezoid import TrapezoidGeometryConfig, TrapezoidTransportEngine

__all__ = [
    "BackendConfig",
    "BulkPhysicsConfig",
    "BulkTransportEngine",
    "PlaneTransportEngine",
    "load_reference_plane_tables",
    "SurfacePhysicsConfig",
    "TrapezoidGeometryConfig",
    "TrapezoidTransportEngine",
    "HostMaterialTables",
    "YieldStats",
    "compare_yields",
    "extract_reference_tables",
    "init_taichi",
]

__version__ = "0.7.3"
