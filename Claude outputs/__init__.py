"""
mpas_core -- shared library for MPAS multiscale diffusion apps.

Used by two thin wrapper applications:
  - Diffusion/mpas_diffsion_filters.py   (multiscale decomposition: writes
                                           one low-pass-filtered band per
                                           scale break-point)
  - Blending/mpas_blending.py            (blending: filters a large-scale
                                           and a small-scale state, forms
                                           the analysis increment, folds it
                                           physics-consistently into a
                                           background state)

Both apps do: load mesh -> METIS partition + halo exchange (cached on
disk) -> build local Laplacian (cached) -> distribute state by cell
ownership -> apply the distributed-PCG diffusion filter -> app-specific
post-processing -> write output. Everything up through "apply the filter"
lives here; only the app-specific post-processing and output shape differ,
and stays in each wrapper.
"""
from .mpi_env import (
    COMM, RANK, SIZE, IS_ROOT, is_root, mpi_abort, guard, timed,
    setup_logging, logger, rlog,
)
from .mesh import MPASMesh, load_mpas_mesh
from .partition import (
    Partition, HaloExchange, build_partition, build_or_load_partition,
    build_or_load_halo_exchange, gather_full_field,
)
from .laplacian import build_local_laplacian, build_or_load_local_laplacian
from .diffusion import DiffusionFilter, apply_filter, effective_scale_km
from .physics import (
    theta_to_temperature, compute_pressure, qv_to_spechum,
    compute_surface_pressure, load_zgrid_with_fallback,
    HydroConstants, linearized_hydrostatic_balance,
)
from .winds import uv_cell_to_edges
from .spectra import PowerSpectrumAnalyzer
from .distribute import slice_axis_cells, load_and_distribute_state
from .io_netcdf import out_format_for, write_new_state_file, copy_and_overwrite
from .cache import mesh_signature, partition_signature

__all__ = [
    "COMM", "RANK", "SIZE", "IS_ROOT", "is_root", "mpi_abort", "guard", "timed",
    "setup_logging", "logger", "rlog",
    "MPASMesh", "load_mpas_mesh",
    "Partition", "HaloExchange", "build_partition", "build_or_load_partition",
    "build_or_load_halo_exchange", "gather_full_field",
    "build_local_laplacian", "build_or_load_local_laplacian",
    "DiffusionFilter", "apply_filter", "effective_scale_km",
    "theta_to_temperature", "compute_pressure", "qv_to_spechum",
    "compute_surface_pressure", "load_zgrid_with_fallback",
    "HydroConstants", "linearized_hydrostatic_balance",
    "uv_cell_to_edges",
    "PowerSpectrumAnalyzer",
    "slice_axis_cells", "load_and_distribute_state",
    "out_format_for", "write_new_state_file", "copy_and_overwrite",
    "mesh_signature", "partition_signature",
]
