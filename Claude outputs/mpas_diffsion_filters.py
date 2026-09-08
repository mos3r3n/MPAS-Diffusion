"""
3D MPAS multiscale diffusion filter -- MPI + METIS version.

Thin wrapper over the shared mpas_core library: mesh loading, METIS
partitioning, halo exchange, the local Laplacian, and the distributed-PCG
diffusion filter all live in mpas_core and are shared with the sibling
Blending/mpas_blending.py app (see that file, and mpas_core/__init__.py's
docstring, for how the two apps split responsibilities). What's specific
to THIS app and stays here: the analysis-state field set (temperature,
spechum, winds, surface_pressure -- with hydrostatic derivation when
surface_pressure is missing), the multiscale band construction (N low-pass
sweeps -> N+1 bands), writing one file per band with cross-band write
concurrency, and the optional cloud-variable exemption from filtering
(FilterConfig.filter_cloud_vars).

Usage
-----
    mpiexec -n ${ncpus} python mpas_diffsion_filters.py [config_file.yaml]
"""
from __future__ import annotations

import os
import sys
import time
import concurrent.futures
import numpy as np
import netCDF4 as nc
import yaml
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

# ── make the sibling mpas_core/ package importable regardless of cwd --
# this file lives in .../regional_mpasjedi/Diffusion/, mpas_core lives in
# .../regional_mpasjedi/mpas_core/, so the parent of this file's directory
# is what needs to be on sys.path. ──
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mpas_core as core
from mpas_core import (
    COMM, RANK, SIZE, is_root, mpi_abort, guard, timed, setup_logging, logger,
)
from mpas_core.mpi_env import MPI_SUM, _HAS_MPI


# ============================================================
# Config
# ============================================================
@dataclass
class FilterConfig:
    mesh_file:          str
    data_file:          str
    scales_km:          List[float]
    output_prefix:      str       = "state_lp"
    n_iter:             int       = 8
    log_level:          str       = "INFO"
    variables:          Optional[List[str]] = field(default=None)
    derive_temperature: bool      = True
    derive_spechum:     bool      = True
    # If surface_pressure isn't in data_file, derive it (hydrostatic
    # extrapolation of the lowest model level's pressure down to the
    # surface -- see mpas_core.compute_surface_pressure()) instead of
    # just dropping the variable. Set False to keep the old behaviour
    # (skip it silently when absent).
    derive_surface_pressure: bool = True

    cloud_vars: List[str] = field(default_factory=lambda: [
        "qc", "qi", "qr", "qs", "qg", "qh", "refl10cm",
    ])
    # Whether the cloud_vars listed above actually get diffused like every
    # other field (True, default) or are exempted -- zeroed out of every
    # low-pass sweep, so they end up entirely in the smallest-scale band
    # (False). Set via YAML: `filter_cloud_vars: false`.
    filter_cloud_vars:  bool = True
    partition_file:     Optional[str] = None

    # Disk cache for mesh/partition-derived artifacts -- shared with the
    # blending app under the same cache directory when both point at the
    # same mesh_file (see mpas_core/cache.py).
    use_cache:          bool = True
    cache_dir:          Optional[str] = None

    # Output I/O tuning. zlib compression is often 5-20x slower than an
    # uncompressed write for floating-point atmospheric fields, and only
    # applies when the output format is NETCDF4* -- see
    # mpas_core.out_format_for()'s docstring.
    output_zlib:        bool  = True
    output_complevel:   int   = 4

    pcg_tol:            float = 1e-8
    pcg_maxiter:        int   = 200
    hyper_order:        int   = 1
    calibrate:          str   = "gaussian" # or "cutoff"

    def __post_init__(self):
        self.scales_km = sorted(float(s) for s in self.scales_km)
        self._validate()

    def _validate(self):
        errors = []
        if not self.scales_km:
            errors.append("scales_km must contain at least one value")
        if self.n_iter < 1:
            errors.append("n_iter must be >= 1")
        if self.pcg_maxiter < 1:
            errors.append("pcg_maxiter must be >= 1")
        if self.pcg_tol <= 0:
            errors.append("pcg_tol must be > 0")
        if self.hyper_order < 1:
            errors.append("hyper_order must be >= 1")
        if self.calibrate not in ("gaussian", "cutoff"):
            errors.append(f"calibrate must be 'gaussian' or 'cutoff', got {self.calibrate!r}")
        if self.calibrate == "gaussian" and self.hyper_order != 1:
            errors.append(
                "calibrate='gaussian' is only valid for hyper_order=1 "
                "(it's specifically what makes the filter converge to a "
                "Gaussian); set calibrate='cutoff' to use hyper_order > 1"
            )
        if not os.path.exists(self.mesh_file):
            errors.append(f"mesh_file not found: {self.mesh_file}")
        if not os.path.exists(self.data_file):
            errors.append(f"data_file not found: {self.data_file}")
        if self.partition_file and not os.path.exists(self.partition_file):
            errors.append(f"partition_file not found: {self.partition_file}")
        if not (0 <= self.output_complevel <= 9):
            errors.append(f"output_complevel must be in [0,9], got {self.output_complevel}")
        if errors:
            raise ValueError("FilterConfig errors:\n  " + "\n  ".join(errors))

    @classmethod
    def from_yaml(cls, path: str) -> "FilterConfig":
        with open(path) as fh:
            raw = yaml.safe_load(fh)
        return cls(**raw)


# ============================================================
# Output directory pre-flight check (before any compute)
# ============================================================
def preflight_output_dir(prefix: str) -> None:
    """Ensure the output directory exists and is writable. Runs on root only."""
    if not is_root():
        return
    out_dir = os.path.dirname(os.path.abspath(prefix)) or "."
    if not os.path.isdir(out_dir):
        logger.info("Creating output directory: %s", out_dir)
        os.makedirs(out_dir, exist_ok=True)
    if not os.access(out_dir, os.W_OK):
        raise PermissionError(f"No write permission on output directory: {out_dir}")
    logger.info("Output directory OK: %s", out_dir)


# ============================================================
# Build analysis state (root only -- see load_and_distribute_state)
# ============================================================
def build_analysis_state(ds: nc.Dataset, cfg: FilterConfig) -> Dict[str, np.ndarray]:
    logger.info("Building analysis state…")
    if "pressure" in ds.variables:
        p = ds.variables["pressure"][:]
    else:
        p = ds.variables["pressure_base"][:] + ds.variables["pressure_p"][:]

    state: Dict[str, np.ndarray] = {}
    if cfg.derive_temperature:
        state["temperature"] = core.theta_to_temperature(ds.variables["theta"][:], p)
    if cfg.derive_spechum:
        state["spechum"] = core.qv_to_spechum(ds.variables["qv"][:])
    for v in ("uReconstructZonal", "uReconstructMeridional", "surface_pressure"):
        if v in ds.variables:
            state[v] = ds.variables[v][:]

    if "surface_pressure" not in state and cfg.derive_surface_pressure:
        zgrid = core.load_zgrid_with_fallback(cfg.mesh_file, ds)
        missing = [v for v in ("theta", "qv") if v not in ds.variables]
        if zgrid is None:
            missing.insert(0, f"zgrid (looked in mesh_file={cfg.mesh_file} and data_file)")
        if missing:
            logger.warning(
                "surface_pressure not in data_file and can't be derived -- "
                "missing %s. Continuing without surface_pressure.", missing,
            )
        else:
            logger.info(
                "surface_pressure not in data_file -- deriving it from "
                "pressure/zgrid/theta/qv (zgrid from mesh_file; hydrostatic "
                "extrapolation to the surface; see "
                "mpas_core.compute_surface_pressure())."
            )
            state["surface_pressure"] = core.compute_surface_pressure(
                p, zgrid, ds.variables["theta"][:], ds.variables["qv"][:],
            )

    # Cloud-related species: read raw (no derivation) if present. Whether
    # these get diffused like everything else or exempted (and end up
    # whole in the smallest-scale band instead of being smoothed) is
    # controlled by cfg.filter_cloud_vars -- see compute_lowpass in run().
    for v in cfg.cloud_vars:
        if v in ds.variables and v not in state:
            state[v] = ds.variables[v][:]

    if cfg.variables:
        missing = [v for v in cfg.variables if v not in state]
        if missing:
            raise KeyError(f"Requested variables not found in state: {missing}")
        state = {k: state[k] for k in cfg.variables}

    logger.info("State variables: %s", list(state.keys()))
    return state


# ============================================================
# Save band to NetCDF
# ============================================================
def save_state(
    input_file: str,
    output_file: str,
    filtered_state: Dict[str, np.ndarray],
    cfg: FilterConfig,
) -> None:
    """Write one self-contained band file. Rank-agnostic on purpose: the
    caller (write_bands_parallel(), below) decides which single rank calls
    this for a given band -- calling this from more than one rank at once
    for the SAME output_file would race; callers must ensure exactly one
    rank calls it per file."""
    logger.warning("[rank %d] Writing: %s", RANK, output_file)
    core.write_new_state_file(
        input_file, output_file, filtered_state,
        coord_vars=("latCell", "lonCell"),
        zlib=cfg.output_zlib, complevel=cfg.output_complevel,
        global_attrs={
            "description": "Multiscale filtered MPAS state",
            "source": input_file,
            "filter": "diffusion_multiscale",
            "created_by": f"mpas_diffsion_filters.py  ranks={SIZE}",
        },
    )
    logger.warning("[rank %d] Done: %s", RANK, output_file)


def write_bands_parallel(
    bands: List[Dict[str, np.ndarray]],
    band_names: List[str],
    cfg: FilterConfig,
    mesh: "core.MPASMesh",
    part: "core.Partition",
) -> Optional[str]:
    """
    Write one file per band -- never split per-rank -- but round-robin
    WHICH single rank does each band's write, so different bands' files
    can be in flight on disk at the same wall-clock time.

    Why round-robin across MPI ranks rather than just using Python threads
    on rank 0 to write several bands "at once": the netCDF-C/HDF5 library
    underneath isn't reliably safe to call concurrently from multiple
    threads within one process unless the specific build was compiled
    with thread-safety enabled -- not something verifiable here.
    Different MPI ranks are different OS processes with entirely separate
    library state, so spreading writer duty across ranks sidesteps that
    question completely: at most ONE thread per process ever touches
    netCDF at a time in this design, so it doesn't matter whether the
    library is thread-safe.

    Every rank still calls gather_full_field() for every band, in the
    same order (it's a collective). What changes is that its destination
    rank rotates band-by-band. Because a synchronous write would block its
    rank's main thread from reaching the NEXT band's gather -- stalling
    every other rank's progress too, since gather is collective -- the
    rank that ends up holding a band's full data hands the actual disk
    write off to a private single-worker background thread and
    immediately continues to the next gather.

    Peak memory: a rank waits for its OWN previous write to finish before
    starting its next one (relevant once there are more bands than ranks)
    -- otherwise a rank could end up holding two full gathered bands in
    memory at once, one mid-write and one freshly queued.

    Returns the first error string seen on ANY rank for ANY band
    (identical on every rank, via bcast), or None if every write
    succeeded -- callers should mpi_abort() on a non-None result.
    """
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    pending: Optional[concurrent.futures.Future] = None
    local_error: Optional[str] = None

    def _write(out_path: str, data: Dict[str, np.ndarray]) -> None:
        t0 = time.perf_counter()
        save_state(cfg.data_file, out_path, data, cfg)
        logger.warning("[rank %d] band write took %.1f s: %s",
                        RANK, time.perf_counter() - t0, out_path)

    def _collect(fut: "concurrent.futures.Future") -> Optional[str]:
        try:
            fut.result()
            return None
        except Exception as exc:
            logger.error("[rank %d] %s", RANK, exc)
            return f"write (rank {RANK}): {exc}"

    for i, (band, bname) in enumerate(zip(bands, band_names)):
        writer_rank = i % SIZE
        band_data = core.gather_full_field(band, part.owned, mesh.nCells, dest_rank=writer_rank)
        if RANK == writer_rank:
            if pending is not None:
                err = _collect(pending)
                if err and local_error is None:
                    local_error = err
            out = f"{cfg.output_prefix}_{bname}.nc"
            pending = executor.submit(_write, out, band_data)

    if pending is not None:
        err = _collect(pending)
        if err and local_error is None:
            local_error = err
    executor.shutdown(wait=True)

    all_errors = COMM.gather(local_error, root=0)
    first_error: Optional[str] = None
    if is_root():
        first_error = next((e for e in all_errors if e is not None), None)
    return COMM.bcast(first_error, root=0)


# ============================================================
# Energy diagnostics (distributed: local partial sums + allreduce)
# ============================================================
def _allreduce_sum(x: float) -> float:
    if SIZE == 1 or not _HAS_MPI:
        return x
    return COMM.allreduce(x, op=MPI_SUM)


def energy_partition_distributed(
    bands: List[Dict[str, np.ndarray]],
    band_names: List[str],
    mesh: "core.MPASMesh",
    owned: np.ndarray,
) -> None:
    """Local partial sum (weighted by this rank's owned-cell areas) plus
    one Allreduce per variable per band -- no full-mesh gather needed
    just for logging."""
    area_owned = mesh.areaCell[owned]
    if is_root():
        logger.info("─── ENERGY PARTITION ───")
    for bname, band in zip(band_names, bands):
        total = 0.0
        parts_str = []
        for vname, v in band.items():
            if v.ndim == 3:
                local_e = float(np.sum(v ** 2 * area_owned[None, :, None]))
            elif v.ndim == 2:
                local_e = float(np.sum(v ** 2 * area_owned[None, :]))
            else:
                local_e = float(np.sum(v ** 2 * area_owned))
            global_e = _allreduce_sum(local_e)
            parts_str.append(f"{vname}={global_e:.2e}")
            total += global_e
        if is_root():
            logger.info("  %-28s total=%.3e  [%s]", bname, total, "  ".join(parts_str))


# ============================================================
# Main driver
# ============================================================
def run(cfg: FilterConfig) -> None:
    setup_logging(cfg.log_level)

    # ── MPI sanity check, logged from EVERY process at WARNING level ──
    logger.warning(
        "MPI status: this_process rank=%d size=%d  pid=%d", RANK, SIZE, os.getpid(),
    )
    if SIZE == 1:
        logger.warning(
            "SIZE=1: this process sees itself as the ONLY rank in "
            "COMM_WORLD. If you launched this under mpirun/srun -n N "
            "(N>1) and expected N-way parallelism, this process is NOT "
            "actually joined to that job's communicator."
        )
    if is_root():
        logger.info("Starting  ranks=%d  scales=%s km", SIZE, cfg.scales_km)

    with guard("preflight"):
        preflight_output_dir(cfg.output_prefix)
    COMM.Barrier()

    with guard("load_mesh"), timed("load mesh"):
        mesh = core.load_mpas_mesh(cfg.mesh_file, load_edges=False)
    COMM.Barrier()

    with guard("partition"), timed("partition mesh (METIS)"):
        membership = core.build_or_load_partition(mesh, cfg)
        part = core.build_partition(mesh, membership)
        mesh_sig = core.mesh_signature(cfg.mesh_file, mesh.nCells)
        part_sig = core.partition_signature(membership)
        halo = core.build_or_load_halo_exchange(part, cfg, mesh_sig, part_sig)
        if is_root():
            counts = np.bincount(membership, minlength=SIZE)
            logger.info("Cell ownership per rank: %s", counts.tolist())
    COMM.Barrier()

    with guard("build_laplacian"), timed("build local Laplacian"):
        L_local = core.build_or_load_local_laplacian(mesh, part, cfg, mesh_sig, part_sig)
    COMM.Barrier()

    local_ids = np.concatenate([part.owned, part.halo])
    with guard("load_state"), timed("load + distribute state"):
        local_state = core.load_and_distribute_state(
            cfg.data_file, lambda ds: build_analysis_state(ds, cfg), local_ids,
        )
        n_owned = part.n_owned
        owned_state: Dict[str, np.ndarray] = {}
        for name, arr in local_state.items():
            owned_state[name] = core.slice_axis_cells(arr, np.arange(n_owned))
    COMM.Barrier()

    cloud_vars: Set[str] = set(cfg.cloud_vars) & set(owned_state.keys())
    if is_root():
        if not cloud_vars:
            logger.info("No cloud variables present in state; nothing to exempt or filter differently.")
        elif cfg.filter_cloud_vars:
            logger.info(
                "Cloud variables (filtered like every other field, "
                "filter_cloud_vars=True): %s", sorted(cloud_vars),
            )
        else:
            logger.info(
                "Cloud variables (unfiltered, kept whole in smallest-scale "
                "band '%s', filter_cloud_vars=False): %s",
                f"below{cfg.scales_km[0]:.0f}km", sorted(cloud_vars),
            )

    def compute_lowpass(filt: "core.DiffusionFilter") -> Dict[str, np.ndarray]:
        out: Dict[str, np.ndarray] = {}
        for name, data in owned_state.items():
            if name in cloud_vars and not cfg.filter_cloud_vars:
                out[name] = np.zeros_like(data, dtype=np.float64)
            else:
                out[name] = core.apply_filter(filt, data)
        return out

    lowpasses: List[Dict[str, np.ndarray]] = []
    for scale in cfg.scales_km:
        label = f"low-pass {scale} km"
        with guard(label), timed(label):
            filt = core.DiffusionFilter(L_local, halo, scale, cfg.n_iter, cfg.pcg_tol, cfg.pcg_maxiter,
                                          hyper_order=cfg.hyper_order, calibrate=cfg.calibrate)
            lowpasses.append(compute_lowpass(filt))
            filt.log_convergence_summary()
        COMM.Barrier()

    # When filter_cloud_vars=False, cloud vars are exact zero in every
    # band except the finest one (compute_lowpass zeroed their lowpass at
    # every scale, so band[i>=1] = 0 - 0 = 0 and the final "above" band is
    # lowpasses[-1] = 0 for them too) -- so skip writing them to those
    # band files at all rather than storing known-all-zero arrays. Every
    # rank computes this identically (same cfg, same cloud_vars set), so
    # it stays a valid collective: the set of gather_full_field() CALLS is
    # unchanged, only which keys each call's dict carries.
    non_finest_vars = (
        list(owned_state.keys()) if cfg.filter_cloud_vars
        else [n for n in owned_state if n not in cloud_vars]
    )

    bands:      List[Dict[str, np.ndarray]] = []
    band_names: List[str]                   = []
    bands.append({n: owned_state[n] - lowpasses[0][n] for n in owned_state})
    band_names.append(f"below{cfg.scales_km[0]:.0f}km")
    for i in range(1, len(cfg.scales_km)):
        bands.append({n: lowpasses[i-1][n] - lowpasses[i][n] for n in non_finest_vars})
        band_names.append(f"{cfg.scales_km[i-1]:.0f}-{cfg.scales_km[i]:.0f}km")
    bands.append({n: lowpasses[-1][n] for n in non_finest_vars})
    band_names.append(f"above{cfg.scales_km[-1]:.0f}km")

    if not cfg.filter_cloud_vars and cloud_vars and is_root():
        logger.info(
            "filter_cloud_vars=False: omitting %s from all bands except "
            "'below%.0fkm' (they're exactly zero everywhere else).",
            sorted(cloud_vars), cfg.scales_km[0],
        )

    energy_partition_distributed(bands, band_names, mesh, part.owned)

    with timed("write all bands"):
        write_error = write_bands_parallel(bands, band_names, cfg, mesh, part)
    if write_error is not None:
        mpi_abort(write_error)
    COMM.Barrier()

    if is_root():
        logger.info("All bands written successfully.")


# ============================================================
# CLI
# ============================================================
def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(
        description="MPAS multiscale diffusion filter (MPI + METIS domain decomposition)"
    )
    parser.add_argument(
        "config", nargs="?", default="filter.yaml",
        help="Path to YAML configuration file (default: filter.yaml)",
    )
    args = parser.parse_args()

    if is_root() and not os.path.exists(args.config):
        parser.error(f"Config file not found: {args.config}")

    with guard("load_config"):
        cfg = FilterConfig.from_yaml(args.config)

    run(cfg)


if __name__ == "__main__":
    main()
