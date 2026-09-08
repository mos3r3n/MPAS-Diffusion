"""
3D MPAS multiscale diffusion blending -- MPI + METIS version.

Thin wrapper over the shared mpas_core library: mesh loading, METIS
partitioning, halo exchange, the local Laplacian, and the distributed-PCG
diffusion filter all live in mpas_core and are shared with the sibling
Diffusion/mpas_diffsion_filters.py app. What's specific to THIS app and
stays here: building the large-scale/small-scale/background state field
sets, computing the analysis increment (large_filtered - small_filtered),
projecting the wind increment onto edges, folding the linearized
hydrostatic-balance increment into pressure/rho/theta, and writing the
final blended file.

Usage
-----
    mpiexec -n ${ncpus} python mpas_blending.py [config_file.yaml]
"""
from __future__ import annotations

import os
import sys
import numpy as np
import netCDF4 as nc
import yaml
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

# ── make the sibling mpas_core/ package importable regardless of cwd --
# this file lives in .../regional_mpasjedi/Blending/, mpas_core lives in
# .../regional_mpasjedi/mpas_core/. ──
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mpas_core as core
from mpas_core import COMM, RANK, SIZE, is_root, guard, timed, setup_logging, logger, rlog


# ============================================================
# Config
# ============================================================
@dataclass
class Config:
    mesh_file:          str = "x1.522172.invariant.nc"
    large_scale_file:   str = "large_scale_file.nc"
    small_scale_file:   str = "small_scale_file.nc"
    blended_file:       str = "mpas_blended_file.nc"
    scale_km:           float = 30.0
    n_iter:             int = 8
    log_level:          str = "INFO"

    output_filtered_fields: bool = False
    output_dir:          str = "filtered_output"
    compute_spectra:     bool = False
    spectral_bins:       int = 32

    # Disk cache -- shared with the filter app under the same cache
    # directory when both point at the same mesh_file.
    use_cache:           bool = True
    cache_dir:            Optional[str] = None
    partition_file:       Optional[str] = None

    pcg_tol:              float = 1e-8
    pcg_maxiter:           int = 200
    hyper_order:            int = 1
    calibrate:               str = "gaussian"   # or "cutoff"
    output_zlib:              bool = True
    output_complevel:          int = 4

    def __post_init__(self):
        self._validate()

    def _validate(self):
        errors = []
        if self.scale_km <= 0:
            errors.append("scale_km must be > 0")
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
                "calibrate='gaussian' is only valid for hyper_order=1; "
                "set calibrate='cutoff' to use hyper_order > 1"
            )
        if not (0 <= self.output_complevel <= 9):
            errors.append(f"output_complevel must be in [0,9], got {self.output_complevel}")
        if errors:
            raise ValueError("Config errors:\n  " + "\n  ".join(errors))

    @classmethod
    def from_yaml(cls, yaml_file: str) -> "Config":
        with open(yaml_file, 'r') as f:
            config_dict = yaml.safe_load(f) or {}
        return cls(**config_dict)


# ============================================================
# Build state dicts from an open Dataset (full-mesh, root only)
# ============================================================
def build_analysis_state(ds: "nc.Dataset") -> dict:
    if "pressure" in ds.variables:
        p = ds.variables["pressure"][:]
    else:
        p = ds.variables["pressure_base"][:] + ds.variables["pressure_p"][:]
    return {
        "temperature":            core.theta_to_temperature(ds.variables["theta"][:], p),
        "qv":                     ds.variables["qv"][:],
        "uReconstructZonal":      ds.variables["uReconstructZonal"][:],
        "uReconstructMeridional": ds.variables["uReconstructMeridional"][:],
        "surface_pressure":       ds.variables["surface_pressure"][:],
    }


def build_blended_state(ds: "nc.Dataset") -> dict:
    if "pressure" in ds.variables:
        p = ds.variables["pressure"][:]
    else:
        p = ds.variables["pressure_base"][:] + ds.variables["pressure_p"][:]
    return {
        "uReconstructZonal":      ds.variables["uReconstructZonal"][:],
        "uReconstructMeridional": ds.variables["uReconstructMeridional"][:],
        "u":                      ds.variables["u"][:],
        "theta":                  ds.variables["theta"][:],
        "temperature":            core.theta_to_temperature(ds.variables["theta"][:], p),
        "qv":                     ds.variables["qv"][:],
        "surface_pressure":       ds.variables["surface_pressure"][:],
        "pressure_p":             ds.variables["pressure_p"][:],
        "pressure":               p,
        "rho":                    ds.variables["rho"][:],
    }


# 'u' lives on nEdges, never cell-partitioned -- see build_blended_state_cell_vars().
_BLENDED_EDGE_VARS = {"u"}


def _build_blended_state_cell_vars(ds: "nc.Dataset") -> dict:
    """Same as build_blended_state() but omitting edge-dimensioned fields
    (currently just 'u') -- state is distributed by CELL ownership, so 'u'
    is handled separately (see run(), which reads it directly on root)."""
    state = build_blended_state(ds)
    return {k: v for k, v in state.items() if k not in _BLENDED_EDGE_VARS}


def _read_full_field_root(file_path: str, var_name: str):
    """Root reads one full variable directly -- used for background 'u',
    which lives on nEdges and is never partitioned by cell ownership."""
    if not is_root():
        return None
    with nc.Dataset(file_path) as ds:
        return np.asarray(ds.variables[var_name][:], dtype=np.float64)


# ============================================================
# Optional diagnostics: filtered-field + spectra output
# ============================================================
class OutputManager:
    def __init__(self, config: Config):
        self.config = config
        self.output_dir = Path(config.output_dir)
        if is_root():
            self.output_dir.mkdir(parents=True, exist_ok=True)

    def save_filtered_state(self, filtered_state: Dict[str, np.ndarray],
                             state_name: str, scale_km: float,
                             input_template: str) -> None:
        if not self.config.output_filtered_fields or not is_root():
            return
        output_file = self.output_dir / f"{state_name}_scale_filtered_{scale_km:.1f}km.nc"
        core.write_new_state_file(
            input_template, str(output_file), filtered_state,
            coord_vars=(
                "latCell", "lonCell", "xCell", "yCell", "zCell",
                "latEdge", "lonEdge", "xEdge", "yEdge", "zEdge",
                "verticesOnCell", "edgesOnCell", "cellsOnCell",
                "nEdgesOnCell", "nEdgesOnEdge", "cellsOnEdge",
                "dcEdge", "dvEdge", "areaCell", "areaEdge",
                "angleEdge", "edgeNormalVectors", "meshDensity",
            ),
            zlib=self.config.output_zlib, complevel=self.config.output_complevel,
            global_attrs={"description": "Multiscale filtered MPAS state",
                           "source": input_template, "filter": "diffusion_multiscale"},
        )

    def save_spectra(self, spectra: Dict, prefix: str = "spectrum") -> None:
        if not is_root():
            return
        filename = self.output_dir / f"{prefix}_{self.config.scale_km:.1f}km.nc"
        with nc.Dataset(filename, 'w', format='NETCDF4') as ds:
            for name, spec in spectra.items():
                if 'large' in spec and 'small' in spec:
                    grp = ds.createGroup(name)

                    large_grp = grp.createGroup('large')
                    large_grp.createDimension('wavenumber', len(spec['large']['wavenumber']))
                    large_grp.createVariable('wavenumber', 'f8', ('wavenumber',))[:] = spec['large']['wavenumber']
                    large_grp.createVariable('power', 'f8', ('wavenumber',))[:] = spec['large']['power']

                    small_grp = grp.createGroup('small')
                    small_grp.createDimension('wavenumber', len(spec['small']['wavenumber']))
                    small_grp.createVariable('wavenumber', 'f8', ('wavenumber',))[:] = spec['small']['wavenumber']
                    small_grp.createVariable('power', 'f8', ('wavenumber',))[:] = spec['small']['power']

                    ratio_grp = grp.createGroup('ratio')
                    wavenum_common = np.union1d(spec['large']['wavenumber'], spec['small']['wavenumber'])
                    ratio_grp.createDimension('wavenumber', len(wavenum_common))
                    ratio_grp.createVariable('wavenumber', 'f8', ('wavenumber',))[:] = wavenum_common
        rlog(f"Saved spectra: {filename}")


# ============================================================
# Main driver
# ============================================================
def run(cfg: Config) -> None:
    """
    Domain-decomposed MPAS blending driver.

    Pipeline (every rank does the cell-based steps for its OWN partition
    only):
      1. every rank reads the full mesh directly, incl. edgeNormalVectors
         + zgrid (needed for wind projection / hydrostatic balance)
      2. METIS partition + halo exchange built/loaded from cache
      3. local Laplacian built/loaded from cache
      4. large-scale / small-scale / blended-background states: root
         reads, slices each rank's owned+halo cells, scatters
      5. distributed PCG diffusion filter applied to large- and
         small-scale owned-cell states
      6. analysis increment = large_filtered - small_filtered, computed
         locally (no communication -- pure elementwise subtraction)
      7. wind increment projected to edges via local-partial + Allreduce
      8. hydrostatic balance increment computed locally per-column (no
         halo needed -- see mpas_core.physics)
      9. increments folded into the blended background locally, then
         gathered to root and written
    """
    setup_logging(cfg.log_level)
    logger.warning(
        "MPI status: this_process rank=%d size=%d  pid=%d", RANK, SIZE, os.getpid(),
    )
    if SIZE == 1:
        logger.warning(
            "SIZE=1: this process sees itself as the ONLY rank in "
            "COMM_WORLD. If you launched under mpirun/srun -n N (N>1) and "
            "expected N-way parallelism, this process is not actually "
            "joined to that job's communicator."
        )
    if is_root():
        logger.info("Starting mpas_blending  ranks=%d  scale=%g km", SIZE, cfg.scale_km)

    with guard("preflight"):
        if is_root():
            out_dir = os.path.dirname(os.path.abspath(cfg.blended_file)) or "."
            os.makedirs(out_dir, exist_ok=True)
    COMM.Barrier()

    with guard("load_mesh"), timed("load mesh"):
        mesh = core.load_mpas_mesh(cfg.mesh_file, load_edges=True)
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

    n_owned = part.n_owned
    local_ids = np.concatenate([part.owned, part.halo])
    owned_idx = np.arange(n_owned)

    with guard("load_state"), timed("load + distribute states"):
        large_local = core.load_and_distribute_state(cfg.large_scale_file, build_analysis_state, local_ids)
        small_local = core.load_and_distribute_state(cfg.small_scale_file, build_analysis_state, local_ids)
        blended_local = core.load_and_distribute_state(
            cfg.small_scale_file, _build_blended_state_cell_vars, local_ids,
        )
        u_background = _read_full_field_root(cfg.small_scale_file, "u")

        large_owned = {n: core.slice_axis_cells(a, owned_idx) for n, a in large_local.items()}
        small_owned = {n: core.slice_axis_cells(a, owned_idx) for n, a in small_local.items()}
        blended_owned = {n: core.slice_axis_cells(a, owned_idx) for n, a in blended_local.items()}
    COMM.Barrier()

    with guard("filter"), timed(f"low-pass {cfg.scale_km} km"):
        filt = core.DiffusionFilter(
            L_local, halo, cfg.scale_km, cfg.n_iter,
            cfg.pcg_tol, cfg.pcg_maxiter, cfg.hyper_order, cfg.calibrate,
        )
        ls_filtered = {n: core.apply_filter(filt, a) for n, a in large_owned.items()}
        ss_filtered = {n: core.apply_filter(filt, a) for n, a in small_owned.items()}
        filt.log_convergence_summary()
    COMM.Barrier()

    need_full_filtered = cfg.output_filtered_fields or cfg.compute_spectra
    ls_full = core.gather_full_field(ls_filtered, part.owned, mesh.nCells, dest_rank=0) if need_full_filtered else None
    ss_full = core.gather_full_field(ss_filtered, part.owned, mesh.nCells, dest_rank=0) if need_full_filtered else None

    output_mgr = OutputManager(cfg) if (cfg.output_filtered_fields or cfg.compute_spectra) else None
    if is_root() and cfg.output_filtered_fields:
        rlog("Saving filtered states...")
        output_mgr.save_filtered_state(ls_full, "large", cfg.scale_km, cfg.small_scale_file)
        output_mgr.save_filtered_state(ss_full, "small", cfg.scale_km, cfg.small_scale_file)
    COMM.Barrier()

    with guard("increments"), timed("compute increments"):
        ana_inc = {name: ls_filtered[name] - ss_filtered[name] for name in ls_filtered}

        # Wind increment: local partial contributions from owned cells,
        # Allreduce-summed to the full nEdges result on every rank.
        ana_inc["u"] = core.uv_cell_to_edges(
            ana_inc["uReconstructZonal"], ana_inc["uReconstructMeridional"],
            mesh.lonCell[part.owned], mesh.latCell[part.owned],
            mesh.edgeNormalVectors,
            mesh.nEdgesOnCell[part.owned], mesh.edgesOnCell[part.owned],
            mesh.nEdges,
        )

        # Hydrostatic balance: purely local per-column physics, no halo.
        ana_inc["pressure_p"], ana_inc["rho"], ana_inc["theta"] = \
            core.linearized_hydrostatic_balance(
                mesh.zgrid[part.owned],
                blended_owned["temperature"], blended_owned["qv"],
                blended_owned["surface_pressure"], blended_owned["pressure"],
                ana_inc["temperature"], ana_inc["qv"], ana_inc["surface_pressure"],
            )

        mpas_cell_vars = {"theta", "qv", "pressure_p", "rho",
                           "uReconstructZonal", "uReconstructMeridional",
                           "surface_pressure"}
        for name, inc in ana_inc.items():
            if name in mpas_cell_vars and name in blended_owned:
                blended_owned[name] = blended_owned[name] + inc
    COMM.Barrier()

    if is_root() and cfg.compute_spectra:
        rlog("  Computing power spectra ...")
        analyzer = core.PowerSpectrumAnalyzer(mesh, cfg.spectral_bins)
        spectra = {}
        for name in ["temperature", "qv", "surface_pressure"]:
            if name in ls_full and name in ss_full:
                spec_large = analyzer.compute_spectrum(ls_full[name], f"large_{name}")
                spec_small = analyzer.compute_spectrum(ss_full[name], f"small_{name}")
                spectra[name] = {'large': spec_large, 'small': spec_small}
        output_mgr.save_spectra(spectra)
    COMM.Barrier()

    with guard("write"), timed("gather + write blended state"):
        full_blended = core.gather_full_field(blended_owned, part.owned, mesh.nCells, dest_rank=0)
        if is_root():
            full_blended["u"] = u_background + ana_inc["u"]
            rlog(f"Saving blended states to {cfg.blended_file}")
            core.copy_and_overwrite(cfg.small_scale_file, cfg.blended_file, full_blended)
    COMM.Barrier()

    if is_root():
        logger.info("Successfully applied blending to MPAS fields")


# ============================================================
# CLI
# ============================================================
def main() -> None:
    config_file = "blending_config.yaml"
    if len(sys.argv) > 1:
        config_file = sys.argv[1]

    if os.path.exists(config_file):
        cfg = Config.from_yaml(config_file)
    else:
        cfg = Config()

    run(cfg)


if __name__ == "__main__":
    main()
