"""
MPAS mesh loading -- shared by the diffusion-filter and blending apps.

Every rank reads the mesh file directly (no root-reads-then-broadcasts
step): every rank needs the full connectivity anyway, to build its own
local Laplacian and work out its own halo/send maps against the METIS
partition, so reading locally avoids pickling a mesh full of big arrays
through bcast.

`edgeNormalVectors` and `zgrid` are optional and only read when
`load_edges=True`. The diffusion filter never needs them (it only ever
touches cell-centered fields), so it uses the default `load_edges=False`
and doesn't require its mesh file to even contain those variables.
Blending needs both (edgeNormalVectors for wind-to-edge projection,
zgrid for hydrostatic balance), so it passes `load_edges=True`.
"""
from __future__ import annotations

import numpy as np
import netCDF4 as nc
from dataclasses import dataclass
from typing import Optional

from .mpi_env import is_root, logger
from .constants import RAD2DEG


@dataclass
class MPASMesh:
    nCells:             int
    nEdgesOnCell:       np.ndarray
    cellsOnCell:        np.ndarray
    areaCell:           np.ndarray
    latCell:            np.ndarray
    lonCell:            np.ndarray
    dcEdge:             np.ndarray
    dvEdge:             Optional[np.ndarray]
    edgesOnCell:        np.ndarray
    # Only populated when load_mpas_mesh(..., load_edges=True) -- see
    # module docstring.
    nEdges:             Optional[int] = None
    edgeNormalVectors:  Optional[np.ndarray] = None
    zgrid:              Optional[np.ndarray] = None

    def __post_init__(self):
        self.areaCell = self.areaCell.astype(np.float64)

    @property
    def lat_deg(self): return self.latCell * RAD2DEG

    @property
    def lon_deg(self): return self.lonCell * RAD2DEG


def load_mpas_mesh(mesh_file: str, load_edges: bool = False) -> MPASMesh:
    """Every rank reads the mesh directly -- see module docstring for why.

    load_edges=True additionally reads edgeNormalVectors and zgrid (and
    nEdges), deriving edgeNormalVectors from cell/edge geometry if the
    file has it present but all-zero (some MPAS mesh generators leave it
    unpopulated)."""
    logger.info("Loading mesh: %s", mesh_file)
    with nc.Dataset(mesh_file) as ds:
        mesh = MPASMesh(
            nCells       = len(ds.dimensions["nCells"]),
            nEdgesOnCell = ds.variables["nEdgesOnCell"][:].astype(np.int32),
            cellsOnCell  = ds.variables["cellsOnCell"][:].astype(np.int32) - 1,
            areaCell     = ds.variables["areaCell"][:],
            latCell      = ds.variables["latCell"][:],
            lonCell      = ds.variables["lonCell"][:],
            dcEdge       = ds.variables["dcEdge"][:],
            dvEdge       = (ds.variables["dvEdge"][:]
                            if "dvEdge" in ds.variables else None),
            edgesOnCell  = ds.variables["edgesOnCell"][:].astype(np.int32) - 1,
        )
        if load_edges:
            mesh.nEdges = len(ds.dimensions["nEdges"])
            mesh.edgeNormalVectors = ds.variables["edgeNormalVectors"][:]
            mesh.zgrid = ds.variables["zgrid"][:]
            if np.all(mesh.edgeNormalVectors == 0.0):
                logger.info("  edgeNormalVectors all zero -- deriving from geometry")
                mesh.edgeNormalVectors = _derive_edge_normals(ds)
    if is_root():
        logger.info("nCells = %d%s", mesh.nCells,
                     f"  nEdges = {mesh.nEdges}" if mesh.nEdges is not None else "")
    return mesh


def _derive_edge_normals(ds) -> np.ndarray:
    """Vectorised edge-normal derivation, used when the mesh file's
    edgeNormalVectors field is present but all-zero."""
    nCells = len(ds.dimensions["nCells"])
    nEdges = len(ds.dimensions["nEdges"])
    xC = ds.variables["xCell"][:]
    yC = ds.variables["yCell"][:]
    zC = ds.variables["zCell"][:]
    xE = ds.variables["xEdge"][:]
    yE = ds.variables["yEdge"][:]
    zE = ds.variables["zEdge"][:]
    coe = ds.variables["cellsOnEdge"][:].astype(np.int64)
    c1 = coe[:, 0]
    c2 = coe[:, 1]
    cell_xyz = np.stack([xC, yC, zC], axis=-1)
    edge_xyz = np.stack([xE, yE, zE], axis=-1)
    interior = (c1 >= 1) & (c1 <= nCells) & (c2 >= 1) & (c2 <= nCells)
    miss1    = (~interior) & (c2 >= 1) & (c2 <= nCells)
    miss2    = (~interior) & (c1 >= 1) & (c1 <= nCells)
    vec = np.zeros((nEdges, 3), dtype=np.float64)
    idx1 = np.clip(c1[interior] - 1, 0, nCells - 1)
    idx2 = np.clip(c2[interior] - 1, 0, nCells - 1)
    vec[interior] = cell_xyz[idx2] - cell_xyz[idx1]
    idx2m = np.clip(c2[miss1] - 1, 0, nCells - 1)
    vec[miss1] = cell_xyz[idx2m] - edge_xyz[miss1]
    idx1m = np.clip(c1[miss2] - 1, 0, nCells - 1)
    vec[miss2] = edge_xyz[miss2] - cell_xyz[idx1m]
    norm = np.linalg.norm(vec, axis=-1, keepdims=True)
    norm = np.where(norm < 1e-30, 1.0, norm)
    return vec / norm
