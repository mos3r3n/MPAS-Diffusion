"""
Local (per-rank) conservative graph Laplacian for the diffusion operator.

Each rank builds only the rows for its OWNED cells, with columns spanning
owned+halo -- this is the only Laplacian either app needs now that
blending shares the filter's distributed-PCG engine (see diffusion.py):
neither app ever assembles a full (nCells, nCells) matrix.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse

from .mpi_env import RANK, logger
from .mesh import MPASMesh
from .partition import Partition
from .cache import laplacian_cache_path, cache_try_load, cache_try_save


def build_local_laplacian(mesh: MPASMesh, part: Partition) -> sparse.csr_matrix:
    """
    Vectorised conservative-Laplacian weight formula, restricted to this
    rank's owned rows, with neighbour columns remapped to LOCAL indices
    (owned cells first, then halo cells) via part.local_of_global.
    Shape: (n_owned, n_owned + n_halo).
    """
    logger.info("[rank %d] building local Laplacian (%d owned cells)…", RANK, part.n_owned)
    n_owned = part.n_owned
    n_total = n_owned + part.n_halo
    if n_owned == 0:
        return sparse.csr_matrix((0, n_total))

    ne = mesh.nEdgesOnCell
    coc = mesh.cellsOnCell
    eoc = mesh.edgesOnCell
    max_ne = coc.shape[1]

    owned = part.owned
    i_idx = np.repeat(owned, max_ne)
    k_idx = np.tile(np.arange(max_ne, dtype=np.int32), len(owned))
    j_all = coc[i_idx, k_idx]
    e_all = eoc[i_idx, k_idx]
    valid = (k_idx < ne[i_idx]) & (j_all >= 0)

    i_v = i_idx[valid]
    j_v = j_all[valid]
    e_v = e_all[valid]

    dc = mesh.dcEdge[e_v]
    dv = mesh.dvEdge[e_v] if mesh.dvEdge is not None else dc
    w  = dv / (dc * mesh.areaCell[i_v])

    row_local = part.local_of_global[i_v]
    col_local = part.local_of_global[j_v]

    rows = row_local.tolist(); cols = col_local.tolist(); vals = (-w).tolist()

    diag = np.bincount(row_local, weights=w, minlength=n_owned)
    rows.extend(range(n_owned)); cols.extend(range(n_owned)); vals.extend(diag)

    L_local = sparse.csr_matrix((vals, (rows, cols)), shape=(n_owned, n_total))
    logger.info("[rank %d] local Laplacian nnz=%d", RANK, L_local.nnz)
    return L_local


def build_or_load_local_laplacian(
    mesh: MPASMesh, part: Partition, cfg, mesh_sig: str, part_sig: str,
) -> sparse.csr_matrix:
    """Cache-aware wrapper -- no Allreduce agreement needed since this is
    purely local work: build_local_laplacian() does zero inter-rank
    communication, so one rank falling back to a rebuild while another
    loads from cache has no effect on any other rank's control flow."""
    cache_path = laplacian_cache_path(cfg, mesh_sig, part_sig)
    expected_shape = (part.n_owned, part.n_owned + part.n_halo)
    L_local = cache_try_load(
        cache_path, cfg.use_cache,
        loader=lambda p: sparse.load_npz(p).tocsr(),
        validate=lambda L: L.shape == expected_shape,
        label="local Laplacian",
    )
    if L_local is not None:
        logger.info("[rank %d] loaded local Laplacian from cache (nnz=%d): %s", RANK, L_local.nnz, cache_path)
        return L_local

    L_local = build_local_laplacian(mesh, part)
    cache_try_save(cache_path, L_local, cfg.use_cache, sparse.save_npz, "local Laplacian")
    return L_local
