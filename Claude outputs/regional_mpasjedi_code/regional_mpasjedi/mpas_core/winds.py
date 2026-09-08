"""
Cell-centre (u, v) wind -> edge-normal wind projection, domain-decomposed.

Each edge's value is the SUM of two 0.5-weighted contributions, one from
each of its two neighbour cells. Since different ranks own disjoint sets
of cells, each rank computes the contribution from ONLY its owned cells
into a full-size (Time, nEdges, nVertLevels) buffer (zero elsewhere), and
an MPI Allreduce-sum across ranks reassembles the exact result a single-
process computation would produce -- every edge ends up with contributions
from both its neighbours' owning ranks summed together. This needs no
edge-based halo/send-map machinery beyond the plain elementwise Allreduce
-- justified because this runs once per blending call, not inside an
iterative solve.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse

from .mpi_env import COMM, _HAS_MPI, SIZE, MPI_SUM


def r3_normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / norm


def uv_cell_to_edges(
    u_owned, v_owned, lonCell_owned, latCell_owned,
    edgeNormalVectors, nEdgesOnCell_owned, edgesOnCell_owned, nEdges,
):
    """
    Project (u, v) cell-centre winds onto edge normals, distributed by
    cell ownership. Every array with an "_owned" suffix must already be
    sliced to this rank's owned cells only (same length, same order);
    edgeNormalVectors and nEdges are global (full mesh), since
    edgesOnCell_owned's entries are global edge indices.

    Returns the SAME full (Time, nEdges, nVertLevels) array on every rank
    -- this is a collective; every rank must call it, including ranks
    that own zero cells (they contribute an all-zero local buffer, a
    correct identity element for the Allreduce sum).
    """
    u_owned = np.asarray(u_owned, dtype=np.float64)
    v_owned = np.asarray(v_owned, dtype=np.float64)
    lon = np.asarray(lonCell_owned, dtype=np.float64)
    lat = np.asarray(latCell_owned, dtype=np.float64)
    env = np.asarray(edgeNormalVectors, dtype=np.float64)
    nTimes, n_owned, nVertLevels = u_owned.shape

    local = np.zeros((nTimes, nEdges, nVertLevels), dtype=np.float64)
    if n_owned > 0:
        east = r3_normalize(np.stack([-np.sin(lon), np.cos(lon), np.zeros(n_owned)], -1))
        north = r3_normalize(np.stack([
            -np.cos(lon) * np.sin(lat), -np.sin(lon) * np.sin(lat), np.cos(lat)], -1))
        for t in range(nTimes):
            u_t = u_owned[t]
            v_t = v_owned[t]
            rows_all, cols_all, data_all = [], [], []
            for iCell in range(n_owned):
                ne = nEdgesOnCell_owned[iCell]
                if ne == 0:
                    continue
                edges = edgesOnCell_owned[iCell, :ne]
                dot_e = env[edges] @ east[iCell]
                dot_n = env[edges] @ north[iCell]
                contrib = (0.5 * dot_e[:, None] * u_t[iCell][None, :]
                         + 0.5 * dot_n[:, None] * v_t[iCell][None, :])
                rows_all.append(np.repeat(edges, nVertLevels))
                cols_all.append(np.tile(np.arange(nVertLevels), ne))
                data_all.append(contrib.ravel())
            if data_all:
                sp = sparse.coo_matrix(
                    (np.concatenate(data_all),
                     (np.concatenate(rows_all), np.concatenate(cols_all))),
                    shape=(nEdges, nVertLevels)
                )
                local[t] = sp.toarray()

    if not _HAS_MPI or SIZE == 1:
        return local

    total = np.empty_like(local)
    COMM.Allreduce(local, total, op=MPI_SUM)
    return total
