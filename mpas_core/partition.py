"""
METIS domain decomposition + halo exchange.

Shared by both apps: partitioning a mesh into per-rank cell ownership and
building the nearest-neighbour halo exchange needed to evaluate a local
Laplacian stencil is identical work regardless of what's ultimately done
with the result (multiscale diffusion bands vs. large/small-scale
blending).
"""
from __future__ import annotations

import os
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .mpi_env import COMM, RANK, SIZE, _HAS_MPI, is_root, logger
from .mesh import MPASMesh
from .cache import (
    mesh_signature, partition_cache_path, halo_cache_path,
    cache_try_load, cache_try_save,
)

try:
    from mpi4py import MPI
except ImportError:
    MPI = None

try:
    import pymetis
    _HAS_PYMETIS = True
except ImportError:
    _HAS_PYMETIS = False


# ============================================================
# METIS partitioning
# ============================================================
def _build_adjacency_list(mesh: MPASMesh) -> List[List[int]]:
    N  = mesh.nCells
    ne = mesh.nEdgesOnCell
    coc = mesh.cellsOnCell
    max_ne = coc.shape[1]

    i_idx = np.repeat(np.arange(N, dtype=np.int64), max_ne)
    k_idx = np.tile(np.arange(max_ne, dtype=np.int32), N)
    j_all = coc[i_idx, k_idx]
    valid = (k_idx < ne[i_idx]) & (j_all >= 0)

    i_v = i_idx[valid]
    j_v = j_all[valid]

    order = np.argsort(i_v, kind="stable")
    i_sorted = i_v[order]
    j_sorted = j_v[order]
    counts = np.bincount(i_sorted, minlength=N)
    splits = np.cumsum(counts)[:-1]
    groups = np.split(j_sorted, splits)
    return [g.tolist() for g in groups]


def build_or_load_partition(mesh: MPASMesh, cfg) -> np.ndarray:
    """
    Return a (nCells,) int32 array mapping each cell to its owning rank.
    Either read from cfg.partition_file, a disk cache, or computed fresh
    via pymetis on root and broadcast.

    Deadlock-safety: the online-METIS branch always reaches COMM.bcast()
    on every rank even when root's partitioning attempt raises -- the
    exception is captured into the broadcast payload instead of
    propagating immediately, so no rank is left waiting on a bcast root
    never issues.

    Cache note: the disk-cache hit/miss decision is a plain local file
    read (not a collective), so it's combined across ranks with one
    Allreduce (logical AND) before anyone commits to "load from cache" vs
    "run METIS together" -- keeps every rank on the same branch.
    """
    N = mesh.nCells

    if SIZE == 1:
        return np.zeros(N, dtype=np.int32)

    if cfg.partition_file:
        membership = np.loadtxt(cfg.partition_file, dtype=np.int64).astype(np.int32)
        if membership.shape[0] != N:
            raise ValueError(
                f"partition_file has {membership.shape[0]} entries, expected {N} (nCells)"
            )
        if membership.min() < 0 or membership.max() >= SIZE:
            raise ValueError(
                f"partition_file rank ids must be in [0,{SIZE}); "
                f"got range [{membership.min()},{membership.max()}]"
            )
        if is_root():
            counts = np.bincount(membership, minlength=SIZE)
            logger.info("Loaded partition from %s  cell counts per rank=%s",
                        cfg.partition_file, counts.tolist())
        return membership

    mesh_sig = mesh_signature(cfg.mesh_file, N)
    cache_path = partition_cache_path(cfg, mesh_sig)
    local_cached = cache_try_load(
        cache_path, cfg.use_cache, np.load,
        validate=lambda cand: cand.shape == (N,) and cand.min() >= 0 and cand.max() < SIZE,
        label="partition",
    )

    local_ok = local_cached is not None
    all_ok = COMM.allreduce(local_ok, op=MPI.LAND) if (_HAS_MPI and SIZE > 1) else local_ok
    if all_ok:
        membership = local_cached.astype(np.int32)
        if is_root():
            counts = np.bincount(membership, minlength=SIZE)
            logger.info("Loaded partition from cache %s  cell counts per rank=%s",
                        cache_path, counts.tolist())
        return membership
    if is_root() and cfg.use_cache and os.path.exists(cache_path):
        logger.warning("Ignoring stale/unreadable partition cache (mismatch on at "
                        "least one rank); recomputing with METIS on all ranks: %s", cache_path)

    payload: Tuple[Optional[np.ndarray], Optional[str]] = (None, None)
    if is_root():
        try:
            if not _HAS_PYMETIS:
                raise ImportError(
                    "pymetis is required for on-the-fly METIS partitioning "
                    "(pip install pymetis), or set partition_file in the "
                    "config to a precomputed gpmetis partition."
                )
            logger.info("Partitioning mesh with METIS into %d parts…", SIZE)
            adjacency = _build_adjacency_list(mesh)
            _, membership_list = pymetis.part_graph(SIZE, adjacency=adjacency)
            membership_arr = np.asarray(membership_list, dtype=np.int32)
            counts = np.bincount(membership_arr, minlength=SIZE)
            logger.info("METIS partition cell counts per rank: %s", counts.tolist())
            payload = (membership_arr, None)
        except Exception as exc:
            payload = (None, str(exc))
    membership, err = COMM.bcast(payload, root=0)
    if err is not None:
        raise RuntimeError(f"METIS partitioning failed: {err}")

    if is_root():
        cache_try_save(cache_path, membership, cfg.use_cache, np.save, "partition")

    return membership


@dataclass
class Partition:
    """Per-rank view of the cell partition: which cells this rank owns,
    which neighbouring "halo" cells (owned elsewhere) it needs to read for
    its stencil, and the send/receive maps needed to exchange them."""
    membership:      np.ndarray
    owned:           np.ndarray
    halo:            np.ndarray
    local_of_global: np.ndarray
    recv_from:       Dict[int, np.ndarray]
    send_to:         Dict[int, np.ndarray]

    @property
    def n_owned(self) -> int: return len(self.owned)

    @property
    def n_halo(self) -> int: return len(self.halo)


def build_partition(mesh: MPASMesh, membership: np.ndarray) -> Partition:
    """Vectorised construction of owned/halo sets and cross-rank
    send/receive maps -- no inter-rank communication needed (every rank
    already has full mesh connectivity + full membership)."""
    N  = mesh.nCells
    ne = mesh.nEdgesOnCell
    coc = mesh.cellsOnCell
    max_ne = coc.shape[1]

    owned = np.where(membership == RANK)[0].astype(np.int64)

    if owned.size == 0:
        halo = np.array([], dtype=np.int64)
        local_of_global = np.full(N, -1, dtype=np.int64)
        return Partition(membership, owned, halo, local_of_global, {}, {})

    i_idx = np.repeat(owned, max_ne)
    k_idx = np.tile(np.arange(max_ne, dtype=np.int32), len(owned))
    j_all = coc[i_idx, k_idx]
    valid = (k_idx < ne[i_idx]) & (j_all >= 0)
    i_v = i_idx[valid]
    j_v = j_all[valid]

    rj = membership[j_v]
    cross = rj != RANK
    i_cross = i_v[cross]
    j_cross = j_v[cross]
    r_cross = rj[cross]

    halo = np.unique(j_cross).astype(np.int64) if j_cross.size else np.array([], dtype=np.int64)

    recv_from: Dict[int, np.ndarray] = {}
    send_to:   Dict[int, np.ndarray] = {}
    for r in np.unique(r_cross).tolist():
        mask = r_cross == r
        recv_from[r] = np.unique(j_cross[mask]).astype(np.int64)
        send_to[r]   = np.unique(i_cross[mask]).astype(np.int64)

    local_of_global = np.full(N, -1, dtype=np.int64)
    local_of_global[owned] = np.arange(len(owned), dtype=np.int64)
    local_of_global[halo]  = len(owned) + np.arange(len(halo), dtype=np.int64)

    return Partition(membership, owned, halo, local_of_global, recv_from, send_to)


class HaloExchange:
    """Non-blocking nearest-neighbour halo exchange. send_pos is learned
    via a live handshake (Alltoall + Isend/Irecv) rather than inferred
    locally from part.send_to, because cellsOnCell isn't guaranteed
    symmetric at boundary cells on real (especially regional/limited-area)
    meshes -- inferring it locally can silently hang the exchange on a
    mismatched pair."""
    def __init__(self, part: Partition):
        self.n_owned = part.n_owned
        self.n_halo  = part.n_halo
        self.recv_pos: Dict[int, np.ndarray] = {
            r: part.local_of_global[ids] for r, ids in part.recv_from.items()
        }
        if _HAS_MPI and SIZE > 1:
            self.send_pos = self._build_send_pos_via_handshake(part)
        else:
            self.send_pos = {
                r: part.local_of_global[ids] for r, ids in part.send_to.items()
            }

    @classmethod
    def _from_positions(
        cls, n_owned: int, n_halo: int,
        recv_pos: Dict[int, np.ndarray], send_pos: Dict[int, np.ndarray],
    ) -> "HaloExchange":
        obj = cls.__new__(cls)
        obj.n_owned  = n_owned
        obj.n_halo   = n_halo
        obj.recv_pos = recv_pos
        obj.send_pos = send_pos
        return obj

    def _build_send_pos_via_handshake(self, part: Partition) -> Dict[int, np.ndarray]:
        req_counts = np.zeros(SIZE, dtype=np.int64)
        for r, ids in part.recv_from.items():
            req_counts[r] = len(ids)
        resp_counts = np.zeros(SIZE, dtype=np.int64)
        COMM.Alltoall(req_counts, resp_counts)

        reqs = []
        recv_bufs: Dict[int, np.ndarray] = {}
        for r in range(SIZE):
            if resp_counts[r] > 0:
                buf = np.empty(int(resp_counts[r]), dtype=np.int64)
                recv_bufs[r] = buf
                reqs.append(COMM.Irecv(buf, source=r, tag=43))
        for r, ids in part.recv_from.items():
            if len(ids) > 0:
                reqs.append(COMM.Isend(np.ascontiguousarray(ids), dest=r, tag=43))
        if reqs:
            MPI.Request.Waitall(reqs)

        return {r: part.local_of_global[ids] for r, ids in recv_bufs.items()}

    def exchange(self, u_local: np.ndarray) -> None:
        """Fill the halo entries of u_local (length n_owned+n_halo) in place."""
        if not _HAS_MPI or SIZE == 1:
            return
        if self.n_halo == 0 and not self.send_pos:
            return
        reqs = []
        recv_bufs: Dict[int, np.ndarray] = {}
        for r, pos in self.recv_pos.items():
            buf = np.empty(len(pos), dtype=np.float64)
            recv_bufs[r] = buf
            reqs.append(COMM.Irecv(buf, source=r, tag=42))
        for r, pos in self.send_pos.items():
            buf = np.ascontiguousarray(u_local[pos])
            reqs.append(COMM.Isend(buf, dest=r, tag=42))
        if reqs:
            MPI.Request.Waitall(reqs)
        for r, pos in self.recv_pos.items():
            u_local[pos] = recv_bufs[r]


def build_or_load_halo_exchange(part: Partition, cfg, mesh_sig: str, part_sig: str) -> "HaloExchange":
    if SIZE == 1:
        return HaloExchange(part)

    cache_path = halo_cache_path(cfg, mesh_sig, part_sig)
    recv_pos: Dict[int, np.ndarray] = {
        r: part.local_of_global[ids] for r, ids in part.recv_from.items()
    }

    def _load_send_pos(path):
        with np.load(path) as npz:
            return {int(r): npz[f"send_{r}"] for r in npz["ranks"].tolist()}

    local_send_pos = cache_try_load(cache_path, cfg.use_cache, _load_send_pos, label="halo")
    local_ok = local_send_pos is not None
    all_ok = COMM.allreduce(local_ok, op=MPI.LAND) if _HAS_MPI else local_ok

    if all_ok:
        logger.info("[rank %d] loaded halo send-map from cache: %s", RANK, cache_path)
        return HaloExchange._from_positions(part.n_owned, part.n_halo, recv_pos, local_send_pos)

    if is_root() and cfg.use_cache and os.path.exists(cache_path):
        logger.warning("Ignoring stale/unreadable halo cache (mismatch on at least "
                        "one rank); redoing the handshake on all ranks")

    halo = HaloExchange(part)

    def _save_send_pos(path, send_pos):
        np.savez(path, ranks=np.array(list(send_pos.keys()), dtype=np.int64),
                 **{f"send_{r}": arr for r, arr in send_pos.items()})

    cache_try_save(cache_path, halo.send_pos, cfg.use_cache, _save_send_pos, "halo")
    return halo


# ============================================================
# Gather owned-length local arrays back to full nCells arrays
# ============================================================
def gather_full_field(
    local_dict: Dict[str, np.ndarray],
    owned: np.ndarray,
    nCells: int,
    dest_rank: int = 0,
) -> Optional[Dict[str, np.ndarray]]:
    """
    Gather every rank's owned-length arrays into full nCells-length arrays
    on `dest_rank`. Unconditional collective -- call it identically on
    every rank, every time (skipping it on some ranks based on local
    conditions will hang the ranks still waiting on it).
    """
    gathered = COMM.gather((owned, local_dict), root=dest_rank)
    if RANK != dest_rank:
        return None
    merged: Dict[str, np.ndarray] = {}
    for owned_r, dict_r in gathered:
        for name, arr in dict_r.items():
            if name not in merged:
                if arr.ndim == 1:
                    full_shape: Tuple[int, ...] = (nCells,)
                elif arr.ndim == 2:
                    full_shape = (arr.shape[0], nCells)
                else:
                    full_shape = (arr.shape[0], nCells, arr.shape[2])
                merged[name] = np.zeros(full_shape, dtype=np.float64)
            if owned_r.size == 0:
                continue
            if arr.ndim == 1:
                merged[name][owned_r] = arr
            elif arr.ndim == 2:
                merged[name][:, owned_r] = arr
            else:
                merged[name][:, owned_r, :] = arr
    return merged
