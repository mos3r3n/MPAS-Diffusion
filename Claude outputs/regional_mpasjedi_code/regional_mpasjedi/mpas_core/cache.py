"""
Disk cache for mesh/partition-derived artifacts.

Three things built at startup depend ONLY on (mesh_file, SIZE) or on
(mesh_file, SIZE, the resulting partition):

  1. the METIS partition (membership: cell -> owning rank)
  2. the local Laplacian each rank builds for its own owned cells
  3. the halo "send-map" each rank learns via an MPI handshake

They come out byte-identical every time for the same mesh + rank count, so
caching them on disk (keyed off a signature of the mesh file: path+size+
mtime+nCells, plus the rank count and, for 2/3, a hash of the partition
membership) turns a repeated run into a cache hit instead of a rebuild.

Shared by both apps under ONE cache directory (`.mpas_core_cache`, next to
the mesh file by default): the filter and blending apps both partition the
SAME mesh file when pointed at it, so if they're run back to back with the
same rank count they now reuse each other's cached partition/Laplacian/
halo instead of only their own -- a direct benefit of the two apps sharing
this library instead of each keeping a separate cache.
"""
from __future__ import annotations

import os
import hashlib
import numpy as np

from .mpi_env import RANK, SIZE, logger


def mesh_signature(mesh_file: str, nCells: int) -> str:
    st = os.stat(mesh_file)
    key = f"{os.path.abspath(mesh_file)}|{st.st_size}|{int(st.st_mtime)}|nCells={nCells}"
    return hashlib.sha1(key.encode()).hexdigest()[:16]


def partition_signature(membership: np.ndarray) -> str:
    return hashlib.sha1(np.ascontiguousarray(membership).tobytes()).hexdigest()[:16]


def cache_dir(cfg) -> str:
    return getattr(cfg, "cache_dir", None) or os.path.join(
        os.path.dirname(os.path.abspath(cfg.mesh_file)), ".mpas_core_cache"
    )


def partition_cache_path(cfg, mesh_sig: str) -> str:
    return os.path.join(cache_dir(cfg), f"partition_{mesh_sig}_np{SIZE}.npy")


def laplacian_cache_path(cfg, mesh_sig: str, part_sig: str) -> str:
    return os.path.join(cache_dir(cfg), f"laplacian_{mesh_sig}_{part_sig}_np{SIZE}_r{RANK}.npz")


def halo_cache_path(cfg, mesh_sig: str, part_sig: str) -> str:
    return os.path.join(cache_dir(cfg), f"halo_{mesh_sig}_{part_sig}_np{SIZE}_r{RANK}.npz")


def cache_try_load(cache_path: str, use_cache: bool, loader, validate=None, label: str = ""):
    """Best-effort cache read: returns the loaded object, or None if
    caching is off, the file is missing, reading raises, or `validate`
    rejects it. Never raises -- callers decide what a miss means."""
    if not use_cache or not os.path.exists(cache_path):
        return None
    try:
        obj = loader(cache_path)
    except Exception as exc:
        logger.warning("[rank %d] failed to read %s cache %s: %s", RANK, label, cache_path, exc)
        return None
    if validate is not None and not validate(obj):
        logger.warning("[rank %d] ignoring invalid %s cache: %s", RANK, label, cache_path)
        return None
    return obj


def cache_try_save(cache_path: str, obj, use_cache: bool, saver, label: str = "") -> None:
    """Best-effort cache write: logs and swallows any failure (disk full,
    read-only mount, ...) rather than failing the run over a cache miss."""
    if not use_cache:
        return
    try:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        saver(cache_path, obj)
        logger.info("[rank %d] saved %s to cache: %s", RANK, label, cache_path)
    except Exception as exc:
        logger.warning("[rank %d] failed to write %s cache %s: %s", RANK, label, cache_path, exc)
