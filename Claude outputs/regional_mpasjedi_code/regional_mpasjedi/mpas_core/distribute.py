"""
Distribute a root-built state dict to every rank's owned+halo cell slice.

Shared by both apps: "root reads a file, builds a dict of full-mesh
arrays via some app-specific builder function, slices each rank's
owned+halo cells out of it, scatters" is identical machinery whether the
builder produces the filter's analysis-state fields or blending's
large/small-scale/background fields -- only the builder callback differs.
"""
from __future__ import annotations

import numpy as np
import netCDF4 as nc
from typing import Callable, Dict, Optional

from .mpi_env import COMM, is_root


def slice_axis_cells(data: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Slice a (nCells,), (Time,nCells) or (Time,nCells,nVertLevels) array
    along the nCells axis."""
    if data.ndim == 1:
        return data[idx]
    if data.ndim == 2:
        return data[:, idx]
    if data.ndim == 3:
        return data[:, idx, :]
    raise ValueError(f"Unsupported ndim={data.ndim}")


def load_and_distribute_state(
    file_path: str,
    builder: Callable[["nc.Dataset"], Dict[str, np.ndarray]],
    local_ids: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    Root reads `file_path`, calls builder(ds) -> full-mesh state dict,
    slices out each rank's owned+halo cells, and scatters -- nobody but
    root ever holds the full nCells-length arrays.

    Deadlock-safety: the gather and final scatter are unconditional
    collectives every rank always reaches; a root-side failure (bad file,
    missing variable, builder exception) is funnelled through one more
    unconditional bcast so it's raised on every rank together instead of
    leaving non-root ranks blocked forever on the scatter.
    """
    all_local_ids = COMM.gather(local_ids, root=0)

    parts: Optional[list] = None
    err: Optional[str] = None
    if is_root():
        try:
            with nc.Dataset(file_path) as ds:
                state_full = builder(ds)
            parts = [
                {name: slice_axis_cells(arr, ids) for name, arr in state_full.items()}
                for ids in all_local_ids
            ]
        except Exception as exc:
            err = str(exc)

    err = COMM.bcast(err, root=0)
    if err is not None:
        raise RuntimeError(f"load_state({file_path}): {err}")

    return COMM.scatter(parts, root=0)
