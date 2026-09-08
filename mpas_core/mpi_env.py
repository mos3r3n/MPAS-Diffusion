"""
MPI bootstrap + logging + collective-safety helpers, shared by every module
in this package. Ported/upgraded from mpas_diffsion_filters.py's bootstrap
(same repo, same MPI conventions) so both codebases behave identically under
a missing mpi4py, a singleton launch, or a mid-run exception on one rank.

Everything here is process-global state initialised at import time — COMM/
RANK/SIZE never change after that, matching the convention already used
throughout mpas_diffsion_filters.py.
"""
from __future__ import annotations

import sys
import time
import logging
from contextlib import contextmanager
from typing import Optional

try:
    from mpi4py import MPI
    _HAS_MPI = True
except ImportError:
    _HAS_MPI = False


# ============================================================
# MPI bootstrap (falls back gracefully if mpi4py absent)
# ============================================================
class _FallbackComm:
    """Trivial single-process stand-in used only when mpi4py isn't
    importable at all. If mpi4py IS importable but the launcher didn't
    actually join this process to a shared communicator, MPI.COMM_WORLD
    still reports rank=0/size=1 on its own -- see the SIZE==1 warning in
    driver.py's run functions, which covers that case too."""
    rank = 0
    size = 1
    def Get_rank(self): return 0
    def Get_size(self): return 1
    def Barrier(self): pass
    def bcast(self, obj, root=0): return obj
    def scatter(self, data, root=0): return data[0] if data else None
    def gather(self, data, root=0): return [data]
    def allreduce(self, sendobj, op=None): return sendobj
    def Allreduce(self, sendbuf, recvbuf, op=None):
        import numpy as np
        np.copyto(recvbuf, sendbuf)
    def Alltoall(self, sendbuf, recvbuf):
        import numpy as np
        np.copyto(recvbuf, sendbuf)
    def Irecv(self, *a, **kw): raise RuntimeError("Irecv unavailable without mpi4py (SIZE=1, shouldn't be called)")
    def Isend(self, *a, **kw): raise RuntimeError("Isend unavailable without mpi4py (SIZE=1, shouldn't be called)")
    def Abort(self, code=1): raise SystemExit(code)


if _HAS_MPI:
    COMM    = MPI.COMM_WORLD
    RANK    = COMM.Get_rank()
    SIZE    = COMM.Get_size()
    MPI_SUM = MPI.SUM
    MPI_LAND = MPI.LAND
    MPI_MAX = MPI.MAX
else:
    COMM    = _FallbackComm()
    RANK    = 0
    SIZE    = 1
    MPI_SUM = None
    MPI_LAND = None
    MPI_MAX = None

IS_ROOT = (RANK == 0)


def is_root() -> bool:
    return RANK == 0


# ============================================================
# MPI-safe abort: broadcast any exception to all ranks
# ============================================================
def mpi_abort(msg: str) -> None:
    """Log and hard-kill all ranks uniformly."""
    logger.error("ABORT [rank %d]: %s", RANK, msg)
    if _HAS_MPI:
        MPI.COMM_WORLD.Abort(1)
    sys.exit(1)


@contextmanager
def guard(stage: str):
    """
    Wrap a code block so that if ANY rank raises, the exception message is
    broadcast and all ranks call mpi_abort -- preventing Barrier hangs.

    Only protects blocks where every rank reaches the SAME set of
    collective calls regardless of success/failure. Blocks that do
    asymmetric work (root-only I/O followed by a bcast/scatter every rank
    must call) need their own local try/except-then-broadcast handling
    inside the block instead -- see blend_io.load_and_distribute_state()
    for that pattern.
    """
    error: Optional[str] = None
    try:
        yield
    except Exception as exc:
        error = f"{stage}: {exc}"
    all_errors = COMM.gather(error, root=0)
    first_error: Optional[str] = None
    if is_root():
        first_error = next((e for e in all_errors if e is not None), None)
    first_error = COMM.bcast(first_error, root=0)
    if first_error is not None:
        mpi_abort(first_error)


# ============================================================
# Logging
# ============================================================
_logging_configured = False


def setup_logging(level_str: str = "INFO") -> None:
    """Idempotent -- safe to call from both run() and directly from a
    driver function called on its own (e.g. in a test), without doubling
    up log lines from a second StreamHandler."""
    global _logging_configured
    if _logging_configured:
        return
    level = getattr(logging, level_str.upper(), logging.INFO)
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    fmt = f"%(asctime)s [rank {RANK:03d}] %(levelname)s - %(message)s"
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(fmt))
    # Non-root ranks only surface WARNING+ by default, same convention as
    # mpas_diffsion_filters.py -- call rlog()/logger.warning() explicitly
    # for anything that must be visible from every rank.
    handler.setLevel(level if is_root() else logging.WARNING)
    root_logger.addHandler(handler)
    _logging_configured = True


logger = logging.getLogger("mpas_blending")


def rlog(msg, *args, level: int = logging.INFO) -> None:
    """Log only from rank 0 -- kept for compatibility with the original
    script's calling convention (rlog("...")) used all through driver.py's
    non-distributed path."""
    if IS_ROOT:
        logger.log(level, msg, *args)


# ============================================================
# Timing helper
# ============================================================
@contextmanager
def timed(label: str):
    t0 = time.perf_counter()
    yield
    if is_root():
        logger.info("  %-40s %.1f s", label, time.perf_counter() - t0)
