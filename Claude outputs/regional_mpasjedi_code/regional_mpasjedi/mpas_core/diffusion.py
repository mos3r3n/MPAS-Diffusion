"""
Distributed diffusion filter: A = I + alpha*L^order, applied via
Jacobi-preconditioned CG -- local sparse mat-vec + halo exchange per
matvec, Allreduce for the two dot products per CG iteration. No rank ever
assembles or factorises a global matrix.

This is the ONE diffusion engine shared by both apps -- multiscale
decomposition (the filter) applies it once per scale break-point to build
a sequence of low-pass bands; blending applies it once per (large-scale,
small-scale) state to build the analysis increment. Same operator, same
solver, different callers.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from scipy import sparse
from typing import Optional, Tuple

from .mpi_env import COMM, SIZE, _HAS_MPI, MPI_SUM, MPI_MAX, is_root, logger
from .partition import HaloExchange


class DistributedDiffusionOperator:
    """A = I + alpha*L^order, applied via `order` rounds of (local sparse
    mat-vec + halo exchange) for the neighbour contributions owned by
    other ranks. L is a graph Laplacian with positive weights (SPD); L
    raised to any positive integer power is SPD too, so CG still applies.

    order=1 is plain diffusion ("del2"). order=2 is biharmonic ("del4")
    hyperdiffusion, order=4 is "del8", etc. Raising the order flattens the
    passband and sharpens the cutoff, at the cost of `order` halo
    exchanges per mat-vec instead of 1."""
    def __init__(self, L_local: sparse.csr_matrix, halo: HaloExchange, alpha: float, order: int = 1):
        if order < 1:
            raise ValueError(f"order must be >= 1, got {order}")
        self.L_local = L_local
        self.halo    = halo
        self.alpha   = alpha
        self.order   = order
        self.n_owned = halo.n_owned
        self.n_halo  = halo.n_halo
        if self.n_owned > 0:
            diag_owned = np.asarray(L_local[:, :self.n_owned].diagonal()).ravel()
        else:
            diag_owned = np.zeros(0)
        # diag_owned**order is the EXACT diagonal of L^order only for
        # order=1; for order>1 it's an approximate (but still nonnegative,
        # still SPD-safe) Jacobi preconditioner -- fine, Jacobi
        # preconditioning only needs to be a cheap valid SPD approximation.
        diag_Lp = diag_owned ** order
        self.diag_A = 1.0 + alpha * diag_Lp
        self.jacobi_inv = np.where(self.diag_A != 0.0, 1.0 / np.where(self.diag_A != 0, self.diag_A, 1.0), 1.0)
        self._v_local = np.empty(self.n_owned + self.n_halo, dtype=np.float64)

    def _apply_L_once(self, v_owned: np.ndarray) -> np.ndarray:
        self._v_local[:self.n_owned] = v_owned
        self.halo.exchange(self._v_local)
        return self.L_local @ self._v_local

    def _apply_L_power(self, v_owned: np.ndarray) -> np.ndarray:
        v = v_owned
        for _ in range(self.order):
            v = self._apply_L_once(v)
        return v

    def matvec(self, u_owned: np.ndarray, out: Optional[np.ndarray] = None) -> np.ndarray:
        Lp = self._apply_L_power(u_owned)
        if out is None:
            return u_owned + self.alpha * Lp
        np.multiply(Lp, self.alpha, out=out)
        out += u_owned
        return out


def _allreduce_sum(x: float) -> float:
    if not _HAS_MPI or SIZE == 1:
        return x
    return COMM.allreduce(x, op=MPI_SUM)


def _allreduce_max(x: float) -> float:
    if not _HAS_MPI or SIZE == 1:
        return x
    return COMM.allreduce(x, op=MPI_MAX)


def _global_dot(a: np.ndarray, b: np.ndarray) -> float:
    return _allreduce_sum(float(np.dot(a, b)))


@dataclass
class CGWorkspace:
    """Preallocated scratch arrays (r, z, p, Ap) for pcg_solve(), sized to
    one DiffusionFilter's operator -- build ONE workspace per scale and
    reuse it across every apply_column() call at that scale (potentially
    thousands: every variable x level x timestep x n_iter) instead of
    allocating four fresh owned-length arrays every call."""
    r:  np.ndarray
    z:  np.ndarray
    p:  np.ndarray
    Ap: np.ndarray

    @classmethod
    def for_size(cls, n: int) -> "CGWorkspace":
        return cls(
            r=np.empty(n, dtype=np.float64),
            z=np.empty(n, dtype=np.float64),
            p=np.empty(n, dtype=np.float64),
            Ap=np.empty(n, dtype=np.float64),
        )


def pcg_solve(
    op: DistributedDiffusionOperator,
    b: np.ndarray,
    x0: Optional[np.ndarray] = None,
    tol: float = 1e-8,
    maxiter: int = 200,
    workspace: Optional[CGWorkspace] = None,
) -> Tuple[np.ndarray, int, bool]:
    """
    Distributed Jacobi-preconditioned CG solve of op @ x == b, where `b`
    and the returned `x` are this rank's OWNED-length vectors. Every
    matvec triggers a halo exchange; every dot product triggers an
    Allreduce.

    Returns (x, iters, converged). `converged=False` means maxiter was hit
    before res_norm/b_norm < tol -- the returned x is NOT accurate to
    `tol` in that case.

    IMPORTANT: every rank must call this the same number of times, in the
    same order, as every other rank -- _global_dot() is a real MPI
    collective; a rank with zero owned cells for this variable must still
    participate with size-0 arrays (numpy handles that fine) rather than
    returning early.
    """
    ws = workspace if workspace is not None else CGWorkspace.for_size(b.shape[0])
    r, z, p, Ap = ws.r, ws.z, ws.p, ws.Ap

    x = np.array(b if x0 is None else x0, dtype=np.float64, copy=True)

    b_norm = np.sqrt(_global_dot(b, b))
    if b_norm == 0.0:
        return np.zeros_like(b), 0, True

    op.matvec(x, out=r)
    np.subtract(b, r, out=r)
    np.multiply(op.jacobi_inv, r, out=z)
    np.copyto(p, z)
    rz_old = _global_dot(r, z)

    converged = False
    it = 0
    for it in range(maxiter):
        res_norm = np.sqrt(_global_dot(r, r))
        if res_norm / b_norm < tol:
            converged = True
            break
        op.matvec(p, out=Ap)
        pAp = _global_dot(p, Ap)
        if pAp == 0.0:
            break
        alpha = rz_old / pAp
        x += alpha * p
        r -= alpha * Ap
        np.multiply(op.jacobi_inv, r, out=z)
        rz_new = _global_dot(r, z)
        beta = rz_new / rz_old if rz_old != 0.0 else 0.0
        p *= beta
        p += z
        rz_old = rz_new

    return x, it, converged


def effective_scale_km(
    scale_km: float, n_iter: int, hyper_order: int = 1, calibrate: str = "gaussian",
) -> float:
    """The filter's actual half-power cutoff wavelength in km, directly
    comparable to a Raymond (1988) filter's cutoff wavelength.

    calibrate="cutoff": scale_km already IS that wavelength by
    construction (identity).

    calibrate="gaussian" (only defined for hyper_order=1): scale_km is a
    Gaussian smoothing radius (sigma), NOT a cutoff wavelength -- the
    actual half-power wavelength is several times larger (approaching
    2*pi*sqrt(2*ln2) ~= 5.34 x scale_km as n_iter -> infinity)."""
    if calibrate == "cutoff":
        return scale_km
    if calibrate != "gaussian":
        raise ValueError(f"unknown calibrate={calibrate!r}, expected 'gaussian' or 'cutoff'")
    if hyper_order != 1:
        raise ValueError("calibrate='gaussian' only defined for hyper_order=1")
    sigma = scale_km * 1000.0
    alpha = sigma ** 2 / (2.0 * n_iter)
    k_half = ((2.0 ** (1.0 / n_iter) - 1.0) / alpha) ** (1.0 / (2 * hyper_order))
    return 2.0 * np.pi / k_half / 1000.0


class DiffusionFilter:
    def __init__(
        self,
        L_local: sparse.csr_matrix,
        halo: HaloExchange,
        length_scale_km: float,
        n_iter: int = 8,
        pcg_tol: float = 1e-8,
        pcg_maxiter: int = 200,
        hyper_order: int = 1,
        calibrate: str = "gaussian",
    ):
        self.n_iter      = n_iter
        self.pcg_tol     = pcg_tol
        self.pcg_maxiter = pcg_maxiter
        k_c = 2.0 * np.pi / (length_scale_km * 1000.0)
        if calibrate == "gaussian":
            if hyper_order != 1:
                raise ValueError(
                    "calibrate='gaussian' is only valid for hyper_order=1 "
                    "(that's specifically what makes the filter converge "
                    "to a Gaussian as n_iter -> infinity); use "
                    "calibrate='cutoff' for hyper_order > 1."
                )
            sigma = length_scale_km * 1_000.0
            alpha = sigma ** 2 / (2.0 * n_iter)
        elif calibrate == "cutoff":
            alpha = (2.0 ** (1.0 / n_iter) - 1.0) / k_c ** (2 * hyper_order)
        else:
            raise ValueError(f"unknown calibrate={calibrate!r}, expected 'gaussian' or 'cutoff'")
        self.op = DistributedDiffusionOperator(L_local, halo, alpha, order=hyper_order)
        self._workspace = CGWorkspace.for_size(self.op.n_owned)
        self.length_scale_km = length_scale_km
        self.hyper_order = hyper_order
        self._solve_count = 0
        self._iter_sum = 0
        self._iter_max = 0
        self._n_not_converged = 0
        if is_root():
            eff_km = effective_scale_km(length_scale_km, n_iter, hyper_order, calibrate)
            gap_note = "" if calibrate == "cutoff" else "  (nominal scale != cutoff wavelength under calibrate='gaussian')"
            logger.info(
                "DiffusionFilter  scale=%g km  order=%d  calibrate=%s  alpha=%.3e  "
                "effective half-power wavelength=%.1f km%s  (distributed PCG, tol=%.1e, maxiter=%d)",
                length_scale_km, hyper_order, calibrate, alpha, eff_km, gap_note, pcg_tol, pcg_maxiter,
            )

    def apply_column(self, x: np.ndarray) -> np.ndarray:
        """Apply n_iter implicit PCG solves to this rank's owned-length vector."""
        u = np.asarray(x, dtype=np.float64).copy()
        for _ in range(self.n_iter):
            u, iters, converged = pcg_solve(
                self.op, u, x0=u, tol=self.pcg_tol, maxiter=self.pcg_maxiter,
                workspace=self._workspace,
            )
            self._solve_count += 1
            self._iter_sum += iters
            self._iter_max = max(self._iter_max, iters)
            if not converged:
                self._n_not_converged += 1
        return u

    def log_convergence_summary(self) -> None:
        """Log aggregate PCG convergence stats across every solve done
        with this filter so far. Call once per scale (after the low-pass
        sweep for that scale finishes)."""
        if self._solve_count == 0 or not is_root():
            return
        avg_iters = self._iter_sum / self._solve_count
        if self._n_not_converged > 0:
            logger.warning(
                "DiffusionFilter scale=%g km order=%d: %d/%d PCG solves did NOT "
                "converge within maxiter=%d (avg iters=%.1f, max iters=%d). "
                "Results at this scale/order are likely inaccurate -- raise "
                "pcg_maxiter, loosen pcg_tol, or reduce hyper_order.",
                self.length_scale_km, self.hyper_order, self._n_not_converged,
                self._solve_count, self.pcg_maxiter, avg_iters, self._iter_max,
            )
        else:
            logger.info(
                "DiffusionFilter scale=%g km order=%d: all %d PCG solves converged "
                "(avg iters=%.1f, max iters=%d, maxiter=%d).",
                self.length_scale_km, self.hyper_order, self._solve_count,
                avg_iters, self._iter_max, self.pcg_maxiter,
            )


def apply_filter(filt: DiffusionFilter, data: np.ndarray) -> np.ndarray:
    """
    data shape (local, owned-cells-only):
      (n_owned,)                   -> single column
      (Time, n_owned)               -> loop over time
      (Time, n_owned, nVertLevels) -> loop over time x levels
    """
    if data.ndim == 1:
        return filt.apply_column(data)
    if data.ndim == 2:
        out = np.empty_like(data, dtype=np.float64)
        for t in range(data.shape[0]):
            out[t] = filt.apply_column(data[t])
        return out
    if data.ndim == 3:
        out = np.empty_like(data, dtype=np.float64)
        for t in range(data.shape[0]):
            for lev in range(data.shape[2]):
                out[t, :, lev] = filt.apply_column(data[t, :, lev])
        return out
    raise ValueError(f"Unsupported data.ndim={data.ndim}; expected 1, 2, or 3.")
