"""
Physics transforms shared by both apps.

All per-cell/per-column computations with no dependency on neighbouring
cells, so they work identically whether the cell axis below is the FULL
mesh or one rank's OWNED-cell slice -- no halo exchange is needed for
anything in this module.
"""
from __future__ import annotations

import numpy as np
import netCDF4 as nc
from dataclasses import dataclass
from typing import Optional

from .constants import P0, RD_CP, GRAVITY, RD, EPS


def theta_to_temperature(theta, pressure):
    return theta * (pressure / P0) ** RD_CP


def compute_pressure(pb, pp):
    return pb + pp


def qv_to_spechum(qv):
    return qv / (1.0 + qv)


def compute_surface_pressure(
    pressure: np.ndarray,   # (Time, ncells, nVertLevels) -- full pressure at mass levels
    zgrid: np.ndarray,      # (ncells, nVertLevelsP1) -- interface heights; index 0 = terrain
    theta: np.ndarray,      # (Time, ncells, nVertLevels) -- potential temperature
    qv: np.ndarray,         # (Time, ncells, nVertLevels) -- water-vapor mixing ratio (kg/kg)
) -> np.ndarray:
    """
    Diagnose surface pressure (Time, ncells) when it isn't in the data
    file, by hydrostatically extrapolating the lowest model (mass)
    level's pressure down to the surface -- the hypsometric equation:

        P_sfc = P1 * exp( g * (z1 - z_sfc) / (Rd * Tv1) )

    where level 1 is the lowest mass level, z_sfc = zgrid[:, 0] is the
    terrain height (bottom interface), z1 is the height of the lowest
    mass level's midpoint, and Tv1 is the VIRTUAL temperature there --
    moist air is less dense than dry air at the same (T, P), so using
    plain T instead of Tv systematically overstates the extrapolated
    surface pressure. Tv is computed from T (theta_to_temperature) and qv
    via the exact mixing-ratio form Tv = T*(qv+eps)/(eps*(1+qv))
    (eps = Rd/Rv), not the common Tv ~= T*(1+0.608*qv) linearization.

    This is a single-layer extrapolation over the half-thickness of the
    lowest model layer (z1 to z_sfc), standard/sufficient for a surface-
    pressure diagnostic -- NOT the same problem as reducing to sea-level
    pressure over elevated terrain (that needs a more careful multi-step
    reduction because the extrapolation distance can be large).
    """
    z_sfc = zgrid[:, 0]                            # (ncells,) terrain height
    z1    = 0.5 * (zgrid[:, 0] + zgrid[:, 1])       # (ncells,) lowest mass-level height
    dz    = (z1 - z_sfc)[None, :]                   # (1, ncells), broadcasts over Time

    p1     = pressure[:, :, 0]                      # (Time, ncells)
    theta1 = theta[:, :, 0]
    qv1    = qv[:, :, 0]

    T1  = theta_to_temperature(theta1, p1)
    Tv1 = T1 * (qv1 + EPS) / (EPS * (1.0 + qv1))

    return p1 * np.exp(GRAVITY * dz / (RD * Tv1))


def load_zgrid_with_fallback(mesh_file: str, ds: "nc.Dataset") -> Optional[np.ndarray]:
    """
    zgrid (vertical grid interface heights) is static/time-invariant in
    MPAS -- it lives in the mesh/invariant file alongside the rest of the
    static grid connectivity, rather than in the time-varying data file.
    Read it from there; fall back to looking for zgrid directly in `ds`
    (the already-open data file) for setups that keep it there instead.
    Returns None (never raises) if zgrid isn't found either way -- the
    caller decides what a miss means.

    Only needed by callers that load the mesh with load_edges=False (so
    mesh.zgrid is None) but still want zgrid for something like
    compute_surface_pressure() above -- if the mesh was already loaded
    with load_edges=True, just use mesh.zgrid directly instead of this.
    """
    with nc.Dataset(mesh_file) as mesh_ds:
        if "zgrid" in mesh_ds.variables:
            return mesh_ds.variables["zgrid"][:]
    return ds.variables["zgrid"][:] if "zgrid" in ds.variables else None


@dataclass
class HydroConstants:
    rgas: float = 287.0
    rv: float = 461.6
    cp: float = 1004.5
    gravity: float = 9.80616
    p0: float = 100000.0

    @property
    def rvordm1(self): return self.rv / self.rgas - 1.0

    @property
    def kappa(self): return self.rgas / self.cp


_DEFAULT_CONSTANTS = HydroConstants()


def linearized_hydrostatic_balance(
    zw, t, qv, ps, p,
    dt, dqv, dps,
    constants=None,
):
    """Tangent-linear hydrostatic balance. `ncells` (the middle dimension
    of t/qv/p/...) may be the full mesh or a rank's owned-cell slice; the
    computation is per-column and doesn't care which."""
    if constants is None:
        constants = _DEFAULT_CONSTANTS
    c = constants
    zw = np.asarray(zw, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    qv = np.asarray(qv, dtype=np.float64)
    ps = np.asarray(ps, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    dt = np.asarray(dt, dtype=np.float64)
    dqv = np.asarray(dqv, dtype=np.float64)
    dps = np.asarray(dps, dtype=np.float64)
    if t.ndim == 3:
        nTimes, ncells, nlevels = t.shape
    else:
        nTimes, ncells, nlevels = 1, t.shape[0], t.shape[1]
        t = t[None, ...]
        qv = qv[None, ...]
        p = p[None, ...]
        dt = dt[None, ...]
        dqv = dqv[None, ...]
        ps = ps[None, ...] if ps.ndim == 2 else ps[None, :]
        dps = dps[None, ...] if dps.ndim == 2 else dps[None, :]
    dp = np.empty((nTimes, ncells, nlevels), dtype=np.float64)
    drho = np.empty((nTimes, ncells, nlevels), dtype=np.float64)
    dtheta = np.empty((nTimes, ncells, nlevels), dtype=np.float64)
    pf = np.empty((ncells, nlevels + 1), dtype=np.float64)
    dpf = np.empty((ncells, nlevels + 1), dtype=np.float64)
    if zw.ndim == 2:
        zu = 0.5 * (zw[:, :-1] + zw[:, 1:])
    else:
        zu = 0.5 * (zw[:, :, :-1] + zw[:, :, 1:])
        zu = zu[0]
    for it in range(nTimes):
        t_it = t[it] if nTimes > 1 else t[0]
        qv_it = qv[it] if nTimes > 1 else qv[0]
        p_it = p[it] if nTimes > 1 else p[0]
        dt_it = dt[it] if nTimes > 1 else dt[0]
        dqv_it = dqv[it] if nTimes > 1 else dqv[0]
        ps_it = ps[it] if nTimes > 1 else ps[0]
        dps_it = dps[it] if nTimes > 1 else dps[0]
        tv_h = t_it * (1.0 + c.rvordm1 * qv_it)
        dtv_h = dt_it * (1.0 + c.rvordm1 * qv_it) + t_it * c.rvordm1 * dqv_it
        k = 0
        pf[:, k] = ps_it
        dpf[:, k] = dps_it
        dz0 = zu[:, k] - zw[:, k] if zw.ndim == 2 else zu[:, k] - zw[it, :, k]
        expo0 = np.exp(-c.gravity * dz0 / (c.rgas * tv_h[:, k]))
        dp[it, :, k] = (
            dpf[:, k] * expo0
            + pf[:, k] * expo0
              * c.gravity * dz0 / (c.rgas * tv_h[:, k] ** 2) * dtv_h[:, k]
        )
        drho[it, :, k] = (
            dp[it, :, k] / (c.rgas * tv_h[:, k] * (1.0 + qv_it[:, k]))
            - p_it[:, k] * dtv_h[:, k]
              / (c.rgas * tv_h[:, k] ** 2 * (1.0 + qv_it[:, k]))
            - p_it[:, k] * dqv_it[:, k]
              / (c.rgas * tv_h[:, k] * (1.0 + qv_it[:, k]) ** 2)
        )
        dtheta[it, :, k] = (
            dt_it[:, k] * (c.p0 / p_it[:, k]) ** c.kappa
            - t_it[:, k] * c.kappa
              * (c.p0 / p_it[:, k]) ** (c.kappa - 1.0)
              * (c.p0 / p_it[:, k] ** 2) * dp[it, :, k]
        )
        for k in range(1, nlevels):
            w = (zu[:, k] - (zw[:, k] if zw.ndim == 2 else zw[it, :, k])) / (zu[:, k] - zu[:, k - 1])
            tv_f = w * tv_h[:, k - 1] + (1.0 - w) * tv_h[:, k]
            dtv_f = w * dtv_h[:, k - 1] + (1.0 - w) * dtv_h[:, k]
            tv1 = 0.5 * (tv_h[:, k - 1] + tv_f)
            dtv1 = 0.5 * (dtv_h[:, k - 1] + dtv_f)
            dz1 = (zw[:, k] if zw.ndim == 2 else zw[it, :, k]) - zu[:, k - 1]
            e1 = np.exp(-c.gravity * dz1 / (c.rgas * tv1))
            pf[:, k] = p_it[:, k - 1] * e1
            dpf[:, k] = (
                dp[it, :, k - 1] * e1
                + p_it[:, k - 1] * e1
                  * c.gravity * dz1 / (c.rgas * tv1 ** 2) * dtv1
            )
            tv2 = 0.5 * (tv_h[:, k] + tv_f)
            dtv2 = 0.5 * (dtv_h[:, k] + dtv_f)
            dz2 = zu[:, k] - (zw[:, k] if zw.ndim == 2 else zw[it, :, k])
            e2 = np.exp(-c.gravity * dz2 / (c.rgas * tv2))
            dp[it, :, k] = (
                dpf[:, k] * e2
                + pf[:, k] * e2
                  * c.gravity * dz2 / (c.rgas * tv2 ** 2) * dtv2
            )
            drho[it, :, k] = (
                dp[it, :, k] / (c.rgas * tv_h[:, k] * (1.0 + qv_it[:, k]))
                - p_it[:, k] * dtv_h[:, k]
                  / (c.rgas * tv_h[:, k] ** 2 * (1.0 + qv_it[:, k]))
                - p_it[:, k] * dqv_it[:, k]
                  / (c.rgas * tv_h[:, k] * (1.0 + qv_it[:, k]) ** 2)
            )
            dtheta[it, :, k] = (
                dt_it[:, k] * (c.p0 / p_it[:, k]) ** c.kappa
                - t_it[:, k] * c.kappa
                  * (c.p0 / p_it[:, k]) ** (c.kappa - 1.0)
                  * (c.p0 / p_it[:, k] ** 2) * dp[it, :, k]
            )
    if nTimes == 1:
        return dp[0], drho[0], dtheta[0]
    return dp, drho, dtheta
