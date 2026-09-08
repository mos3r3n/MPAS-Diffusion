"""
Verify FilterConfig.filter_cloud_vars actually controls whether cloud
variables get diffused (True, new default) or exempted/zeroed-in-lowpass
(False, old behavior) -- the bug report was a `!`-vs-`not` typo plus a
config field (filter_cloud_vars) that didn't exist yet; this confirms the
newly wired-in toggle behaves as intended in both positions.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # tests/ (for make_synthetic_mesh)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # regional_mpasjedi/ (for mpas_core)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Diffusion"))  # for the wrapper module
import shutil
import tempfile
import numpy as np
import netCDF4 as nc

from make_synthetic_mesh import build_mesh_arrays, write_mesh_file, write_data_file

import mpas_diffsion_filters as filt_app

tmpdir = tempfile.mkdtemp(prefix="cloudvar_toggle_")
try:
    arrs = build_mesh_arrays(n=6)
    mesh_file = os.path.join(tmpdir, "mesh.nc")
    data_file = os.path.join(tmpdir, "filter_data.nc")
    write_mesh_file(mesh_file, arrs)
    write_data_file(data_file, arrs, seed=3, with_surface_pressure=True)

    # Add a cloud variable ("qc") to the data file so the toggle has
    # something to act on.
    with nc.Dataset(data_file, "r+") as ds:
        rng = np.random.default_rng(11)
        qc = np.clip(0.001 * rng.normal(size=ds.variables["theta"].shape) + 0.001, 0, None)
        v = ds.createVariable("qc", "f8", ds.variables["theta"].dimensions)
        v[:] = qc

    def run_case(filter_cloud_vars, out_prefix):
        cfg = filt_app.FilterConfig(
            mesh_file=mesh_file, data_file=data_file,
            scales_km=[15.0, 40.0], n_iter=5,
            output_prefix=os.path.join(tmpdir, out_prefix),
            use_cache=False, pcg_tol=1e-9, pcg_maxiter=300,
            filter_cloud_vars=filter_cloud_vars,
        )
        filt_app.run(cfg)
        with nc.Dataset(f"{cfg.output_prefix}_below15km.nc") as d0:
            assert "qc" in d0.variables, "qc should always be present in the finest band"
            qc0 = np.asarray(d0.variables["qc"][:])
        has_qc_15_40 = None
        has_qc_above = None
        with nc.Dataset(f"{cfg.output_prefix}_15-40km.nc") as d1:
            has_qc_15_40 = "qc" in d1.variables
            qc1 = np.asarray(d1.variables["qc"][:]) if has_qc_15_40 else None
        with nc.Dataset(f"{cfg.output_prefix}_above40km.nc") as d2:
            has_qc_above = "qc" in d2.variables
            qc2 = np.asarray(d2.variables["qc"][:]) if has_qc_above else None
        return qc0, qc1, qc2, has_qc_15_40, has_qc_above

    # --- filter_cloud_vars=False: qc must be entirely in the smallest
    # band (below15km); the other two band FILES should not even contain
    # a qc variable (it's known to be exactly zero there, so it's omitted
    # rather than written out).
    qc0_f, qc1_f, qc2_f, has_15_40_f, has_above_f = run_case(False, "out_false/state_lp")
    assert not has_15_40_f, "15-40km band file should NOT have a qc variable when filter_cloud_vars=False"
    assert not has_above_f, "above40km band file should NOT have a qc variable when filter_cloud_vars=False"
    with nc.Dataset(data_file) as ds:
        qc_orig = np.asarray(ds.variables["qc"][:])
    assert np.allclose(qc0_f, qc_orig), "with filter_cloud_vars=False, below15km band should equal the raw qc field"
    print("filter_cloud_vars=False: qc fully in smallest band; omitted (not just zero) from the other band files -- OK")

    # --- filter_cloud_vars=True: qc should actually be spread/smoothed
    # across bands like any other field, and be PRESENT (not omitted) in
    # every band file, since it's no longer known to be all-zero.
    qc0_t, qc1_t, qc2_t, has_15_40_t, has_above_t = run_case(True, "out_true/state_lp")
    assert has_15_40_t, "15-40km band file SHOULD have qc when filter_cloud_vars=True"
    assert has_above_t, "above40km band file SHOULD have qc when filter_cloud_vars=True"
    assert not np.all(qc1_t == 0.0), "15-40km band should NOT be all-zero for qc when filter_cloud_vars=True"
    assert not np.all(qc2_t == 0.0), "above40km band should NOT be all-zero for qc when filter_cloud_vars=True"
    assert not np.allclose(qc0_t, qc_orig), "with filter_cloud_vars=True, below15km band should differ from the raw field (it's now a genuine low-pass residual)"
    # band sum should still reconstruct the original field either way
    total_t = qc0_t + qc1_t + qc2_t
    rel_t = np.max(np.abs(total_t - qc_orig)) / np.max(np.abs(qc_orig))
    print(f"filter_cloud_vars=True: qc spread across bands, band-sum reconstruction rel_diff={rel_t:.3e}")
    assert rel_t < 1e-4

    # default (no filter_cloud_vars passed) should match True (the agreed default)
    cfg_default = filt_app.FilterConfig(
        mesh_file=mesh_file, data_file=data_file, scales_km=[15.0, 40.0], n_iter=5,
        output_prefix=os.path.join(tmpdir, "out_default/state_lp"), use_cache=False,
    )
    assert cfg_default.filter_cloud_vars is True, "default for filter_cloud_vars should be True"

    print("\nALL filter_cloud_vars TOGGLE CHECKS PASSED")
finally:
    shutil.rmtree(tmpdir, ignore_errors=True)
