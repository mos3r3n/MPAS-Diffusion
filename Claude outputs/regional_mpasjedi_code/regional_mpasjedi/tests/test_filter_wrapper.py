"""End-to-end smoke test for the rewritten Diffusion/mpas_diffsion_filters.py
wrapper (now built on mpas_core), including surface_pressure derivation
from a data file that lacks it, run under SIZE=1 in this sandbox."""
import os
import sys
import shutil
import tempfile
import numpy as np
import netCDF4 as nc

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # tests/ (for make_synthetic_mesh)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # regional_mpasjedi/ (for mpas_core)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Diffusion"))  # for the wrapper module
from make_synthetic_mesh import build_mesh_arrays, write_mesh_file, write_data_file

import mpas_diffsion_filters as filt_app

tmpdir = tempfile.mkdtemp(prefix="filter_test_")
try:
    arrs = build_mesh_arrays(n=6)
    mesh_file = os.path.join(tmpdir, "mesh.nc")
    data_file = os.path.join(tmpdir, "filter_data.nc")
    write_mesh_file(mesh_file, arrs)
    # with_surface_pressure=False -- exercise the hydrostatic derivation path
    write_data_file(data_file, arrs, seed=3, with_surface_pressure=False)

    cfg = filt_app.FilterConfig(
        mesh_file=mesh_file, data_file=data_file,
        scales_km=[15.0, 40.0], n_iter=5,
        output_prefix=os.path.join(tmpdir, "out", "state_lp"),
        use_cache=False, pcg_tol=1e-9, pcg_maxiter=300,
    )
    filt_app.run(cfg)

    expected_bands = ["below15km", "15-40km", "above40km"]
    for bname in expected_bands:
        path = f"{cfg.output_prefix}_{bname}.nc"
        assert os.path.exists(path), f"missing band file {path}"
        with nc.Dataset(path) as ds:
            assert "surface_pressure" in ds.variables, "surface_pressure was not derived/written"
            assert "temperature" in ds.variables
            assert "spechum" in ds.variables
            assert ds.variables["temperature"].shape[1] == arrs["nCells"]
        print(f"OK: {bname} written with expected variables")

    # Band sum should reconstruct the original field (low-pass telescoping).
    with nc.Dataset(f"{cfg.output_prefix}_below15km.nc") as d0, \
         nc.Dataset(f"{cfg.output_prefix}_15-40km.nc") as d1, \
         nc.Dataset(f"{cfg.output_prefix}_above40km.nc") as d2:
        total = (np.asarray(d0.variables["temperature"][:])
                 + np.asarray(d1.variables["temperature"][:])
                 + np.asarray(d2.variables["temperature"][:]))

    with nc.Dataset(data_file) as ds, nc.Dataset(mesh_file) as mds:
        theta = ds.variables["theta"][:]
        p = ds.variables["pressure_base"][:] + ds.variables["pressure_p"][:]
        temperature_orig = theta * (p / 100000.0) ** (2.0 / 7.0)  # RD_CP, matches mpas_core.constants

    diff = np.max(np.abs(total - temperature_orig))
    rel = diff / np.max(np.abs(temperature_orig))
    print(f"band-sum reconstruction: max_abs_diff={diff:.3e}  max_rel_diff={rel:.3e}")
    assert rel < 1e-4, f"bands don't sum back to the original field: rel={rel}"

    print("ALL FILTER-WRAPPER CHECKS PASSED")
finally:
    shutil.rmtree(tmpdir, ignore_errors=True)
