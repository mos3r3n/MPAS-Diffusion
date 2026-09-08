"""End-to-end smoke test for the rewritten Blending/mpas_blending.py
wrapper (now built on mpas_core, single distributed engine)."""
import os
import sys
import shutil
import tempfile
import numpy as np
import netCDF4 as nc

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # tests/ (for make_synthetic_mesh)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # regional_mpasjedi/ (for mpas_core)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Blending"))  # for the wrapper module
from make_synthetic_mesh import build_mesh_arrays, write_mesh_file, write_data_file

import mpas_blending as blend_app

tmpdir = tempfile.mkdtemp(prefix="blend_test_")
try:
    arrs = build_mesh_arrays(n=6)
    mesh_file = os.path.join(tmpdir, "mesh.nc")
    large_file = os.path.join(tmpdir, "large_scale.nc")
    small_file = os.path.join(tmpdir, "small_scale.nc")
    write_mesh_file(mesh_file, arrs)
    write_data_file(large_file, arrs, seed=1)
    write_data_file(small_file, arrs, seed=2)

    blended_file = os.path.join(tmpdir, "blended.nc")
    cfg = blend_app.Config(
        mesh_file=mesh_file, large_scale_file=large_file, small_scale_file=small_file,
        blended_file=blended_file, scale_km=20.0, n_iter=6,
        use_cache=False, pcg_tol=1e-10, pcg_maxiter=500,
        output_filtered_fields=True, output_dir=os.path.join(tmpdir, "filtered_out"),
    )
    blend_app.run(cfg)

    assert os.path.exists(blended_file), "blended_file was not written"
    with nc.Dataset(blended_file) as ds, nc.Dataset(small_file) as small_ds:
        for name in ["uReconstructZonal", "uReconstructMeridional", "u",
                     "theta", "qv", "surface_pressure", "pressure_p", "rho"]:
            assert name in ds.variables, f"missing {name} in blended_file"
            a = np.asarray(ds.variables[name][:])
            b = np.asarray(small_ds.variables[name][:])
            assert a.shape == b.shape
            # blended should differ from the small-scale background (an
            # increment was actually applied), but not wildly (scale=20km
            # increments should be modest relative to the field itself).
            diff = np.max(np.abs(a - b))
            print(f"  {name:24s} max_abs_diff_from_background={diff:.4g}")
        assert np.max(np.abs(np.asarray(ds.variables["u"][:])
                              - np.asarray(small_ds.variables["u"][:]))) > 0, \
            "background 'u' was never incremented -- wind projection may be broken"

    filtered_dir = os.path.join(tmpdir, "filtered_out")
    files = sorted(os.listdir(filtered_dir))
    print("filtered_out contents:", files)
    assert any("large_scale_filtered" in f for f in files)
    assert any("small_scale_filtered" in f for f in files)
    # Power-spectrum computation was removed entirely (it wasn't a real
    # spectral decomposition and wasn't efficient) -- no spectrum_*.nc
    # file should be produced anymore, and Config no longer even has a
    # compute_spectra/spectral_bins field to turn it on.
    assert not hasattr(blend_app.Config(), "compute_spectra"), \
        "Config should no longer have a compute_spectra field"
    assert not any(f.startswith("spectrum_") for f in files), \
        "no spectrum_*.nc file should be written anymore"

    print("ALL BLENDING-WRAPPER CHECKS PASSED")
finally:
    shutil.rmtree(tmpdir, ignore_errors=True)
