"""
Verification harness for the standalone add_surface_pressure.py script.
Reuses make_synthetic_mesh's generators: the mesh file (has zgrid) stands
in for the "invariant" file, and write_data_file(with_surface_pressure=False)
stands in for an mpasout file missing surface_pressure.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # tests/ (for make_synthetic_mesh)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # regional_mpasjedi/ (for add_surface_pressure.py)
import shutil
import subprocess
import tempfile
import numpy as np
import netCDF4 as nc

from make_synthetic_mesh import build_mesh_arrays, write_mesh_file, write_data_file

SCRIPT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "add_surface_pressure.py")

tmpdir = tempfile.mkdtemp(prefix="add_sfcp_")
try:
    arrs = build_mesh_arrays(n=6)
    invariant_file = os.path.join(tmpdir, "invariant.nc")
    write_mesh_file(invariant_file, arrs)  # has zgrid

    # --- Case A: mpasout with pressure_base + pressure_p, no surface_pressure ---
    mpasout_a = os.path.join(tmpdir, "mpasout_a.nc")
    write_data_file(mpasout_a, arrs, seed=3, with_surface_pressure=False)

    r = subprocess.run([sys.executable, SCRIPT, invariant_file, mpasout_a],
                        capture_output=True, text=True)
    print("=== Case A stdout ===\n", r.stdout)
    print("=== Case A stderr ===\n", r.stderr)
    assert r.returncode == 0, "script failed on case A"

    with nc.Dataset(mpasout_a) as ds:
        assert "surface_pressure" in ds.variables
        v = ds.variables["surface_pressure"]
        assert v.dimensions == ("Time", "nCells")
        assert v.dtype == np.float32
        assert v.units == "Pa"
        assert v.long_name == "Surface pressure"
        sp = np.asarray(v[:])
        print("surface_pressure range:", sp.min(), sp.max())
        # sanity: should be close to but a bit above the lowest-level pressure
        p_lowest = np.asarray(ds.variables["pressure_base"][:, :, 0]) + \
                   np.asarray(ds.variables["pressure_p"][:, :, 0])
        assert np.all(sp > p_lowest.astype(np.float32) - 1.0), \
            "surface_pressure should be >= lowest-level pressure (extrapolating downward)"
        # synthetic mesh has a coarse first layer (~1000 m from terrain to
        # the lowest mass-level midpoint), so a ~10-15 kPa hydrostatic
        # extrapolation is expected here (real MPAS meshes have much
        # thinner near-surface layers and a correspondingly smaller gap).
        assert np.all(sp < p_lowest.astype(np.float32) + 15000.0), \
            "surface_pressure implausibly far from lowest-level pressure"

    # --- Case B: re-run without --force -> should skip, not change values ---
    with nc.Dataset(mpasout_a) as ds:
        sp_before = np.asarray(ds.variables["surface_pressure"][:]).copy()

    r2 = subprocess.run([sys.executable, SCRIPT, invariant_file, mpasout_a],
                         capture_output=True, text=True)
    print("=== Case B (skip) stdout ===\n", r2.stdout)
    assert r2.returncode == 0
    assert "[skip]" in r2.stdout
    with nc.Dataset(mpasout_a) as ds:
        sp_after = np.asarray(ds.variables["surface_pressure"][:])
    assert np.array_equal(sp_before, sp_after), "skip path should not modify data"

    # --- Case C: --force alone -> compare only, must NOT modify the file ---
    r3 = subprocess.run([sys.executable, SCRIPT, invariant_file, mpasout_a, "--force"],
                         capture_output=True, text=True)
    print("=== Case C (--force, compare-only) stdout ===\n", r3.stdout)
    assert r3.returncode == 0
    assert "[compare]" in r3.stdout
    assert "NOT modified" in r3.stdout
    with nc.Dataset(mpasout_a) as ds:
        sp_after_compare = np.asarray(ds.variables["surface_pressure"][:])
    assert np.array_equal(sp_before, sp_after_compare), \
        "--force without --overwrite must not modify the file"

    # --- Case C2: --force --overwrite -> actually recomputes and writes ---
    r3b = subprocess.run([sys.executable, SCRIPT, invariant_file, mpasout_a,
                          "--force", "--overwrite"],
                         capture_output=True, text=True)
    print("=== Case C2 (--force --overwrite) stdout ===\n", r3b.stdout)
    assert r3b.returncode == 0
    assert "[done]" in r3b.stdout

    # --- Case D: mpasout with direct 'pressure' variable instead of base+p ---
    mpasout_d = os.path.join(tmpdir, "mpasout_d.nc")
    write_data_file(mpasout_d, arrs, seed=4, with_surface_pressure=False)
    with nc.Dataset(mpasout_d, "r+") as ds:
        pb = np.asarray(ds.variables["pressure_base"][:])
        pp = np.asarray(ds.variables["pressure_p"][:])
        v = ds.createVariable("pressure", "f8", ("Time", "nCells", "nVertLevels"))
        v[:] = pb + pp
        # netCDF4 can't delete a variable in place; pressure_base/pressure_p
        # stay in the file, but read_pressure() prefers 'pressure' when
        # present so this still exercises the direct-'pressure' branch.
    r4 = subprocess.run([sys.executable, SCRIPT, invariant_file, mpasout_d],
                         capture_output=True, text=True)
    print("=== Case D ('pressure' path) stdout ===\n", r4.stdout)
    print("=== Case D stderr ===\n", r4.stderr)
    assert r4.returncode == 0
    with nc.Dataset(mpasout_d) as ds:
        assert "surface_pressure" in ds.variables

    # Confirm case D (direct 'pressure') gives same result as case A
    # (base+p path) since pressure = pressure_base+pressure_p identically
    # and both files share the same seed=3 vs seed=4 -- instead, directly
    # verify D's surface_pressure matches a manual recompute using its own
    # 'pressure' variable, to check the read_pressure() 'pressure' branch.
        import importlib.util
    spec = importlib.util.spec_from_file_location("add_sfcp_mod", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    with nc.Dataset(mpasout_d) as ds, nc.Dataset(invariant_file) as inv:
        zgrid = np.asarray(inv.variables["zgrid"][:, :2], dtype=np.float64)
        theta1 = np.asarray(ds.variables["theta"][:, :, 0], dtype=np.float64)
        qv1 = np.asarray(ds.variables["qv"][:, :, 0], dtype=np.float64)
        p1 = np.asarray(ds.variables["pressure"][:, :, 0], dtype=np.float64)
        z_sfc = zgrid[:, 0]
        z1 = 0.5 * (zgrid[:, 0] + zgrid[:, 1])
        expected = mod.compute_surface_pressure(p1, z_sfc, z1, theta1, qv1)
        actual = np.asarray(ds.variables["surface_pressure"][:], dtype=np.float64)
        rel = np.max(np.abs(expected.astype(np.float32) - actual)) / np.max(np.abs(actual))
        print(f"case D manual-recompute rel diff: {rel:.3e}")
        assert rel < 1e-6

    # --- Case E: multiple mpasout files in one invocation, one deliberately broken ---
    mpasout_e1 = os.path.join(tmpdir, "mpasout_e1.nc")
    mpasout_e2_bad = os.path.join(tmpdir, "mpasout_e2_bad.nc")
    write_data_file(mpasout_e1, arrs, seed=5, with_surface_pressure=False)
    # e2_bad: missing 'theta' entirely -> should fail but not abort other files
    with nc.Dataset(mpasout_e2_bad, "w", format="NETCDF3_64BIT_OFFSET") as ds:
        ds.createDimension("Time", None)
        ds.createDimension("nCells", arrs["nCells"])
        ds.createDimension("nVertLevels", arrs["nVertLevels"])
        v = ds.createVariable("qv", "f8", ("Time", "nCells", "nVertLevels"))
        v[:] = 0.01
        v = ds.createVariable("pressure_base", "f8", ("Time", "nCells", "nVertLevels"))
        v[:] = 90000.0
        v = ds.createVariable("pressure_p", "f8", ("Time", "nCells", "nVertLevels"))
        v[:] = 0.0

    r5 = subprocess.run([sys.executable, SCRIPT, invariant_file, mpasout_e1, mpasout_e2_bad],
                         capture_output=True, text=True)
    print("=== Case E (multi-file, one bad) stdout ===\n", r5.stdout)
    print("=== Case E stderr ===\n", r5.stderr)
    assert r5.returncode == 1, "should exit 1 since one file failed"
    assert "[done]" in r5.stdout, "good file should still have succeeded"
    assert "[error]" in r5.stderr, "bad file should have reported an error"
    with nc.Dataset(mpasout_e1) as ds:
        assert "surface_pressure" in ds.variables, "good file in multi-file run should still be written"
    with nc.Dataset(mpasout_e2_bad) as ds:
        assert "surface_pressure" not in ds.variables, "bad file should not have been written"

    # --- Case F: --workers parallel path, multiple independent good files ---
    mpasout_f1 = os.path.join(tmpdir, "mpasout_f1.nc")
    mpasout_f2 = os.path.join(tmpdir, "mpasout_f2.nc")
    mpasout_f3 = os.path.join(tmpdir, "mpasout_f3.nc")
    write_data_file(mpasout_f1, arrs, seed=6, with_surface_pressure=False)
    write_data_file(mpasout_f2, arrs, seed=7, with_surface_pressure=False)
    write_data_file(mpasout_f3, arrs, seed=8, with_surface_pressure=False)

    r6 = subprocess.run([sys.executable, SCRIPT, invariant_file,
                          mpasout_f1, mpasout_f2, mpasout_f3, "--workers", "3"],
                         capture_output=True, text=True)
    print("=== Case F (--workers 3) stdout ===\n", r6.stdout)
    print("=== Case F stderr ===\n", r6.stderr)
    assert r6.returncode == 0, "parallel run should succeed"
    assert r6.stdout.count("[done]") == 3, "all three files should report done"
    for f in (mpasout_f1, mpasout_f2, mpasout_f3):
        with nc.Dataset(f) as ds:
            assert "surface_pressure" in ds.variables, f"{f} missing surface_pressure after parallel run"

    print("\nALL add_surface_pressure.py CHECKS PASSED")
finally:
    shutil.rmtree(tmpdir, ignore_errors=True)
