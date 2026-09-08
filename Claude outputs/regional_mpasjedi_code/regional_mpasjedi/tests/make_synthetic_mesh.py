"""
Build a small synthetic MPAS-like mesh + data files for testing, without
needing real cluster data. Not part of the deliverable -- test/dev only.

Mesh: an n x n periodic (torus) grid, so every cell has exactly 4
neighbours and no boundary fill-value padding is needed in cellsOnCell.
"""
import numpy as np
import netCDF4 as nc


def build_mesh_arrays(n=6):
    nCells = n * n
    nEdges = 2 * n * n
    nVertLevels = 4
    nVertLevelsP1 = nVertLevels + 1

    def cidx(i, j):
        return (i % n) * n + (j % n)

    def h_edge(i, j):
        return (i % n) * n + (j % n)

    def v_edge(i, j):
        return n * n + (i % n) * n + (j % n)

    nEdgesOnCell = np.full(nCells, 4, dtype=np.int32)
    cellsOnCell = np.zeros((nCells, 4), dtype=np.int32)
    edgesOnCell = np.zeros((nCells, 4), dtype=np.int32)

    for i in range(n):
        for j in range(n):
            c = cidx(i, j)
            right = cidx(i, j + 1)
            left  = cidx(i, j - 1)
            down  = cidx(i + 1, j)
            up    = cidx(i - 1, j)
            cellsOnCell[c] = [right + 1, left + 1, down + 1, up + 1]
            edgesOnCell[c] = [h_edge(i, j) + 1, h_edge(i, j - 1) + 1,
                               v_edge(i, j) + 1, v_edge(i - 1, j) + 1]

    dlat = 0.02
    dlon = 0.02
    latCell = np.zeros(nCells)
    lonCell = np.zeros(nCells)
    for i in range(n):
        for j in range(n):
            latCell[cidx(i, j)] = (i - n / 2) * dlat
            lonCell[cidx(i, j)] = (j - n / 2) * dlon + np.pi

    areaCell = np.full(nCells, 1.0e6, dtype=np.float64)
    dcEdge = np.full(nEdges, 1000.0, dtype=np.float64)
    dvEdge = np.full(nEdges, 1000.0, dtype=np.float64)

    rng = np.random.default_rng(42)
    edgeNormalVectors = rng.normal(size=(nEdges, 3))
    edgeNormalVectors /= np.linalg.norm(edgeNormalVectors, axis=-1, keepdims=True)

    base_heights = np.linspace(0.0, 8000.0, nVertLevelsP1)
    zgrid = np.tile(base_heights, (nCells, 1)) + (latCell[:, None] * 500.0)

    return dict(
        n=n, nCells=nCells, nEdges=nEdges, nVertLevels=nVertLevels,
        nVertLevelsP1=nVertLevelsP1,
        nEdgesOnCell=nEdgesOnCell, cellsOnCell=cellsOnCell, edgesOnCell=edgesOnCell,
        latCell=latCell, lonCell=lonCell, areaCell=areaCell,
        dcEdge=dcEdge, dvEdge=dvEdge, edgeNormalVectors=edgeNormalVectors,
        zgrid=zgrid,
    )


def write_mesh_file(path, arrs):
    with nc.Dataset(path, "w", format="NETCDF3_64BIT_OFFSET") as ds:
        ds.createDimension("nCells", arrs["nCells"])
        ds.createDimension("nEdges", arrs["nEdges"])
        ds.createDimension("maxEdges", 4)
        ds.createDimension("nVertLevelsP1", arrs["nVertLevelsP1"])
        ds.createDimension("R3", 3)

        v = ds.createVariable("nEdgesOnCell", "i4", ("nCells",)); v[:] = arrs["nEdgesOnCell"]
        v = ds.createVariable("cellsOnCell", "i4", ("nCells", "maxEdges")); v[:] = arrs["cellsOnCell"]
        v = ds.createVariable("edgesOnCell", "i4", ("nCells", "maxEdges")); v[:] = arrs["edgesOnCell"]
        v = ds.createVariable("latCell", "f8", ("nCells",)); v[:] = arrs["latCell"]
        v = ds.createVariable("lonCell", "f8", ("nCells",)); v[:] = arrs["lonCell"]
        v = ds.createVariable("areaCell", "f8", ("nCells",)); v[:] = arrs["areaCell"]
        v = ds.createVariable("dcEdge", "f8", ("nEdges",)); v[:] = arrs["dcEdge"]
        v = ds.createVariable("dvEdge", "f8", ("nEdges",)); v[:] = arrs["dvEdge"]
        v = ds.createVariable("edgeNormalVectors", "f8", ("nEdges", "R3")); v[:] = arrs["edgeNormalVectors"]
        v = ds.createVariable("zgrid", "f8", ("nCells", "nVertLevelsP1")); v[:] = arrs["zgrid"]


def write_data_file(path, arrs, seed=0, with_surface_pressure=True):
    nCells = arrs["nCells"]
    nEdges = arrs["nEdges"]
    nVertLevels = arrs["nVertLevels"]
    rng = np.random.default_rng(seed)

    theta = 300.0 + 5.0 * rng.normal(size=(1, nCells, nVertLevels))
    qv = np.clip(0.01 + 0.002 * rng.normal(size=(1, nCells, nVertLevels)), 1e-5, None)
    pressure_base = np.tile(np.linspace(95000.0, 30000.0, nVertLevels), (1, nCells, 1))
    pressure_p = 50.0 * rng.normal(size=(1, nCells, nVertLevels))
    uZonal = 10.0 * rng.normal(size=(1, nCells, nVertLevels))
    uMerid = 5.0 * rng.normal(size=(1, nCells, nVertLevels))
    surface_pressure = 101000.0 + 100.0 * rng.normal(size=(1, nCells))
    u_edge = 8.0 * rng.normal(size=(1, nEdges, nVertLevels))
    rho = 1.0 + 0.05 * rng.normal(size=(1, nCells, nVertLevels))

    with nc.Dataset(path, "w", format="NETCDF3_64BIT_OFFSET") as ds:
        ds.createDimension("Time", None)
        ds.createDimension("nCells", nCells)
        ds.createDimension("nEdges", nEdges)
        ds.createDimension("nVertLevels", nVertLevels)

        fields = [
            ("theta", theta, ("Time", "nCells", "nVertLevels")),
            ("qv", qv, ("Time", "nCells", "nVertLevels")),
            ("pressure_base", pressure_base, ("Time", "nCells", "nVertLevels")),
            ("pressure_p", pressure_p, ("Time", "nCells", "nVertLevels")),
            ("uReconstructZonal", uZonal, ("Time", "nCells", "nVertLevels")),
            ("uReconstructMeridional", uMerid, ("Time", "nCells", "nVertLevels")),
            ("u", u_edge, ("Time", "nEdges", "nVertLevels")),
            ("rho", rho, ("Time", "nCells", "nVertLevels")),
        ]
        if with_surface_pressure:
            fields.append(("surface_pressure", surface_pressure, ("Time", "nCells")))
        for name, arr, dims in fields:
            v = ds.createVariable(name, "f8", dims)
            v[:] = arr


if __name__ == "__main__":
    import sys, os
    outdir = sys.argv[1] if len(sys.argv) > 1 else "/tmp/synthetic_mpas"
    os.makedirs(outdir, exist_ok=True)
    arrs = build_mesh_arrays(n=6)
    write_mesh_file(os.path.join(outdir, "mesh.nc"), arrs)
    write_data_file(os.path.join(outdir, "large_scale.nc"), arrs, seed=1)
    write_data_file(os.path.join(outdir, "small_scale.nc"), arrs, seed=2)
    write_data_file(os.path.join(outdir, "filter_data.nc"), arrs, seed=3, with_surface_pressure=False)
    print("wrote synthetic mesh+data to", outdir)
