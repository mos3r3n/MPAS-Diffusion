"""
Shared NetCDF writer utilities.

write_new_state_file() -- create a brand-new, self-contained NetCDF file
                           (dimensions + a small set of coordinate
                           variables copied from a template + the given
                           state variables), using a strict "define
                           everything, then write everything" phase
                           separation.

                           This split matters far more than it looks like
                           it should: netCDF's classic model reserves
                           header space once, at the first transition from
                           "define mode" into "data mode". Any
                           createVariable()/setncattr() call AFTER data
                           has already been written forces the library
                           back into define mode (nc_redef()), and if the
                           now-bigger header no longer fits the space
                           reserved for it, the ENTIRE data section
                           already on disk has to be shifted down to make
                           room -- an O(file size) copy. Interleaving
                           "create a variable, write its data" in a loop
                           pays that cost roughly once per variable;
                           setting global attributes only after every
                           variable's data is written pays it once more,
                           on the FULL file, at the very end. Defining
                           every variable (and every attribute) before the
                           first data write means there is exactly one
                           define->data transition, sized correctly from
                           the start, no matter how many variables exist.

                           Used by the filter app (one file per diffusion
                           band) and, optionally, by the blending app (the
                           filtered-large/filtered-small diagnostic dump).

copy_and_overwrite()   -- copy a template file, then overwrite EXISTING
                           variables' data in place. No new variables are
                           created, so there's no redef concern here to
                           begin with. Used by the blending app to produce
                           the final blended_file from the small-scale
                           background file.
"""
from __future__ import annotations

import shutil
import numpy as np
import netCDF4 as nc
from typing import Dict, Iterable, Optional

from .mpi_env import logger, rlog


FORMAT_MAP = {
    "NETCDF4":               "NETCDF4_CLASSIC",
    "NETCDF4_CLASSIC":       "NETCDF4_CLASSIC",
    "NETCDF3_CLASSIC":       "NETCDF3_64BIT_OFFSET",
    "NETCDF3_64BIT_OFFSET":  "NETCDF3_64BIT_OFFSET",
    "NETCDF3_64BIT_DATA":    "NETCDF3_64BIT_DATA",
}


def out_format_for(input_file: str) -> str:
    """
    SMIOL/MPAS can only read NETCDF3_* or NETCDF4_CLASSIC. Map any
    NetCDF-4 source to NETCDF4_CLASSIC so output stays HDF5-free unless
    the source is already NetCDF-3 flavour.

    This also determines whether compression/chunking are even possible:
    NETCDF3_* formats support neither (classic model, no HDF5
    underneath).
    """
    with nc.Dataset(input_file) as src:
        src_format = src.file_format
    return FORMAT_MAP.get(src_format, "NETCDF3_64BIT_OFFSET")


def chunk_sizes(dims, shape):
    """
    Explicit HDF5 chunk shape for a NETCDF4* variable: don't chunk along
    Time or vertical levels (chunk = full extent), chunk along nCells in
    blocks of up to 32768. Only meaningful for NETCDF4* output -- NETCDF3
    has no concept of chunking, so callers must not pass chunksizes at
    all for that format (createVariable raises if you try).
    """
    chunks = []
    for dname, dlen in zip(dims, shape):
        if dname == "Time":
            chunks.append(1)
        elif dname == "nCells":
            chunks.append(max(1, min(32768, dlen)))
        else:
            chunks.append(max(1, dlen))
    return tuple(chunks)


def write_new_state_file(
    input_file: str,
    output_file: str,
    state: Dict[str, np.ndarray],
    coord_vars: Iterable[str] = ("latCell", "lonCell"),
    zlib: bool = True,
    complevel: int = 4,
    global_attrs: Optional[Dict[str, str]] = None,
) -> None:
    """
    Create a NEW NetCDF file with the same dimensions as `input_file`,
    containing `coord_vars` (copied byte-for-byte from the source, with
    their attributes) plus every entry in `state`. See module docstring
    for why define and write are strictly separated.

    zlib/complevel only take effect for NETCDF4* output (NETCDF3 supports
    neither) -- see out_format_for()'s docstring for when that applies.
    """
    logger.info("Writing: %s", output_file)
    out_format = out_format_for(input_file)
    use_zlib   = zlib and out_format.startswith("NETCDF4")
    use_complevel = complevel if use_zlib else 0
    use_chunks = out_format.startswith("NETCDF4")
    logger.info("Output format=%s  (zlib=%s, complevel=%d)", out_format, use_zlib, use_complevel)

    with nc.Dataset(input_file) as src, nc.Dataset(output_file, "w", format=out_format) as dst:
        for name, dim in src.dimensions.items():
            dst.createDimension(name, None if dim.isunlimited() else len(dim))

        # ── define phase: every variable + every attribute, no data yet ──
        coord_srcs: Dict[str, "nc.Variable"] = {}
        for vname in coord_vars:
            if vname in src.variables:
                vsrc = src.variables[vname]
                vdst = dst.createVariable(vname, vsrc.dtype, vsrc.dimensions)
                vdst.setncatts({k: vsrc.getncattr(k) for k in vsrc.ncattrs()})
                coord_srcs[vname] = vsrc

        created: Dict[str, "nc.Variable"] = {}
        for name, data in state.items():
            if name in src.variables:
                dims, dtype = src.variables[name].dimensions, src.variables[name].dtype
                if data.ndim == 3 and data.shape[0] == 1 and len(dims) == 2:
                    data = data[0]
                    state[name] = data
            elif data.ndim == 3:
                dims, dtype = ("Time", "nCells", "nVertLevels"), "f4"
            elif data.ndim == 2:
                dims, dtype = ("Time", "nCells"), "f4"
            else:
                dims, dtype = ("nCells",), "f4"

            create_kwargs = dict(zlib=use_zlib, complevel=use_complevel)
            if use_chunks:
                create_kwargs["chunksizes"] = chunk_sizes(dims, data.shape)
            v = dst.createVariable(name, dtype, dims, **create_kwargs)
            if name in src.variables:
                for attr in src.variables[name].ncattrs():
                    v.setncattr(attr, src.variables[name].getncattr(attr))
            created[name] = v

        for k, val in (global_attrs or {}).items():
            setattr(dst, k, val)

        # ── data phase: exactly one define->data transition happens here ──
        for vname, vsrc in coord_srcs.items():
            dst.variables[vname][:] = vsrc[:]
        for name, data in state.items():
            if data.dtype != np.float32:
                data = data.astype(np.float32, copy=False)
            created[name][:] = data
    logger.info("Done: %s", output_file)


def copy_and_overwrite(template_file: str, output_file: str, state: Dict[str, np.ndarray]) -> None:
    """Copy template_file to output_file, then overwrite EXISTING
    variables' data in place from `state` (variables not already present
    in the template are silently skipped). No new variables are created,
    so there's no redef concern here."""
    shutil.copy(template_file, output_file)
    rlog(f"Copied template → {output_file}")
    with nc.Dataset(output_file, "r+") as ds:
        for name, data in state.items():
            if name not in ds.variables:
                continue
            rlog(f"  writing {name}")
            ds.variables[name][:] = data
    rlog("Finished writing state.")
