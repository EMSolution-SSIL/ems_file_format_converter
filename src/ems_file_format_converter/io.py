# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Science Solutions International Laboratory, Inc.
from pathlib import Path

import meshio

from . import atlas
from . import femap
from . import gmsh
from . import unv


_READERS = {
    "atl": atlas.read_mesh,
    "atlas": atlas.read_mesh,
    "unv": unv.read_mesh,
    "neu": femap.read_mesh,
    "femap": femap.read_mesh,
    "msh": gmsh.read_mesh,
    "gmsh": gmsh.read_mesh,
    "gmsh4": gmsh.read_mesh,
    "gmsh41": gmsh.read_mesh,
}

_WRITERS = {
    "atl": atlas.write_mesh,
    "atlas": atlas.write_mesh,
    "unv": unv.write_mesh,
    "neu": femap.write_mesh,
    "femap": femap.write_mesh,
    "msh": gmsh.write_mesh,
    "gmsh": gmsh.write_mesh,
    "gmsh4": gmsh.write_mesh,
    "gmsh41": gmsh.write_mesh,
}


def _normalize_format(mesh_format: str | None) -> str | None:
    if mesh_format is None:
        return None
    normalized = mesh_format.lower().lstrip(".")
    return normalized or None


def _format_from_path(path: str | Path) -> str | None:
    return _normalize_format(Path(path).suffix)


def read_mesh(
    path: str | Path,
    file_format: str | None = None,
    *,
    progress: bool = False,
    progress_interval: int = 500_000,
) -> meshio.Mesh:
    """Read a mesh using EMS-specific readers when available, else meshio.

    ``progress`` and ``progress_interval`` are currently used by the streaming
    Femap Neutral reader. Other formats accept the options and read normally.
    """
    mesh_format = _normalize_format(file_format) or _format_from_path(path)
    reader = _READERS.get(mesh_format)
    if reader is not None:
        if mesh_format in {"neu", "femap"}:
            return reader(
                Path(path),
                progress=progress,
                progress_interval=progress_interval,
            )
        return reader(Path(path))
    return meshio.read(path, file_format=file_format)


def write_mesh(
    path: str | Path,
    mesh: meshio.Mesh,
    file_format: str | None = None,
) -> None:
    """Write a mesh using EMS-specific writers when available, else meshio."""
    mesh_format = _normalize_format(file_format) or _format_from_path(path)
    writer = _WRITERS.get(mesh_format)
    if writer is not None:
        writer(Path(path), mesh)
        return
    meshio.write(path, mesh, file_format=file_format)
