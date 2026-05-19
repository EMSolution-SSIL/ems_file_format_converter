# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Science Solutions International Laboratory, Inc.
from pathlib import Path
from typing import Dict

import numpy as np

from ems_file_format_converter import atlas, femap, unv

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "motor"


def _counts_by_type(mesh) -> Dict[str, int]:
    d: Dict[str, int] = {}
    for blk in mesh.cells:
        d[blk.type] = d.get(blk.type, 0) + blk.data.shape[0]
    return d


def _flat(mesh, key: str) -> np.ndarray:
    if key not in mesh.cell_data:
        return np.array([], dtype=int)
    return np.concatenate([np.asarray(a) for a in mesh.cell_data[key]])


def _assert_mesh_equivalent(ref, out, rtol: float = 1e-5, atol: float = 1e-10) -> None:
    np.testing.assert_allclose(out.points, ref.points, rtol=rtol, atol=atol)
    np.testing.assert_array_equal(out.point_data["id"], ref.point_data["id"])
    assert _counts_by_type(out) == _counts_by_type(ref)
    np.testing.assert_array_equal(_flat(out, "element_id"), _flat(ref, "element_id"))
    np.testing.assert_array_equal(_flat(out, "property_id"), _flat(ref, "property_id"))


def test_convert_motor_atlas_to_neu(tmp_path: Path):
    mesh_in = DATA_DIR / "post_geom.atl"
    mesh_out = tmp_path / "mesh_out.neu"

    mesh = atlas.read_mesh(mesh_in)
    femap.write_mesh(mesh_out, mesh)

    assert mesh_out.exists() and mesh_out.stat().st_size > 0

    mesh2 = femap.read_mesh(mesh_out)
    _assert_mesh_equivalent(mesh, mesh2)


def test_convert_motor_atlas_to_unv(tmp_path: Path):
    mesh_in = DATA_DIR / "post_geom.atl"
    mesh_out = tmp_path / "mesh_out.unv"

    mesh = atlas.read_mesh(mesh_in)
    unv.write_mesh(mesh_out, mesh)

    assert mesh_out.exists() and mesh_out.stat().st_size > 0

    mesh2 = unv.read_mesh(mesh_out)
    _assert_mesh_equivalent(mesh, mesh2)


def test_convert_motor_neu_to_atlas(tmp_path: Path):
    mesh_in = DATA_DIR / "post_geom.neu"
    mesh_out = tmp_path / "mesh_out.atl"

    mesh = femap.read_mesh(mesh_in)
    atlas.write_mesh(mesh_out, mesh)

    assert mesh_out.exists() and mesh_out.stat().st_size > 0

    mesh2 = atlas.read_mesh(mesh_out)
    _assert_mesh_equivalent(mesh, mesh2)


def test_convert_motor_neu_to_unv(tmp_path: Path):
    mesh_in = DATA_DIR / "post_geom.neu"
    mesh_out = tmp_path / "mesh_out.unv"

    mesh = femap.read_mesh(mesh_in)
    unv.write_mesh(mesh_out, mesh)

    assert mesh_out.exists() and mesh_out.stat().st_size > 0

    mesh2 = unv.read_mesh(mesh_out)
    _assert_mesh_equivalent(mesh, mesh2)


def test_convert_motor_unv_to_atlas(tmp_path: Path):
    mesh_in = DATA_DIR / "post_geom.unv"
    mesh_out = tmp_path / "mesh_out.atl"

    mesh = unv.read_mesh(mesh_in)
    atlas.write_mesh(mesh_out, mesh)

    assert mesh_out.exists() and mesh_out.stat().st_size > 0

    mesh2 = atlas.read_mesh(mesh_out)
    _assert_mesh_equivalent(mesh, mesh2)


def test_convert_motor_unv_to_neu(tmp_path: Path):
    mesh_in = DATA_DIR / "post_geom.unv"
    mesh_out = tmp_path / "mesh_out.neu"

    mesh = unv.read_mesh(mesh_in)
    femap.write_mesh(mesh_out, mesh)

    assert mesh_out.exists() and mesh_out.stat().st_size > 0

    mesh2 = femap.read_mesh(mesh_out)
    _assert_mesh_equivalent(mesh, mesh2)
