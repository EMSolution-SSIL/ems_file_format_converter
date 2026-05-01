# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Science Solutions International Laboratory, Inc.
"""Cross-format conversion tests using the CoreCoil data set.

Each test reads a mesh (or post data) from one format, writes it in another,
re-reads the result, and verifies that structure and canonical metadata are
preserved throughout.
"""
from pathlib import Path
from typing import Dict

import numpy as np

from ems_file_format_converter import atlas, femap, unv

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "CoreCoil"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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
    """Assert that `out` has the same geometry, node IDs, cell counts, and
    element_id / property_id arrays as `ref`."""
    np.testing.assert_allclose(out.points, ref.points, rtol=rtol, atol=atol,
                                err_msg="Point coordinates differ after conversion")
    np.testing.assert_array_equal(out.point_data["id"], ref.point_data["id"],
                                   err_msg="point_data['id'] differs after conversion")
    assert _counts_by_type(out) == _counts_by_type(ref), (
        f"Cell counts differ: {_counts_by_type(out)} vs {_counts_by_type(ref)}"
    )
    np.testing.assert_array_equal(
        _flat(out, "element_id"), _flat(ref, "element_id"),
        err_msg="element_id differs after conversion",
    )
    np.testing.assert_array_equal(
        _flat(out, "property_id"), _flat(ref, "property_id"),
        err_msg="property_id differs after conversion",
    )


# ---------------------------------------------------------------------------
# ATLAS → Femap NEU
# ---------------------------------------------------------------------------

def test_convert_corecoil_atlas_to_neu(tmp_path: Path):
    mesh_in = DATA_DIR / "post_geom_org.atl"
    post_in = DATA_DIR / "magnetic_org.atl"
    mesh_out = tmp_path / "mesh_out.neu"
    post_out = tmp_path / "post_out.neu"

    mesh = atlas.read_atlas(mesh_in)
    steps = atlas.read_atlas_post(post_in)

    femap.write_neu(mesh_out, mesh)
    femap.write_neu_post(post_out, steps, mode="vector+scalar")

    assert mesh_out.exists() and mesh_out.stat().st_size > 0
    assert post_out.exists() and post_out.stat().st_size > 0

    mesh2 = femap.read_neu(mesh_out)
    _assert_mesh_equivalent(mesh, mesh2)

    steps2 = femap.read_neu_post(post_out)
    assert len(steps2) == len(steps)
    s0, s1 = steps[0], steps2[0]
    assert set(s0["elements"].keys()) == set(s1["elements"].keys())
    assert set(s0["nodes"].keys()) == set(s1["nodes"].keys())


# ---------------------------------------------------------------------------
# ATLAS → UNV
# ---------------------------------------------------------------------------

def test_convert_corecoil_atlas_to_unv(tmp_path: Path):
    mesh_in = DATA_DIR / "post_geom_org.atl"
    post_in = DATA_DIR / "magnetic_org.atl"
    mesh_out = tmp_path / "mesh_out.unv"
    post_out = tmp_path / "post_out.unv"

    mesh = atlas.read_atlas(mesh_in)
    steps = atlas.read_atlas_post(post_in)

    unv.write_unv(mesh_out, mesh)
    unv.write_unv_post(post_out, steps)

    assert mesh_out.exists() and mesh_out.stat().st_size > 0
    assert post_out.exists() and post_out.stat().st_size > 0

    mesh2 = unv.read_unv(mesh_out)
    _assert_mesh_equivalent(mesh, mesh2)

    steps2 = unv.read_unv_post(post_out)
    assert len(steps2) == len(steps)
    s0, s1 = steps[0], steps2[0]
    assert set(s0["elements"].keys()) == set(s1["elements"].keys())
    assert set(s0["nodes"].keys()) == set(s1["nodes"].keys())


# ---------------------------------------------------------------------------
# Femap NEU → ATLAS
# ---------------------------------------------------------------------------

def test_convert_corecoil_neu_to_atlas(tmp_path: Path):
    mesh_in = DATA_DIR / "post_geom_org.neu"
    mesh_out = tmp_path / "mesh_out.atl"

    mesh = femap.read_neu(mesh_in)
    atlas.write_atlas(mesh_out, mesh)

    assert mesh_out.exists() and mesh_out.stat().st_size > 0

    mesh2 = atlas.read_atlas(mesh_out)
    _assert_mesh_equivalent(mesh, mesh2)


# ---------------------------------------------------------------------------
# Femap NEU → UNV
# ---------------------------------------------------------------------------

def test_convert_corecoil_neu_to_unv(tmp_path: Path):
    mesh_in = DATA_DIR / "post_geom_org.neu"
    mesh_out = tmp_path / "mesh_out.unv"

    mesh = femap.read_neu(mesh_in)
    unv.write_unv(mesh_out, mesh)

    assert mesh_out.exists() and mesh_out.stat().st_size > 0

    mesh2 = unv.read_unv(mesh_out)
    _assert_mesh_equivalent(mesh, mesh2)


# ---------------------------------------------------------------------------
# UNV → ATLAS
# ---------------------------------------------------------------------------

def test_convert_corecoil_unv_to_atlas(tmp_path: Path):
    mesh_in = DATA_DIR / "post_geom_org.unv"
    mesh_out = tmp_path / "mesh_out.atl"

    mesh = unv.read_unv(mesh_in)
    atlas.write_atlas(mesh_out, mesh)

    assert mesh_out.exists() and mesh_out.stat().st_size > 0

    mesh2 = atlas.read_atlas(mesh_out)
    _assert_mesh_equivalent(mesh, mesh2)


# ---------------------------------------------------------------------------
# UNV → Femap NEU
# ---------------------------------------------------------------------------

def test_convert_corecoil_unv_to_neu(tmp_path: Path):
    mesh_in = DATA_DIR / "post_geom_org.unv"
    mesh_out = tmp_path / "mesh_out.neu"

    mesh = unv.read_unv(mesh_in)
    femap.write_neu(mesh_out, mesh)

    assert mesh_out.exists() and mesh_out.stat().st_size > 0

    mesh2 = femap.read_neu(mesh_out)
    _assert_mesh_equivalent(mesh, mesh2)


# ---------------------------------------------------------------------------
# Chained: ATLAS → NEU → UNV → ATLAS
# ---------------------------------------------------------------------------

def test_convert_corecoil_chained_atl_to_neu_to_unv_to_atl(tmp_path: Path):
    """Verify that metadata survives a three-step chain."""
    mesh0 = atlas.read_atlas(DATA_DIR / "post_geom_org.atl")

    p1 = tmp_path / "step1.neu"
    femap.write_neu(p1, mesh0)
    mesh1 = femap.read_neu(p1)

    p2 = tmp_path / "step2.unv"
    unv.write_unv(p2, mesh1)
    mesh2 = unv.read_unv(p2)

    p3 = tmp_path / "step3.atl"
    atlas.write_atlas(p3, mesh2)
    mesh3 = atlas.read_atlas(p3)

    _assert_mesh_equivalent(mesh0, mesh3)


# ---------------------------------------------------------------------------
# Post-data round-trips using the current_org triplets
# ---------------------------------------------------------------------------

def test_convert_corecoil_post_atl_to_neu(tmp_path: Path):
    steps = atlas.read_atlas_post(DATA_DIR / "current_org.atl")
    out = tmp_path / "current.neu"
    femap.write_neu_post(out, steps, mode="vector+scalar")
    assert out.exists() and out.stat().st_size > 0
    steps2 = femap.read_neu_post(out)
    assert len(steps2) == len(steps)
    s0, s1 = steps[0], steps2[0]
    assert set(s0["elements"].keys()) == set(s1["elements"].keys())
    assert set(s0["nodes"].keys()) == set(s1["nodes"].keys())


def test_convert_corecoil_post_atl_to_unv(tmp_path: Path):
    steps = atlas.read_atlas_post(DATA_DIR / "current_org.atl")
    out = tmp_path / "current.unv"
    unv.write_unv_post(out, steps)
    assert out.exists() and out.stat().st_size > 0
    steps2 = unv.read_unv_post(out)
    assert len(steps2) == len(steps)
    s0, s1 = steps[0], steps2[0]
    assert set(s0["elements"].keys()) == set(s1["elements"].keys())
    assert set(s0["nodes"].keys()) == set(s1["nodes"].keys())
