from pathlib import Path

import numpy as np

from atlas import atlas


def test_read_and_roundtrip_atlas_mesh(tmp_path: Path):
    sample_path = Path(__file__).resolve().parents[1] / "sample" / "mesh_sample.atl"
    assert sample_path.exists(), f"Missing sample file: {sample_path}"

    mesh = atlas.read_atlas(sample_path)

    assert mesh.points.ndim == 2 and mesh.points.shape[1] == 3
    assert len(mesh.cells) > 0

    assert "id" in mesh.point_data
    assert "id" in mesh.cell_data
    assert "iprop" in mesh.cell_data

    orig_point_ids = np.asarray(mesh.point_data["id"]).copy()

    total_elems = 0
    orig_elem_ids = []
    orig_iprops = []
    for i, block in enumerate(mesh.cells):
        data = block.data if hasattr(block, "data") else block[1]
        total_elems += len(data)
        orig_elem_ids.extend(np.asarray(mesh.cell_data["id"][i]).tolist())
        orig_iprops.extend(np.asarray(mesh.cell_data["iprop"][i]).tolist())

    out_path = tmp_path / "roundtrip.atlas"
    atlas.write_atlas(out_path, mesh)
    mesh2 = atlas.read_atlas(out_path)

    np.testing.assert_equal(mesh2.point_data["id"], orig_point_ids)
    assert mesh2.points.shape == mesh.points.shape

    total_elems2 = sum(len(b.data if hasattr(b, "data") else b[1]) for b in mesh2.cells)
    assert total_elems2 == total_elems

    elem_ids2 = []
    iprops2 = []
    for i, _ in enumerate(mesh2.cells):
        elem_ids2.extend(np.asarray(mesh2.cell_data["id"][i]).tolist())
        iprops2.extend(np.asarray(mesh2.cell_data["iprop"][i]).tolist())

    assert elem_ids2 == orig_elem_ids
    assert iprops2 == orig_iprops


def test_read_and_roundtrip_atlas_post(tmp_path: Path):
    sample = Path(__file__).resolve().parents[1] / "sample" / "post_sample.atl"
    assert sample.exists(), f"Missing sample file: {sample}"

    steps = atlas.read_atlas_post(sample)
    assert len(steps) >= 2

    s0 = steps[0]
    assert {"step", "substep", "time", "elements", "nodes"} <= set(s0.keys())
    assert isinstance(s0["elements"], dict) and len(s0["elements"]) > 0
    assert isinstance(s0["nodes"], dict) and len(s0["nodes"]) > 0

    eid0, ed0 = next(iter(s0["elements"].items()))
    assert isinstance(eid0, int)
    assert np.asarray(ed0["vector"]).shape == (3,)
    assert isinstance(ed0["value"], float)

    out = tmp_path / "post_roundtrip.dat"
    atlas.write_atlas_post(out, steps)
    steps2 = atlas.read_atlas_post(out)

    assert len(steps2) == len(steps)

    for a, b in zip(steps[:2], steps2[:2]):
        assert a["step"] == b["step"]
        assert a["substep"] == b["substep"]
        np.testing.assert_allclose(a["time"], b["time"], rtol=0, atol=1e-12)
        assert set(a["elements"].keys()) == set(b["elements"].keys())
        assert set(a["nodes"].keys()) == set(b["nodes"].keys())

        for ids in list(a["elements"].keys())[:3]:
            va = a["elements"][ids]
            vb = b["elements"][ids]
            np.testing.assert_allclose(va["vector"], vb["vector"], rtol=0, atol=1e-12)
            np.testing.assert_allclose(va["value"], vb["value"], rtol=0, atol=1e-12)
        for ids in list(a["nodes"].keys())[:3]:
            va = a["nodes"][ids]
            vb = b["nodes"][ids]
            np.testing.assert_allclose(va["vector"], vb["vector"], rtol=0, atol=1e-12)
            np.testing.assert_allclose(va["value"], vb["value"], rtol=0, atol=1e-12)
