from pathlib import Path

import numpy as np
import meshio

from ems_file_format_converter import unv


def test_read_and_roundtrip_unv_mesh(tmp_path: Path):
    sample_path = Path(__file__).resolve().parents[1] / "sample" / "mesh_sample.unv"
    assert sample_path.exists(), f"Missing sample file: {sample_path}"

    mesh = unv.read_mesh(sample_path)
    assert mesh.points.shape[1] == 3
    assert len(mesh.cells) > 0
    assert "id" in mesh.point_data
    assert "element_id" in mesh.cell_data
    assert "property_id" in mesh.cell_data
    assert "material_id" in mesh.cell_data
    assert "id" not in mesh.cell_data
    assert "prop1" not in mesh.cell_data
    assert "prop2" not in mesh.cell_data

    # Round-trip write and re-read
    out = tmp_path / "mesh_roundtrip.unv"
    unv.write_mesh(out, mesh)
    mesh2 = unv.read_mesh(out)

    np.testing.assert_allclose(mesh2.points, mesh.points)
    np.testing.assert_array_equal(mesh2.point_data["id"], mesh.point_data["id"])
    assert "element_id" in mesh2.cell_data
    assert "property_id" in mesh2.cell_data
    assert "material_id" in mesh2.cell_data
    np.testing.assert_array_equal(
        np.concatenate([a for a in mesh2.cell_data["element_id"]]),
        np.concatenate([a for a in mesh.cell_data["element_id"]]),
    )
    np.testing.assert_array_equal(
        np.concatenate([a for a in mesh2.cell_data["property_id"]]),
        np.concatenate([a for a in mesh.cell_data["property_id"]]),
    )
    np.testing.assert_array_equal(
        np.concatenate([a for a in mesh2.cell_data["material_id"]]),
        np.concatenate([a for a in mesh.cell_data["material_id"]]),
    )
    # Compare cell types and sizes
    types2 = [getattr(c, "type", None) or getattr(c, "type") for c in mesh2.cells]
    types1 = [getattr(c, "type", None) or getattr(c, "type") for c in mesh.cells]
    assert types2 == types1

    # Compare counts per cell type (robust to block grouping differences)
    def counts_by_type(m):
        out = {}
        for blk in m.cells:
            t = blk.type
            out[t] = out.get(t, 0) + blk.data.shape[0]
        return out

    assert counts_by_type(mesh2) == counts_by_type(mesh)


def test_read_and_roundtrip_unv_post(tmp_path: Path):
    sample_path = Path(__file__).resolve().parents[1] / "sample" / "post_sample.unv"
    assert sample_path.exists(), f"Missing sample file: {sample_path}"

    steps = unv.read_post(sample_path)
    assert len(steps) >= 1

    out = tmp_path / "post_roundtrip.unv"
    unv.write_post(out, steps, mode="components", name="TestData")
    steps2 = unv.read_post(out)

    assert len(steps2) == len(steps)
    for a, b in zip(steps, steps2):
        assert a["step"] == b["step"]
        assert a["substep"] == b["substep"]
        np.testing.assert_allclose(a["time"], b["time"], rtol=0, atol=1e-12)
        assert set(a["elements"].keys()) == set(b["elements"].keys())
        assert set(a["nodes"].keys()) == set(b["nodes"].keys())


if __name__ == "__main__":
    test_read_and_roundtrip_unv_mesh(Path("."))
    test_read_and_roundtrip_unv_post(Path("."))
