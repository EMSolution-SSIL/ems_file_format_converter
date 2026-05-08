from pathlib import Path

import numpy as np

from ems_file_format_converter import femap as neu


def test_read_and_roundtrip_neu_mesh(tmp_path: Path):
    for fname in ("mesh_sample.neu",):
        sample_path = Path(__file__).resolve().parents[1] / "sample" / fname
        assert sample_path.exists(), f"Missing sample file: {sample_path}"
        mesh = neu.read_mesh(sample_path)
        assert mesh.points.shape[1] == 3
        assert len(mesh.cells) > 0
        assert "id" in mesh.point_data
        assert "element_id" in mesh.cell_data
        assert "property_id" in mesh.cell_data
        assert "eid" not in mesh.cell_data
        assert "matid" not in mesh.cell_data
        out = tmp_path / f"rt_{fname}"
        neu.write_mesh(out, mesh)
        mesh2 = neu.read_mesh(out)
        np.testing.assert_allclose(mesh2.points, mesh.points)
        assert "element_id" in mesh2.cell_data
        assert "property_id" in mesh2.cell_data
        np.testing.assert_array_equal(
            np.concatenate([a for a in mesh2.cell_data["element_id"]]),
            np.concatenate([a for a in mesh.cell_data["element_id"]]),
        )
        np.testing.assert_array_equal(
            np.concatenate([a for a in mesh2.cell_data["property_id"]]),
            np.concatenate([a for a in mesh.cell_data["property_id"]]),
        )

        # Compare counts per cell type
        def counts(m):
            d = {}
            for blk in m.cells:
                d[blk.type] = d.get(blk.type, 0) + blk.data.shape[0]
            return d

        assert counts(mesh2) == counts(mesh)


def test_read_and_roundtrip_neu_post(tmp_path: Path):
    for fname in ("post_sample451.neu", "post_sample1051.neu"):
        sample_path = Path(__file__).resolve().parents[1] / "sample" / fname
        assert sample_path.exists(), f"Missing sample file: {sample_path}"
        steps = neu.read_post(sample_path)
        assert len(steps) >= 1
        out = tmp_path / f"rt_{fname}"
        neu.write_post(out, steps, mode="components")
        # Confirm file was written
        assert out.exists(), f"Output not created: {out}"
        assert out.stat().st_size > 0, "Output file is empty"


if __name__ == "__main__":
    test_read_and_roundtrip_neu_mesh(Path("."))
    test_read_and_roundtrip_neu_post(Path("."))
