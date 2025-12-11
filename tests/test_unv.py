from pathlib import Path

import numpy as np
import meshio

from unv import unv


def test_read_and_roundtrip_unv_mesh(tmp_path: Path):
    sample_path = Path(__file__).resolve().parents[1] / "sample" / "mesh_sample.unv"
    assert sample_path.exists(), f"Missing sample file: {sample_path}"

    mesh = unv.read_unv(sample_path)
    assert mesh.points.shape[1] == 3
    assert len(mesh.cells) > 0

    # Round-trip write and re-read
    out = tmp_path / "mesh_roundtrip.unv"
    unv.write_unv(out, mesh)
    mesh2 = unv.read_unv(out)

    np.testing.assert_allclose(mesh2.points, mesh.points)
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

    steps = unv.read_unv_post(sample_path)
    assert len(steps) >= 1

    out = tmp_path / "post_roundtrip.unv"
    unv.write_unv_post(out, steps, mode="vector+scalar", name="TestData")
    steps2 = unv.read_unv_post(out)

    assert len(steps2) == len(steps)
    for a, b in zip(steps, steps2):
        assert a["step"] == b["step"]
        assert a["substep"] == b["substep"]
        np.testing.assert_allclose(a["time"], b["time"], rtol=0, atol=1e-12)
        assert set(a["elements"].keys()) == set(b["elements"].keys())
        assert set(a["nodes"].keys()) == set(b["nodes"].keys())
