from pathlib import Path

import numpy as np

from femap import neu


def test_read_and_roundtrip_neu_mesh(tmp_path: Path):
    for fname in ("sample_mesh.neu", "sample_mesh10.3.neu"):
        sample_path = Path(__file__).resolve().parents[1] / "sample" / fname
        assert sample_path.exists(), f"Missing sample file: {sample_path}"
        mesh = neu.read_neu(sample_path)
        assert mesh.points.shape[1] == 3
        assert len(mesh.cells) > 0
        out = tmp_path / f"rt_{fname}"
        neu.write_neu(out, mesh)
        mesh2 = neu.read_neu(out)
        np.testing.assert_allclose(mesh2.points, mesh.points)

        # Compare counts per cell type
        def counts(m):
            d = {}
            for blk in m.cells:
                d[blk.type] = d.get(blk.type, 0) + blk.data.shape[0]
            return d

        assert counts(mesh2) == counts(mesh)


def test_read_and_roundtrip_neu_post(tmp_path: Path):
    for fname in ("sample_post.neu", "sample_post10.3.neu"):
        sample_path = Path(__file__).resolve().parents[1] / "sample" / fname
        assert sample_path.exists(), f"Missing sample file: {sample_path}"
        steps = neu.read_neu_post(sample_path)
        assert len(steps) >= 1
        out = tmp_path / f"rt_{fname}"
        neu.write_neu_post(out, steps, mode="vector+scalar")
        steps2 = neu.read_neu_post(out)
        assert len(steps2) == len(steps)
        for a, b in zip(steps, steps2):
            assert a["step"] == b["step"]
            assert a["substep"] == b["substep"]
            np.testing.assert_allclose(a["time"], b["time"], rtol=0, atol=1e-12)
            assert set(a["elements"].keys()) == set(b["elements"].keys())
            assert set(a["nodes"].keys()) == set(b["nodes"].keys())
