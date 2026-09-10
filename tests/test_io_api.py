from pathlib import Path

import meshio
import numpy as np

from ems_file_format_converter import read_mesh, write_mesh


def _sample_path(name: str) -> Path:
    return Path(__file__).resolve().parents[1] / "sample" / name


def _concat_cell_data(mesh: meshio.Mesh, key: str) -> np.ndarray:
    return np.concatenate([np.asarray(values) for values in mesh.cell_data[key]])


def _assert_mesh_metadata_preserved(before: meshio.Mesh, after: meshio.Mesh) -> None:
    np.testing.assert_allclose(after.points, before.points)
    np.testing.assert_array_equal(after.point_data["id"], before.point_data["id"])
    np.testing.assert_array_equal(
        _concat_cell_data(after, "element_id"),
        _concat_cell_data(before, "element_id"),
    )
    np.testing.assert_array_equal(
        _concat_cell_data(after, "property_id"),
        _concat_cell_data(before, "property_id"),
    )


def _assert_counts_by_type_preserved(before: meshio.Mesh, after: meshio.Mesh) -> None:
    def counts(mesh: meshio.Mesh) -> dict[str, int]:
        result = {}
        for block in mesh.cells:
            result[block.type] = result.get(block.type, 0) + block.data.shape[0]
        return result

    assert counts(after) == counts(before)


def test_unified_api_roundtrips_atlas_mesh(tmp_path: Path):
    mesh = read_mesh(_sample_path("mesh_sample.atl"))

    out = tmp_path / "roundtrip.atl"
    write_mesh(out, mesh)
    mesh2 = read_mesh(out)

    _assert_mesh_metadata_preserved(mesh, mesh2)
    _assert_counts_by_type_preserved(mesh, mesh2)


def test_unified_api_roundtrips_unv_mesh(tmp_path: Path):
    mesh = read_mesh(_sample_path("mesh_sample.unv"))

    out = tmp_path / "roundtrip.unv"
    write_mesh(out, mesh)
    mesh2 = read_mesh(out)

    _assert_mesh_metadata_preserved(mesh, mesh2)
    _assert_counts_by_type_preserved(mesh, mesh2)


def test_unified_api_roundtrips_neu_mesh(tmp_path: Path):
    mesh = read_mesh(_sample_path("mesh_sample.neu"))

    out = tmp_path / "roundtrip.neu"
    write_mesh(out, mesh)
    mesh2 = read_mesh(out)

    _assert_mesh_metadata_preserved(mesh, mesh2)
    _assert_counts_by_type_preserved(mesh, mesh2)


def test_femap_reader_reports_streaming_progress(capsys):
    mesh = read_mesh(_sample_path("mesh_sample.neu"), progress=True, progress_interval=1)

    output = capsys.readouterr().out
    assert len(mesh.points) > 0
    assert "Femap read: start" in output
    assert "nodes=" in output
    assert "elements=" in output
    assert "phase=complete" in output


def test_unified_api_uses_meshio_for_generic_formats(tmp_path: Path):
    mesh = meshio.Mesh(
        points=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        ),
        cells=[("triangle", np.array([[0, 1, 2]]))],
        point_data={"id": np.array([101, 102, 103])},
        cell_data={
            "element_id": [np.array([201])],
            "property_id": [np.array([301])],
        },
    )

    out = tmp_path / "mesh.vtu"
    write_mesh(out, mesh)
    mesh2 = read_mesh(out)

    _assert_mesh_metadata_preserved(mesh, mesh2)
    _assert_counts_by_type_preserved(mesh, mesh2)


def test_unified_api_accepts_explicit_format_alias(tmp_path: Path):
    mesh = read_mesh(_sample_path("mesh_sample.atl"), file_format="atlas")

    out = tmp_path / "roundtrip.mesh"
    write_mesh(out, mesh, file_format="atlas")
    mesh2 = read_mesh(out, file_format="atlas")

    _assert_mesh_metadata_preserved(mesh, mesh2)
    _assert_counts_by_type_preserved(mesh, mesh2)
