from pathlib import Path


def test_version():
    # Ensure atlas module import works
    from atlas import atlas

    assert atlas is not None


def test_atlas_rw():
    # Directly read/write using atlas API instead of subprocess
    from atlas import atlas

    cwd = Path(__file__).resolve().parent

    # Treat files as ATLAS mesh & post data and round-trip
    mesh_in = cwd / "post_geom_org.atl"
    mesh_out = cwd / "post_geom_out.atl"
    post_in = cwd / "magnetic_org.atl"
    post_out = cwd / "magnetic_out.atl"

    # Read
    steps1 = atlas.read_atlas(mesh_in)
    steps2 = atlas.read_atlas_post(post_in)

    # Write
    atlas.write_atlas(mesh_out, steps1)
    atlas.write_atlas_post(post_out, steps2, mode="vector+scalar")

    # Verify existence
    assert mesh_out.exists(), f"Not written: {mesh_out}"
    assert post_out.exists(), f"Not written: {post_out}"


def test_unv_rw():
    # Ensure unv module import works
    from unv import unv

    cwd = Path(__file__).resolve().parent

    # Treat files as ATLAS mesh & post data and round-trip
    mesh_in = cwd / "post_geom_org.unv"
    mesh_out = cwd / "post_geom_out.unv"
    post_in = cwd / "magnetic_org.unv"
    post_out = cwd / "magnetic_out.unv"

    # Read
    steps1 = unv.read_unv(mesh_in)
    steps2 = unv.read_unv_post(post_in)

    # Write
    unv.write_unv(mesh_out, steps1)
    unv.write_unv_post(post_out, steps2, mode="vector+scalar")

    # Verify existence
    assert mesh_out.exists(), f"Not written: {mesh_out}"
    assert post_out.exists(), f"Not written: {post_out}"


def test_femap_rw():
    # Ensure femap module import works
    from femap import neu

    cwd = Path(__file__).resolve().parent

    # Treat files as Femap mesh & post data and round-trip
    mesh_in = cwd / "post_geom_org.neu"
    mesh_out = cwd / "post_geom_out.neu"
    post_in = cwd / "magnetic_org.neu"
    post_out = cwd / "magnetic_out.neu"

    # Read
    steps1 = neu.read_neu(mesh_in)
    steps2 = neu.read_neu_post(post_in)

    # Write
    neu.write_neu(mesh_out, steps1)
    neu.write_neu_post(post_out, steps2, mode="vector+scalar")

    # Verify existence
    assert mesh_out.exists(), f"Not written: {mesh_out}"
    assert post_out.exists(), f"Not written: {post_out}"


if __name__ == "__main__":
    test_version()
    test_atlas_rw()
    test_unv_rw()
    test_femap_rw()
