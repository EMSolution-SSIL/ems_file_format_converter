# EMS File Format Converter

A lightweight converter for CAE mesh and post data. Supports formats used by Science Solutions International Laboratory, Inc. (SSIL) electromagnetic solver "EMSolution": ATLAS test file format (`.atl`), I‑DEAS universal file format (`.unv`), and Femap Neutral file format (`.neu`), focusing on round‑tripping metadata such as IDs and property numbers.

## Features
- **Supported formats**: ATLAS (mesh & post), UNV (mesh & post), Femap NEU (mesh & post)
- **Metadata preservation**: Node ID, Element ID, and `iprop` when available
- **CLI**: Single entry point for conversions and post I/O
- **Tests/CI**: `pytest` included; GitHub Actions runs tests and publishes on tagged releases

## Installation
Requires Python 3.10+.

Please install from the Wheel attached to a Release (or build locally):

```powershell
# Install from a downloaded Wheel
pip install --force-reinstall path\to\ems_file_format_converter-0.1.0-py3-none-any.whl

# Build locally then install
python -m build
pip install --force-reinstall dist/ems_file_format_converter-0.1.0-py3-none-any.whl
```

## CLI Usage

Convert meshes (input auto-detected by extension, output format determined by file extension):

```powershell
ems-file-format-converter mesh_sample.atl out.unv
ems-file-format-converter sample_mesh.unv out.atl
```

Explicitly specify formats:

```powershell
ems-file-format-converter mesh_sample.dat out.unv --informat atl --outformat unv
```

Post data I/O (mode: `components|scalar|vector|vector+scalar`):

```powershell
ems-file-format-converter --post-in post_sample.atl --post-out rt_post.atl --post-mode components
```

`--post-mode` meanings:
- `components` (default): write all components (component1..N) without truncation
- `scalar`: write component1 only
- `vector`: write component1..3 (pads missing with 0)
- `vector+scalar`: write component1..4 (pads missing with 0)

Supported extensions:
- ATLAS: `.atl`
- UNV: `.unv`
- Femap Neutral: `.neu`

## Python API (examples)

```python
from ems_file_format_converter import atlas
mesh = atlas.read_mesh("sample/mesh_sample.atl")
atlas.write_mesh("out.atl", mesh)

steps = atlas.read_post("sample/post_sample.atl")
atlas.write_post("out_post.atl", steps, mode="components")
```

For UNV and Femap NEU use `ems_file_format_converter.unv` and `ems_file_format_converter.femap` modules respectively.

## Tests

```powershell
pytest -q
```

## License

MIT License. See `LICENSE`.
