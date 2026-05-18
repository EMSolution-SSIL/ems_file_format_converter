# EMS File Format Converter

A lightweight converter for CAE mesh and post data. Supports formats used by Science Solutions International Laboratory, Inc. (SSIL) electromagnetic solver "EMSolution": ATLAS test file format (`.atl`), I‑DEAS universal file format (`.unv`), and Femap Neutral file format (`.neu`), focusing on round‑tripping metadata such as IDs and property numbers.

## Features
- **Supported formats**: ATLAS (mesh & post), UNV (mesh & post), Femap NEU (mesh & post)
- **Metadata preservation**: Node ID, Element ID, and `iprop` when available
- **CLI**: Single entry point for conversions and post I/O
- **Universal package**: Pure Python package with no OS-specific native extension
- **Tests/CI**: `pytest` included; GitHub Actions can run manual CI on `ubuntu-latest` and `windows-latest`
- **PyPI publishing**: Manual publish to PyPI is supported from GitHub Actions

## Installation
Requires Python 3.10+.

Install from PyPI:

```powershell
pip install ems-file-format-converter
```

You can also install from a Wheel attached to a Release or build locally:

```powershell
# Install from a downloaded Wheel
pip install --force-reinstall path\to\ems_file_format_converter-0.1.0-py3-none-any.whl

# Build locally then install
python -m build
pip install --force-reinstall dist/ems_file_format_converter-0.1.0-py3-none-any.whl
```

## CLI Usage

Convert meshes (input auto-detected by extension, specify output format):

```powershell
ems-file-format-converter --in mesh_sample.atl --out out.unv
ems-file-format-converter --in sample_mesh.unv --out out.atl
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

## GitHub Actions Workflow

The `CI and Publish` GitHub Actions workflow is manual-only. It does not run on `push` or `pull_request`.

- Open the `Actions` tab in GitHub and run `CI and Publish`.
- For normal CI validation, leave `publish_to_pypi` set to `false`.
- The workflow then tests and build-checks the package on both `ubuntu-latest` and `windows-latest`.
- Since this package is a universal pure Python package, the multi-OS jobs are for compatibility validation rather than producing OS-specific artifacts.
- After that, it builds the release distributions (`sdist` and `py3-none-any` wheel) once on Ubuntu and runs `twine check`.

## Publishing to PyPI

PyPI publication is handled by the same manual workflow.

- Create a Git tag such as `v0.5.1`.
- In `Run workflow`, start the workflow normally from the default branch.
- Set `publish_to_pypi` to `true`.
- Set `release_tag` to the tag name, for example `v0.5.1`.
- The workflow checks out that tag, builds the distributions, and uploads them to PyPI.
- PyPI Trusted Publishing must be configured for this GitHub repository in advance.

## License

MIT License. See `LICENSE`.
