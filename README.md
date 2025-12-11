# EMS File Format Converter

このリポジトリは、CAE向けメッシュ／ポストデータの簡易コンバータです。サイエンスソリューションズ社（Science Solutions International Laboratory, Inc.; SSIL）の電磁界解析ソフトウェア「EMSolution」で用いるATLASテキストファイルフォーマット（`.atl`）、I‑DEAS Universal file format（`.unv`）、Femap Neutral file format（`.neu`）の読み書きに対応し、IDや物性番号などのメタデータのラウンドトリップ保存を重視しています。

## 特長
- **対応フォーマット**: ATLAS（メッシュ・ポスト）、UNV（メッシュ・ポスト）、Femap NEU（メッシュ・ポスト）
- **メタデータ保持**: Node ID、Element ID、物性番号（`iprop`）を可能な限り保持
- **CLI**: 単一のCLIから変換・ポスト入出力が可能
- **テスト/CI**: `pytest` 完備、GitHub Actionsで自動テストとタグ時のPyPI公開

## インストール
事前にPython 3.10以上が必要です。

PyPI公開前の段階では、Releaseに添付した`whl`（Wheel）ファイルからのインストールを推奨します。

```powershell
# 例: ダウンロードしたWHLをインストール
pip install --force-reinstall path\to\ems_file_format_converter-0.1.0-py3-none-any.whl

# ソースからビルドしてWHL作成 → インストール
python -m build
pip install --force-reinstall dist/ems_file_format_converter-0.1.0-py3-none-any.whl
```

## 使い方（CLI）

メッシュ変換（入力は拡張子で自動判別、出力形式を指定）:

```powershell
ems-file-format-converter --in mesh_sample.atl --out out.unv
ems-file-format-converter --in sample_mesh.unv --out out.atl
```

ポストデータの読み書き（モード指定: `scalar|vector|vector+scalar`）:

```powershell
ems-file-format-converter --post-in post_sample.atl --post-out rt_post.atl --post-mode vector+scalar
```

対応拡張子:
- ATLAS: `.atl`
- UNV: `.unv`
- Femap Neutral: `.neu`

## Python API（例）

```python
from atlas import atlas
mesh = atlas.read_atlas("sample/mesh_sample.atl")
atlas.write_atlas("out.atl", mesh)

steps = atlas.read_atlas_post("sample/post_sample.atl")
atlas.write_atlas_post("out_post.atl", steps, mode="vector+scalar")
```

UNVやFemap NEUも同様に `unv.unv` / `femap.neu` のモジュールを利用できます。

## テスト

```powershell
pytest -q
```

## ライセンス

MITライセンスです。`LICENSE` を参照してください。

## 英語版README

英語版は `README_en.md` を参照してください。
# ems_file_format_converter
CAE file format converter
