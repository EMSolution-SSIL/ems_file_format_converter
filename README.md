# EMS File Format Converter

このリポジトリは、CAE向けメッシュ／ポストデータの簡易コンバータです。サイエンスソリューションズ社（Science Solutions International Laboratory, Inc.; SSIL）の電磁界解析ソフトウェア「EMSolution」で用いるATLASテキストファイルフォーマット（`.atl`）、I‑DEAS Universal file format（`.unv`）、Femap Neutral file format（`.neu`）の読み書きに対応し、IDや物性番号などのメタデータのラウンドトリップ保存を重視しています。

## 特長
- **対応フォーマット**: ATLAS（メッシュ・ポスト）、UNV（メッシュ・ポスト）、Femap NEU（メッシュ・ポスト）
- **メタデータ保持**: Node ID、Element ID、物性番号（`iprop`）を可能な限り保持
- **CLI**: 単一のCLIから変換・ポスト入出力が可能
- **ユニバーサルパッケージ**: OS依存のネイティブ拡張を含まない pure Python パッケージ
- **テスト/CI**: `pytest` 完備、GitHub Actionsで `ubuntu-latest` / `windows-latest` の手動CI実行に対応
- **PyPI公開**: GitHub Actionsから手動でPyPI公開可能

## インストール
事前にPython 3.10以上が必要です。

PyPI からインストールできます。

```powershell
pip install ems-file-format-converter
```

Releaseに添付した`whl`（Wheel）ファイルからのインストール、またはローカルビルドも可能です。

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

ポストデータの読み書き（モード指定: `components|scalar|vector|vector+scalar`）:

```powershell
ems-file-format-converter --post-in post_sample.atl --post-out rt_post.atl --post-mode components
```

`--post-mode` の意味:
- `components`（デフォルト）: すべての成分（component1..N）をそのまま出力（部分的に切り捨てない）
- `scalar`: component1 のみ出力
- `vector`: component1..3 を出力（不足分は 0）
- `vector+scalar`: component1..4 を出力（不足分は 0）

対応拡張子:
- ATLAS: `.atl`
- UNV: `.unv`
- Femap Neutral: `.neu`

## Python API（例）

```python
from ems_file_format_converter import atlas
mesh = atlas.read_mesh("sample/mesh_sample.atl")
atlas.write_mesh("out.atl", mesh)

steps = atlas.read_post("sample/post_sample.atl")
atlas.write_post("out_post.atl", steps, mode="components")
```

UNVやFemap NEUも同様に `ems_file_format_converter.unv` / `ems_file_format_converter.femap` のモジュールを利用できます。

## テスト

```powershell
pytest -q
```

## GitHub Actions ワークフロー

GitHub Actions の `CI and Publish` ワークフローは手動実行専用です。`push` や `pull_request` では起動しません。

- GitHub の `Actions` タブから `CI and Publish` を選び、`Run workflow` を実行します。
- 通常のCI確認では `publish_to_pypi` を `false` にします。
- この場合、`ubuntu-latest` と `windows-latest` の両方でテストと `python -m build` によるビルド確認を行います。
- 本パッケージはユニバーサルな pure Python パッケージのため、OSごとに別配布物を作るのではなく、互換性確認のために複数OSで検証しています。
- 配布物（`sdist` と `py3-none-any` wheel）はUbuntu上で1回だけ生成し、`twine check` を実行します。

## PyPI 公開

PyPI公開も同じワークフローから手動で行います。

- あらかじめ `v0.5.1` のような `v*.*.*` 形式のGitタグを作成しておきます。
- `Run workflow` 実行時は通常どおりデフォルトブランチから起動します。
- `publish_to_pypi` を `true` にし、`release_tag` に `v0.5.1` のようなタグ名を入力して実行します。
- ワークフローはそのタグをチェックアウトしてビルドし、ビルド済み配布物をPyPIへ公開します。
- 公開にはPyPI側で GitHub Actions Trusted Publishing の設定が必要です。

## ライセンス

MITライセンスです。`LICENSE` を参照してください。

## 英語版README

英語版は `README_en.md` を参照してください。
