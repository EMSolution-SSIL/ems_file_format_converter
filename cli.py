# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Science Solutions International Laboratory, Inc.
# ems-file-format-converter/cli.py
from pathlib import Path
import argparse
import meshio

from atlas import atlas
from femap import neu
from unv import unv

READERS = {
    ".neu": neu.read_neu,
    ".unv": unv.read_unv,
    ".atl": atlas.read_atlas,  # ATLAS 独自拡張子（社内仕様）
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument("--informat", choices=["neu", "unv", "atl"])
    parser.add_argument("--outformat", help="meshio の file_format を指定 (任意)")
    # Post data options
    parser.add_argument("--post-in", help="post data input file (STEP/EVAL/STRE)")
    parser.add_argument("--post-out", help="post data output file")
    parser.add_argument(
        "--post-mode",
        choices=["scalar", "vector", "vector+scalar"],
        default="vector+scalar",
        help="Post data write mode: values per ID",
    )
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)

    # If post conversion requested, handle separately
    if args.post_in or args.post_out:
        if not args.post_in:
            raise SystemExit("--post-in is required when using post options")
        if not args.post_out:
            raise SystemExit("--post-out is required when using post options")
        steps = atlas.read_atlas_post(Path(args.post_in))
        atlas.write_atlas_post(Path(args.post_out), steps, mode=args.post_mode)
        return

    # 入力フォーマット判定 (mesh)
    if args.informat:
        informat = args.informat
    else:
        informat = in_path.suffix.lstrip(".").lower()

    if informat in ("neu", "unv", "atl"):
        if informat == "neu":
            mesh = neu.read_neu(in_path)
        elif informat == "unv":
            mesh = unv.read_unv(in_path)
        else:
            # "atlas" または "atl"
            mesh = atlas.read_atlas(in_path)
    else:
        # meshio がそのまま読める形式は meshio に任せる
        mesh = meshio.read(in_path)

    # Decide writer
    outfmt = (args.outformat or "").lower() if args.outformat else None
    if (outfmt in ("atlas", "atl")) or (outfmt is None and out_path.suffix.lower() == ".atl"):
        atlas.write_atlas(out_path, mesh)
    elif (outfmt == "unv") or (outfmt is None and out_path.suffix.lower() == ".unv"):
        unv.write_unv(out_path, mesh)
    else:
        meshio.write(out_path, mesh, file_format=args.outformat)
