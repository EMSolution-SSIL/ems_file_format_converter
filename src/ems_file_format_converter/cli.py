# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Science Solutions International Laboratory, Inc.
from pathlib import Path
import argparse

from . import atlas
from .io import read_mesh, write_mesh


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument(
        "--informat",
        choices=["neu", "femap", "unv", "atl", "atlas", "msh", "gmsh"],
    )
    parser.add_argument("--outformat", help="meshio file_format or EMS format")
    parser.add_argument(
        "--progress",
        action="store_true",
        help="show Femap NEU node/element loading progress",
    )
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=500_000,
        metavar="N",
        help="report Femap NEU progress every N records (default: 500000)",
    )
    # Post data options
    parser.add_argument("--post-in", help="post data input file (STEP/EVAL/STRE)")
    parser.add_argument("--post-out", help="post data output file")
    parser.add_argument(
        "--post-mode",
        choices=["scalar", "vector", "vector+scalar", "components"],
        default="components",
        help="Post data write mode (default: components)",
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
        steps = atlas.read_post(Path(args.post_in))
        atlas.write_post(Path(args.post_out), steps, mode=args.post_mode)
        return

    mesh = read_mesh(
        in_path,
        file_format=args.informat,
        progress=args.progress,
        progress_interval=args.progress_interval,
    )
    write_mesh(out_path, mesh, file_format=args.outformat)
