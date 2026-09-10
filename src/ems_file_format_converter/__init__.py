# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Science Solutions International Laboratory, Inc.
from . import atlas, femap, gmsh, unv
from .io import read_mesh, write_mesh

__version__ = "0.6.0"

__all__ = ["atlas", "femap", "gmsh", "unv", "read_mesh", "write_mesh", "__version__"]
