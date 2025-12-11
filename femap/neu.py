# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Science Solutions International Laboratory, Inc.
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import meshio

# Minimal Femap Neutral (.neu) reader/writer supporting v4.1 and v10.3 samples.
# We parse nodes and elements from common sections and map to meshio types similar to ATLAS.

FEMAP_TO_MESHIO: Dict[str, Tuple[str, int]] = {
    # Femap element names to meshio types (approximate)
    "TRIA": ("triangle", 3),
    "QUAD": ("quad", 4),
    "TETRA": ("tetra", 4),
    "PYRAMID": ("pyramid", 5),
    "WEDGE": ("wedge", 6),
    "BRICK": ("hexahedron", 8),
}

MESHIO_TO_FEMAP: Dict[str, Tuple[str, int]] = {v[0]: (k, v[1]) for k, v in FEMAP_TO_MESHIO.items()}


def _detect_version(lines: List[str]) -> str:
    text = "".join(lines)
    if "$ Femap Neutral File" in text and "Version 10" in text:
        return "10.3"
    if "$ Femap Neutral File" in text and "Version 4" in text:
        return "4.1"
    # Fallback by heuristics
    return "unknown"


def read_neu(path: str | Path) -> meshio.Mesh:
    path = Path(path)
    with path.open(encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    version = _detect_version(lines)

    nodes: Dict[int, Tuple[float, float, float]] = {}
    cells_by_type: Dict[str, List[List[int]]] = {}

    it = iter(lines)
    mode = None

    def parse_node_line(s: str) -> Tuple[int, float, float, float] | None:
        parts = s.strip().split()
        if len(parts) < 4:
            return None
        try:
            nid = int(parts[0])
            x, y, z = map(float, parts[1:4])
            return nid, x, y, z
        except ValueError:
            return None

    def parse_elem_line(s: str) -> Tuple[str, int, List[int]] | None:
        # Accept formats like: id type prop ... n1 n2 n3...
        parts = s.strip().split()
        if len(parts) < 6:
            return None
        try:
            _eid = int(parts[0])
        except ValueError:
            return None
        etype_token = parts[1].upper()
        # Some files may use numeric codes; convert to names if necessary
        if etype_token.isdigit():
            code = int(etype_token)
            # Map common codes
            if code in (91, 92):
                etype_token = "TRIA"
            elif code in (94, 95):
                etype_token = "QUAD"
            elif code in (111,):
                etype_token = "TETRA"
            elif code in (112,):
                etype_token = "WEDGE"
            elif code in (113,):
                etype_token = "PYRAMID"
            elif code in (115, 116):
                etype_token = "BRICK"
        if etype_token not in FEMAP_TO_MESHIO:
            return None
        ctype, expected = FEMAP_TO_MESHIO[etype_token]
        # Nodes usually start later in the line; try last tokens
        try:
            nodes_list = [int(x) for x in parts if x.isdigit()][-expected:]
        except Exception:
            return None
        if len(nodes_list) != expected:
            return None
        return ctype, expected, nodes_list

    for raw in it:
        s = raw.strip()
        if not s:
            continue
        if s.startswith("$"):
            # comment/title
            if "Nodes" in s:
                mode = "nodes"
            elif "Elements" in s:
                mode = "elements"
            continue
        if mode == "nodes":
            parsed = parse_node_line(s)
            if parsed:
                nid, x, y, z = parsed
                nodes[nid] = (x, y, z)
            else:
                mode = None
        elif mode == "elements":
            parsed = parse_elem_line(s)
            if parsed:
                ctype, _exp, conn = parsed
                cells_by_type.setdefault(ctype, []).append(conn)
            else:
                # keep scanning
                pass

    # Build mesh
    sorted_node_ids = sorted(nodes)
    id2idx = {nid: i for i, nid in enumerate(sorted_node_ids)}
    points = np.array([nodes[nid] for nid in sorted_node_ids], dtype=float)

    cells = []
    for ctype, conn_list in cells_by_type.items():
        data = np.array([[id2idx[n] for n in conn] for conn in conn_list], dtype=int)
        cells.append((ctype, data))

    return meshio.Mesh(points=points, cells=cells, point_data={"id": np.array(sorted_node_ids, int)})


def write_neu(path: str | Path, mesh: meshio.Mesh, version: str = "10.3") -> None:
    path = Path(path)

    # Normalize cells
    cells: List[Tuple[str, np.ndarray]] = []
    for block in mesh.cells:
        try:
            ctype = block.type
            cdata = block.data
        except AttributeError:
            ctype, cdata = block
        cells.append((ctype, np.asarray(cdata, dtype=int)))

    # Node IDs
    point_ids = None
    if isinstance(getattr(mesh, "point_data", None), dict) and "id" in mesh.point_data:
        pid = np.asarray(mesh.point_data["id"]).reshape(-1)
        if pid.size == len(mesh.points):
            point_ids = pid.astype(int)

    out: List[str] = []
    out.append(f"$ Femap Neutral File - Version {version}\n")
    out.append("$ Nodes\n")
    for i, (x, y, z) in enumerate(np.asarray(mesh.points, float)):
        nid = int(point_ids[i]) if point_ids is not None else i + 1
        out.append(f"{nid:8d} {x: .8e} {y: .8e} {z: .8e}\n")
    out.append("$ Elements\n")

    eid_counter = 1
    for ctype, conn in cells:
        if ctype not in MESHIO_TO_FEMAP:
            raise ValueError(f"Unsupported cell type for Femap writer: {ctype}")
        etok, expected = MESHIO_TO_FEMAP[ctype]
        for e in conn:
            eid = eid_counter
            eid_counter += 1
            node_ids = [int(point_ids[int(n)]) if point_ids is not None else int(n) + 1 for n in e[:expected]]
            out.append(f"{eid:8d} {etok:<8s}    1    " + " ".join(f"{nid:8d}" for nid in node_ids) + "\n")

    path.write_text("".join(out), encoding="utf-8")


# --- Femap post data (neutral) ---


def read_neu_post(path: str | Path) -> List[dict]:
    path = Path(path)
    with path.open(encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    it = iter(lines)
    steps: List[dict] = []

    def _read_block() -> Dict[int, Dict[str, np.ndarray | float]]:
        data: Dict[int, Dict[str, np.ndarray | float]] = {}
        for raw in it:
            s = raw.strip()
            if not s:
                continue
            if s.startswith("$"):
                break
            parts = s.split()
            try:
                idv = int(parts[0])
            except ValueError:
                continue
            vals = [float(x) for x in parts[1:]]
            vec = (vals + [0.0, 0.0, 0.0])[:3]
            val = vals[-1] if vals else 0.0
            data[idv] = {"vector": np.array(vec, float), "value": float(val)}
        return data

    for raw in it:
        s = raw.strip()
        if not s:
            continue
        if s.startswith("$") and "Step" in s:
            # Header example: $ Step 1 Time 1.000E+00 Substep 1
            parts = s.split()
            try:
                step = int(parts[2])
            except Exception:
                step = len(steps) + 1
            try:
                time_idx = parts.index("Time") + 1
                time = float(parts[time_idx])
            except Exception:
                time = 0.0
            try:
                sub_idx = parts.index("Substep") + 1
                sub = int(parts[sub_idx])
            except Exception:
                sub = 1
            # Assume next a node or element data block
            # First block: Element Data
            elems = _read_block()
            # Next header
            nxt = next(it, "").strip()
            nodes = {}
            if "Node" in nxt:
                nodes = _read_block()
            steps.append({"step": step, "substep": sub, "time": time, "elements": elems, "nodes": nodes})

    return steps


def write_neu_post(path: str | Path, steps: List[dict], mode: str = "vector+scalar") -> None:
    path = Path(path)
    out: List[str] = []

    def fmt(v: float) -> str:
        return f"{v: .8e}"

    for st in steps:
        step = int(st.get("step", 1))
        sub = int(st.get("substep", 1))
        time = float(st.get("time", 0.0))
        out.append(f"$ Step {step} Time {time: .3e} Substep {sub}\n")
        # Element data
        out.append("$ Element Data\n")
        for eid in sorted(st.get("elements", {})):
            rec = st["elements"][eid]
            vx, vy, vz = np.asarray(rec.get("vector", [0.0, 0.0, 0.0]), float).tolist()
            val = float(rec.get("value", 0.0))
            if mode == "scalar":
                out.append(f"{eid:8d} {fmt(val)}\n")
            elif mode == "vector":
                out.append(f"{eid:8d} {fmt(vx)} {fmt(vy)} {fmt(vz)}\n")
            else:
                out.append(f"{eid:8d} {fmt(vx)} {fmt(vy)} {fmt(vz)} {fmt(val)}\n")
        # Node data
        out.append("$ Node Data\n")
        for nid in sorted(st.get("nodes", {})):
            rec = st["nodes"][nid]
            vx, vy, vz = np.asarray(rec.get("vector", [0.0, 0.0, 0.0]), float).tolist()
            val = float(rec.get("value", 0.0))
            if mode == "scalar":
                out.append(f"{nid:8d} {fmt(val)}\n")
            elif mode == "vector":
                out.append(f"{nid:8d} {fmt(vx)} {fmt(vy)} {fmt(vz)}\n")
            else:
                out.append(f"{nid:8d} {fmt(vx)} {fmt(vy)} {fmt(vz)} {fmt(val)}\n")

    path.write_text("".join(out), encoding="utf-8")
