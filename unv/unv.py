# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Science Solutions International Laboratory, Inc.
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import meshio

# UNV element type codes (internal convention) mapped to meshio types
# Based on provided correspondence table
UNV_TO_MESHIO: Dict[int, Tuple[str, int]] = {
    161: ("vertex", 1),  # Lumped mass -> ATLAS 101
    11: ("line", 2),  # Rod -> ATLAS 102
    21: ("line", 2),  # Linear beam -> ATLAS 102
    91: ("triangle", 3),  # Linear triangle -> ATLAS 203
    94: ("quad", 4),  # Linear quadrilateral -> ATLAS 204
    95: ("quad8", 8),  # Parabolic quadrilateral -> ATLAS 208
    111: ("tetra", 4),  # Linear tetra -> ATLAS 304
    112: ("wedge", 6),  # Linear wedge -> ATLAS 306
    113: ("pyramid", 5),  # Linear pyramid -> ATLAS 305
    115: ("hexahedron", 8),  # Linear brick -> ATLAS 308
    116: ("hexahedron20", 20),  # Parabolic brick -> ATLAS 320
}

# Inverse map for writing: meshio type -> UNV code and node count
MESHIO_TO_UNV: Dict[str, Tuple[int, int]] = {v[0]: (k, v[1]) for k, v in UNV_TO_MESHIO.items()}


def read_unv(path: str | Path) -> meshio.Mesh:
    """Read a minimal I-DEAS UNV mesh with sections 2411 (nodes) and 2412 (elements).
    This supports the provided samples and common linear elements.
    """
    path = Path(path)
    with path.open() as f:
        lines = f.readlines()

    it = iter(lines)
    nodes: Dict[int, Tuple[float, float, float]] = {}
    cells_by_type: Dict[str, List[List[int]]] = {}

    def _read_nodes_block():
        # 2411 typically has an ID line followed by coordinate line(s)
        current_id: int | None = None
        for raw in it:
            s = raw.strip()
            if s.startswith("-1"):
                break
            if not s:
                continue
            parts = s.split()
            # Try integer id first token
            try:
                nid_candidate = int(parts[0])
                # If line has more floats after, this could be combined form
                if len(parts) >= 4:
                    x, y, z = map(float, parts[1:4])
                    nodes[nid_candidate] = (x, y, z)
                    current_id = None
                else:
                    current_id = nid_candidate
                continue
            except ValueError:
                pass
            # If not integer first token, try coordinates with stored id
            if current_id is not None:
                vals = [float(x) for x in parts]
                if len(vals) >= 3:
                    x, y, z = vals[:3]
                    nodes[current_id] = (x, y, z)
                    current_id = None

    def _read_elements_block():
        # block 2412: variable header lines; for the sample, assume: id, type, mat, color, n1..nN (one or multiple lines)
        for raw in it:
            s = raw.strip()
            if s.startswith("-1"):
                break
            if not s:
                continue
            parts = s.split()
            # Require first token integer (element id)
            try:
                eid = int(parts[0])
            except ValueError:
                continue
            etype_code = int(parts[1])
            if etype_code not in UNV_TO_MESHIO:
                # Skip unknown types gracefully
                # Consume until terminator for this element
                while True:
                    try:
                        nxt = next(it).strip()
                    except StopIteration:
                        break
                    if not nxt or nxt.startswith("-1"):
                        break
                continue
            # UNV etype_code mapping is complex; infer by connectivity length
            if len(parts) >= 7:
                conn = [int(p) for p in parts[6:]]
            else:
                conn = []
            # For round-trip, expect nodes on same line as header written by our writer
            ctype, expected = UNV_TO_MESHIO[etype_code]
            conn = conn[:expected]
            cells_by_type.setdefault(ctype, []).append(conn)

    for raw in it:
        s = raw.strip()
        if not s:
            continue
        if s == "-1":
            # separator
            continue
        if s.isdigit():
            sec = int(s)
            if sec == 2411:
                _read_nodes_block()
            elif sec == 2412:
                _read_elements_block()
            else:
                # skip other sections
                for raw2 in it:
                    if raw2.strip().startswith("-1"):
                        break

    # Build mesh
    sorted_node_ids = sorted(nodes)
    id2idx = {nid: i for i, nid in enumerate(sorted_node_ids)}
    points = np.array([nodes[nid] for nid in sorted_node_ids], dtype=float)

    cells = []
    for ctype, conn_list in cells_by_type.items():
        data = np.array([[id2idx[n] for n in conn] for conn in conn_list], dtype=int)
        cells.append((ctype, data))

    return meshio.Mesh(points=points, cells=cells, point_data={"id": np.array(sorted_node_ids, int)})


def write_unv(path: str | Path, mesh: meshio.Mesh) -> None:
    """Write a minimal UNV mesh using sections 2411 and 2412."""
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

    # 2411 nodes
    out.append("    -1\n")
    out.append("  2411\n")
    for i, (x, y, z) in enumerate(np.asarray(mesh.points, float)):
        nid = int(point_ids[i]) if point_ids is not None else i + 1
        out.append(f"{nid:10d}{x:16.5e}{y:16.5e}{z:16.5e}\n")
    out.append("    -1\n")

    # 2412 elements
    out.append("    -1\n")
    out.append("  2412\n")
    eid_counter = 1
    for ctype, conn in cells:
        # Map meshio type to UNV type code
        if ctype not in MESHIO_TO_UNV:
            raise ValueError(f"Unsupported cell type for UNV writer: {ctype}")
        etype, expected = MESHIO_TO_UNV[ctype]
        for e in conn:
            eid = eid_counter
            eid_counter += 1
            # Header-like line: id, type, mat, color, something, then nodelist
            out.append(f"{eid:10d}{etype:10d}{0:10d}{0:10d}{0:10d}{0:10d}")
            # node IDs
            for n in e[:expected]:
                nid = int(point_ids[int(n)]) if point_ids is not None else int(n) + 1
                out.append(f"{nid:10d}")
            out.append("\n")
    out.append("    -1\n")

    path.write_text("".join(out), encoding="utf-8")


def read_unv_post(path: str | Path) -> List[dict]:
    """Read UNV post data sections 55 (nodes) and 56 (elements) per step."""
    path = Path(path)
    with path.open() as f:
        lines = f.readlines()

    it = iter(lines)
    steps: List[dict] = []

    def _read_data_block() -> Dict[int, Dict[str, np.ndarray | float]]:
        data: Dict[int, Dict[str, np.ndarray | float]] = {}
        # After header, values lines: id then values
        for raw in it:
            s = raw.strip()
            if not s:
                continue
            if s.startswith("-1"):
                break
            parts = s.split()
            # Skip lines that don't start with integer id
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
        if s == "-1":
            continue
        if s.isdigit():
            sec = int(s)
            if sec in (55, 56):
                # Read name line
                name_line = next(it).strip()
                # Title line and 4 blank/title lines
                _ = next(it)
                _ = next(it)
                _ = next(it)
                _ = next(it)
                # Parameters line
                _ = next(it)
                # Sub/step line
                sub_line = next(it).strip()
                # Time line
                time_line = next(it).strip()
                # Parse sub/step
                sp = sub_line.split()
                sub_step = int(sp[-2])
                step = int(sp[-1])
                time = float(time_line)
                # Now data till -1
                if sec == 56:
                    elems = _read_data_block()
                    # Expect next section 55
                    # Accumulate step record
                    step_rec = {"step": step, "substep": sub_step, "time": time, "elements": elems, "nodes": {}}
                    steps.append(step_rec)
                elif sec == 55:
                    nodes = _read_data_block()
                    if not steps or steps[-1]["step"] != step:
                        steps.append({"step": step, "substep": sub_step, "time": time, "elements": {}, "nodes": nodes})
                    else:
                        steps[-1]["nodes"] = nodes
            else:
                # Skip other sections
                for raw2 in it:
                    if raw2.strip().startswith("-1"):
                        break

    return steps


def write_unv_post(path: str | Path, steps: List[dict], mode: str = "vector+scalar", name: str = "Result") -> None:
    """Write UNV post data sections 56 (element) and 55 (node) per step following given header format."""
    path = Path(path)
    out: List[str] = []

    def term():
        out.append("    -1\n")

    def fmt_e(v: float) -> str:
        return f"{v:13.5e}"

    for st in steps:
        step = int(st.get("step", 1))
        sub = int(st.get("substep", 1))
        time = float(st.get("time", 0.0))
        # Section 56: element data
        term()
        out.append("    56\n")
        out.append(f"{name}\nElement Data \n\n\n\n")
        out.append(f"         1         4         3         8         2  {6:8d}\n")
        out.append(f"         2         1  {sub:8d}  {step:8d}\n")
        out.append(f"{time:13.5e}\n")
        # Values
        elems = st.get("elements", {})
        for eid in sorted(elems):
            rec = elems[eid]
            vx, vy, vz = np.asarray(rec.get("vector", [0.0, 0.0, 0.0]), float).tolist()
            val = float(rec.get("value", 0.0))
            if mode == "scalar":
                out.append(f"{eid:8d} {fmt_e(val)}\n")
            elif mode == "vector":
                out.append(f"{eid:8d} {fmt_e(vx)} {fmt_e(vy)} {fmt_e(vz)}\n")
            else:
                out.append(f"{eid:8d} {fmt_e(vx)} {fmt_e(vy)} {fmt_e(vz)} {fmt_e(val)}\n")
        term()
        # Section 55: node data
        term()
        out.append("    55\n")
        out.append(f"{name}\nNode Data\n\n\n\n")
        out.append(f"         1         4         3         8         2  {6:8d}\n")
        out.append(f"         2         1  {sub:8d}  {step:8d}\n")
        out.append(f"{time:13.5e}\n")
        nodes = st.get("nodes", {})
        for nid in sorted(nodes):
            rec = nodes[nid]
            vx, vy, vz = np.asarray(rec.get("vector", [0.0, 0.0, 0.0]), float).tolist()
            val = float(rec.get("value", 0.0))
            if mode == "scalar":
                out.append(f"{nid:8d} {fmt_e(val)}\n")
            elif mode == "vector":
                out.append(f"{nid:8d} {fmt_e(vx)} {fmt_e(vy)} {fmt_e(vz)}\n")
            else:
                out.append(f"{nid:8d} {fmt_e(vx)} {fmt_e(vy)} {fmt_e(vz)} {fmt_e(val)}\n")
        term()

    path.write_text("".join(out), encoding="utf-8")
