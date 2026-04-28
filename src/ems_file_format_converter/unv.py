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
    elem_ids_by_type: Dict[str, List[int]] = {}
    elem_props1_by_type: Dict[str, List[int]] = {}
    elem_props2_by_type: Dict[str, List[int]] = {}

    def _read_nodes_block():
        # 2411: first line "id 0 0 15" (metadata ignored), second line has x y z
        for raw in it:
            s = raw.strip()
            if s.startswith("-1"):
                break
            if not s:
                continue
            parts = s.split()
            # Expect first token to be node id
            try:
                nid = int(parts[0])
            except ValueError:
                continue
            # Read the next line for coordinates
            coord_line = next(it, "").strip()
            if not coord_line:
                # Skip until we find a non-empty coordinate line or terminator
                while coord_line == "":
                    coord_line = next(it, "").strip()
                    if coord_line.startswith("-1"):
                        break
            if coord_line and not coord_line.startswith("-1"):
                try:
                    x, y, z = map(float, coord_line.split()[:3])
                    nodes[nid] = (x, y, z)
                except Exception:
                    # If parsing fails, ignore this node entry
                    pass

    def _read_elements_block():
        # 2412: header line then one or more lines of node ids (8 per line)
        for raw in it:
            s = raw.strip()
            if s.startswith("-1"):
                break
            if not s:
                continue
            parts = s.split()
            # Header: eid, etype, prop1, prop2, 15, ncount
            try:
                eid = int(parts[0])
                etype_code = int(parts[1])
                prop1 = int(parts[2])
                prop2 = int(parts[3])
                ncount = int(parts[5])
            except Exception:
                # Not a valid header; skip line
                continue
            if etype_code not in UNV_TO_MESHIO:
                # Skip unknown types; consume node lines accordingly
                # Read node ids lines until count satisfied or terminator
                remaining = ncount
                while remaining > 0:
                    nxt = next(it, "")
                    if not nxt:
                        break
                    t = nxt.strip()
                    if t.startswith("-1"):
                        break
                    nums = t.split()
                    remaining -= len(nums)
                continue
            ctype, expected = UNV_TO_MESHIO[etype_code]
            # Read node ids across as many lines as necessary (8 per line typical)
            conn_ids: List[int] = []
            remaining = ncount
            while remaining > 0:
                nxt = next(it, "")
                if not nxt:
                    break
                t = nxt.strip()
                if t.startswith("-1"):
                    break
                nums = [int(x) for x in t.split()]
                conn_ids.extend(nums)
                remaining -= len(nums)
            # Truncate or validate length against expected
            conn_ids = conn_ids[:expected]
            cells_by_type.setdefault(ctype, []).append(conn_ids)
            elem_ids_by_type.setdefault(ctype, []).append(eid)
            elem_props1_by_type.setdefault(ctype, []).append(prop1)
            elem_props2_by_type.setdefault(ctype, []).append(prop2)

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
    cell_data: Dict[str, List[np.ndarray]] = {"id": [], "prop1": [], "prop2": []}
    for ctype, conn_list in cells_by_type.items():
        # Map connectivity node IDs to zero-based indices; skip elements referencing missing nodes
        mapped_conns: List[List[int]] = []
        ids_kept: List[int] = []
        props1_kept: List[int] = []
        props2_kept: List[int] = []
        for idx, conn in enumerate(conn_list):
            try:
                mapped = [id2idx[n] for n in conn]
            except KeyError:
                # Skip this element if any node id is missing
                continue
            mapped_conns.append(mapped)
            # Keep aligned metadata when element is kept
            if ctype in elem_ids_by_type and idx < len(elem_ids_by_type[ctype]):
                ids_kept.append(int(elem_ids_by_type[ctype][idx]))
            if ctype in elem_props1_by_type and idx < len(elem_props1_by_type[ctype]):
                props1_kept.append(int(elem_props1_by_type[ctype][idx]))
            if ctype in elem_props2_by_type and idx < len(elem_props2_by_type[ctype]):
                props2_kept.append(int(elem_props2_by_type[ctype][idx]))
        if not mapped_conns:
            continue
        data = np.asarray(mapped_conns, dtype=int)
        cells.append((ctype, data))
        cell_data["id"].append(np.asarray(ids_kept, dtype=int))
        cell_data["prop1"].append(np.asarray(props1_kept, dtype=int))
        cell_data["prop2"].append(np.asarray(props2_kept, dtype=int))

    return meshio.Mesh(
        points=points, cells=cells, point_data={"id": np.array(sorted_node_ids, int)}, cell_data=cell_data
    )


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
    # Element IDs per block (optional)
    id_blocks: List[np.ndarray] = []
    prop1_blocks: List[np.ndarray] = []
    prop2_blocks: List[np.ndarray] = []
    if isinstance(getattr(mesh, "cell_data", None), dict) and "id" in mesh.cell_data:
        id_blocks = [np.asarray(a, dtype=int) for a in mesh.cell_data["id"]]
    if isinstance(getattr(mesh, "cell_data", None), dict) and "prop1" in mesh.cell_data:
        prop1_blocks = [np.asarray(a, dtype=int) for a in mesh.cell_data["prop1"]]
    if isinstance(getattr(mesh, "cell_data", None), dict) and "prop2" in mesh.cell_data:
        prop2_blocks = [np.asarray(a, dtype=int) for a in mesh.cell_data["prop2"]]
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
        out.append(f"{nid:10d}         0         0        15\n")
        out.append(f"{x:25.12e}{y:25.12e}{z:25.12e}\n")
    out.append("    -1\n")

    # 2412 elements
    out.append("    -1\n")
    out.append("  2412\n")
    eid_counter = 1
    for bidx, (ctype, conn) in enumerate(cells):
        # Map meshio type to UNV type code
        if ctype not in MESHIO_TO_UNV:
            raise ValueError(f"Unsupported cell type for UNV writer: {ctype}")
        etype, expected = MESHIO_TO_UNV[ctype]
        for eidx, e in enumerate(conn):
            eid = eid_counter
            eid_counter += 1
            if id_blocks and bidx < len(id_blocks):
                blk = id_blocks[bidx]
                if blk is not None and blk.size == len(conn):
                    eid = int(blk[eidx])
            # Properties: default to 1 unless provided per block
            prop1 = 1
            prop2 = 1
            if prop1_blocks and bidx < len(prop1_blocks):
                p1b = prop1_blocks[bidx]
                if p1b is not None and p1b.size == len(conn):
                    prop1 = int(p1b[eidx])
            if prop2_blocks and bidx < len(prop2_blocks):
                p2b = prop2_blocks[bidx]
                if p2b is not None and p2b.size == len(conn):
                    prop2 = int(p2b[eidx])
            node_ids = []
            for n in e[:expected]:
                nid = int(point_ids[int(n)]) if point_ids is not None else int(n) + 1
                node_ids.append(nid)
            # Header line: eid, etype, prop1, prop2, 15, count
            out.append(f"{eid:10d}{etype:10d}{prop1:10d}{prop2:10d}{15:10d}{len(node_ids):10d}\n")
            # Node ids lines: 8 per line, I10 formatting
            for i in range(0, len(node_ids), 8):
                chunk = node_ids[i : i + 8]
                out.append("".join(f"{nid:10d}" for nid in chunk) + "\n")
    out.append("    -1\n")

    path.write_text("".join(out), encoding="utf-8")


def read_unv_post(path: str | Path) -> List[dict]:
    """Read UNV post data with headers for sections 56 (elements) and 55 (nodes).

    Each data entry uses the two-line block format:
      - Line 1: "<ID> <N>" (ID is node/element id; N is data-per-line, typically 6 or 8)
      - Line 2: six reals: x, y, z, |vec|, extra1, extra2
      - If N > 6: read a third line with two more reals
    """
    path = Path(path)
    with path.open() as f:
        lines = [ln.rstrip() for ln in f]

    it = iter(lines)
    steps: List[dict] = []

    def _read_entries() -> Dict[int, Dict[str, np.ndarray | float]]:
        entries: Dict[int, Dict[str, np.ndarray | float]] = {}
        # Read until terminator "-1" or next section header
        for raw in it:
            s = raw.strip()
            if not s:
                continue
            if s.startswith("-1"):
                break
            parts = s.split()
            if len(parts) < 2:
                # not a data header line
                continue
            try:
                idv = int(parts[0])
                nvals = int(parts[1])
            except ValueError:
                # not a data header line
                continue
            # values line (six reals)
            vals_line = next(it, "").strip()
            vals: List[float] = []
            if vals_line:
                vals.extend(float(x) for x in vals_line.split())
            # optional extra line for N>6
            if nvals > 6:
                extra_line = next(it, "").strip()
                if extra_line:
                    vals.extend(float(x) for x in extra_line.split())
            # normalize
            vals = vals + [0.0] * max(0, 4 - len(vals))
            vx = float(vals[0]) if len(vals) > 0 else 0.0
            vy = float(vals[1]) if len(vals) > 1 else 0.0
            vz = float(vals[2]) if len(vals) > 2 else 0.0
            absval = float(vals[3]) if len(vals) > 3 else 0.0
            entries[idv] = {"vector": np.array([vx, vy, vz], float), "value": absval}
        return entries

    while True:
        try:
            raw = next(it)
        except StopIteration:
            break
        s = raw.strip()
        if not s:
            continue
        if s == "-1":
            # section separator
            continue
        if s.isdigit():
            sec = int(s)
            if sec in (56, 55):
                # name/title
                name_line = next(it, "").strip()
                # title line and padding
                _ = next(it, "")
                _ = next(it, "")
                _ = next(it, "")
                _ = next(it, "")
                # parameters line (contains data-per-line as last integer)
                param_line = next(it, "").strip()
                # sub/step line
                sub_line = next(it, "").strip()
                # time line
                time_line = next(it, "").strip()
                # parse sub/step/time
                try:
                    sp = sub_line.split()
                    sub_step = int(sp[-2])
                    step = int(sp[-1])
                except Exception:
                    sub_step, step = 1, 1
                try:
                    time = float(time_line)
                except Exception:
                    time = 0.0
                entries = _read_entries()
                if sec == 56:
                    steps.append(
                        {
                            "step": step,
                            "substep": sub_step,
                            "time": time,
                            "elements": entries,
                            "nodes": {},
                            "name56": name_line,
                        }
                    )
                else:
                    # 55
                    if not steps or steps[-1]["step"] != step:
                        steps.append(
                            {
                                "step": step,
                                "substep": sub_step,
                                "time": time,
                                "elements": {},
                                "nodes": entries,
                                "name55": name_line,
                            }
                        )
                    else:
                        steps[-1]["nodes"] = entries
                        steps[-1]["name55"] = name_line
            else:
                # skip other sections
                continue

    return steps


def write_unv_post(
    path: str | Path,
    steps: List[dict],
    data_per_line: int = 6,
    mode: str | None = None,
    name: str | None = None,
) -> None:
    """Write simplified UNV post data in two-line block format.

    For each entry (nodes preferred), write:
      Line 1: "<ID> <N>" where N = data_per_line (6 default; if 8, spill 2 to next line)
      Line 2: six reals: x, y, z, |vec|, extra1, extra2 (unused are 0)
      If N == 8: write remaining two reals on the next line (from rec["extra"], or 0s).
    """
    path = Path(path)
    out: List[str] = []

    def fmt(v: float) -> str:
        # Fixed-width 13.5e without trimming to match C's %13.5e
        return f"{v:13.5e}"

    # Choose nodes if present, else elements
    entries: Dict[int, Dict[str, np.ndarray | float]] = {}
    if steps:
        st = steps[0]
        nodes = st.get("nodes", {})
        elems = st.get("elements", {})
        entries = nodes if nodes else elems

    # Write section 56 (elements) header, then entries; then section 55 (nodes) with same style
    title_name = name or "Result"
    for st in steps or [{}]:
        # Section 56 (elements)
        out.append("    -1\n")
        out.append("    56\n")
        out.append(f"{(st.get('name56') or title_name)}\n")
        out.append("Element Data \n\n\n\n")
        out.append(f"         1         4         3         8         2  {data_per_line:8d}\n")
        step = int(st.get("step", 1))
        sub = int(st.get("substep", 1))
        out.append(f"         2         1  {sub:8d}  {step:8d}\n")
        time = float(st.get("time", 0.0))
        out.append(f"{time:13.5e}\n")
        elems = st.get("elements", {})
        for idv in sorted(elems):
            rec = elems[idv]
            vx, vy, vz = np.asarray(rec.get("vector", [0.0, 0.0, 0.0]), float).tolist()
            absval = float(rec.get("value", 0.0))
            extra = rec.get("extra", [])
            vals6: List[float] = [vx, vy, vz, absval]
            e1 = float(extra[0]) if isinstance(extra, (list, tuple)) and len(extra) > 0 else 0.0
            e2 = float(extra[1]) if isinstance(extra, (list, tuple)) and len(extra) > 1 else 0.0
            vals6.extend([e1, e2])
            out.append(f"{idv:10d}{data_per_line:10d}\n")
            out.append("".join(fmt(v) for v in vals6) + "\n")
            if data_per_line >= 8:
                e3 = float(extra[2]) if isinstance(extra, (list, tuple)) and len(extra) > 2 else 0.0
                e4 = float(extra[3]) if isinstance(extra, (list, tuple)) and len(extra) > 3 else 0.0
                out.append("".join(fmt(v) for v in [e3, e4]) + "\n")
        out.append("    -1\n")

        # Section 55 (nodes)
        out.append("    -1\n")
        out.append("    55\n")
        out.append(f"{(st.get('name55') or title_name)}\n")
        out.append("Node Data\n\n\n\n")
        out.append(f"         1         4         3         8         2  {data_per_line:8d}\n")
        out.append(f"         2         1  {sub:8d}  {step:8d}\n")
        out.append(f"{time:13.5e}\n")
        nodes = st.get("nodes", {})
        for idv in sorted(nodes):
            rec = nodes[idv]
            vx, vy, vz = np.asarray(rec.get("vector", [0.0, 0.0, 0.0]), float).tolist()
            absval = float(rec.get("value", 0.0))
            extra = rec.get("extra", [])
            vals6: List[float] = [vx, vy, vz, absval]
            e1 = float(extra[0]) if isinstance(extra, (list, tuple)) and len(extra) > 0 else 0.0
            e2 = float(extra[1]) if isinstance(extra, (list, tuple)) and len(extra) > 1 else 0.0
            vals6.extend([e1, e2])
            out.append(f"{idv:10d}{data_per_line:10d}\n")
            out.append("".join(fmt(v) for v in vals6) + "\n")
            if data_per_line >= 8:
                e3 = float(extra[2]) if isinstance(extra, (list, tuple)) and len(extra) > 2 else 0.0
                e4 = float(extra[3]) if isinstance(extra, (list, tuple)) and len(extra) > 3 else 0.0
                out.append("".join(fmt(v) for v in [e3, e4]) + "\n")
        out.append("    -1\n")

    path.write_text("".join(out), encoding="utf-8")
