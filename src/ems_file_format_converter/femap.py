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
    # Femap element names to meshio types (including higher-order where applicable)
    "POINT": ("vertex", 1),
    "LINE": ("line", 2),
    "TRIA": ("triangle", 3),
    "TRIA6": ("triangle6", 6),
    "QUAD": ("quad", 4),
    "QUAD8": ("quad8", 8),
    "TETRA": ("tetra", 4),
    "TETRA10": ("tetra10", 10),
    "PYRAMID": ("pyramid", 5),
    "PYRAMID13": ("pyramid13", 13),
    "WEDGE": ("wedge", 6),
    "WEDGE15": ("wedge15", 15),
    "BRICK": ("hexahedron", 8),
    "BRICK20": ("hexahedron20", 20),
}

MESHIO_TO_FEMAP: Dict[str, Tuple[str, int]] = {v[0]: (k, v[1]) for k, v in FEMAP_TO_MESHIO.items()}


def read_neu(path: str | Path) -> meshio.Mesh:
    """Read Femap Neutral (.neu) mesh supporting v4.1/v10.3 samples.

    Handles classic neutral sections:
      - 403: nodes (CSV: id, ..., x, y, z)
      - 404: elements (descriptor line with count, followed by connectivity line)
    """
    path = Path(path)
    with path.open(encoding="utf-8", errors="ignore") as f:
        lines = [ln.rstrip() for ln in f]

    nodes: Dict[int, Tuple[float, float, float]] = {}
    # Store per-type elements as tuples: (connectivity, matid, eid)
    cells_by_type: Dict[str, List[Tuple[List[int], int, int]]] = {}

    i = 0
    n = len(lines)

    def parse_csv_ints(s: str) -> List[int]:
        return [int(x) for x in s.replace(" ", "").split(",") if x and x.replace("-", "").isdigit()]

    def parse_csv_floats(s: str) -> List[float]:
        parts = [x for x in s.replace(" ", "").split(",") if x]
        vals: List[float] = []
        for x in parts:
            try:
                vals.append(float(x))
            except Exception:
                pass
        return vals

    # First, handle our simple "$ Nodes"/"$ Elements" format if present
    if any("$ Nodes" in ln for ln in lines) and any("$ Elements" in ln for ln in lines):
        mode = None

        def parse_node_line_ws(s: str) -> Tuple[int, float, float, float] | None:
            parts = s.strip().split()
            if len(parts) < 4:
                return None
            try:
                nid = int(parts[0])
                x, y, z = map(float, parts[1:4])
                return nid, x, y, z
            except Exception:
                return None

        def parse_elem_line_ws(s: str) -> Tuple[str, List[int], int, int] | None:
            parts = s.strip().split()
            if len(parts) < 3:
                return None
            try:
                eid = int(parts[0])
            except Exception:
                return None
            etok = parts[1].upper()
            if etok not in FEMAP_TO_MESHIO:
                return None
            ctype, expected = FEMAP_TO_MESHIO[etok]
            # collect integer tokens after the element token; some lines include property/material placeholders
            try:
                ints = [int(x) for x in parts[2:] if x.isdigit()]
                # take last expected integers as node ids
                nodes_list = ints[-expected:] if len(ints) >= expected else ints
            except Exception:
                return None
            if len(nodes_list) != expected:
                return None
            # simplified format doesn't carry matid; set to 0
            return ctype, nodes_list, 0, eid

        for ln in lines:
            s = ln.strip()
            if not s:
                continue
            if s.startswith("$"):
                if "Nodes" in s:
                    mode = "nodes"
                elif "Elements" in s:
                    mode = "elements"
                continue
            if mode == "nodes":
                p = parse_node_line_ws(s)
                if p:
                    nid, x, y, z = p
                    nodes[nid] = (x, y, z)
            elif mode == "elements":
                p = parse_elem_line_ws(s)
                if p:
                    ctype, conn, matid, eid = p
                    cells_by_type.setdefault(ctype, []).append((conn, matid, eid))
    else:
        # Fallback: classic neutral sections 403/404
        while i < n:
            s = lines[i].strip()
            i += 1
            if not s:
                continue
            if s == "-1":
                # next line may be section id
                if i < n and lines[i].strip().isdigit():
                    sec = int(lines[i].strip())
                    i += 1
                    if sec == 403:
                        # nodes until next -1
                        while i < n:
                            t = lines[i].strip()
                            if t == "-1":
                                i += 1
                                break
                            # CSV record: id,..., x, y, z,
                            ints = parse_csv_ints(t)
                            floats = parse_csv_floats(t)
                            if ints:
                                nid = ints[0]
                                if len(floats) >= 3:
                                    x, y, z = floats[-3:]
                                    nodes[nid] = (x, y, z)
                            i += 1
                    elif sec == 404:
                        # elements: descriptor + connectivity, repeated until next -1
                        while i < n:
                            t = lines[i].strip()
                            if t == "-1":
                                i += 1
                                break
                            desc = t
                            desc_ints = parse_csv_ints(desc)
                            if len(desc_ints) < 5:
                                i += 1
                                continue
                            # descriptor fields: [eid,124,matid,femap_type,topology]
                            eid = desc_ints[0]
                            matid = desc_ints[2]
                            topology = desc_ints[4]
                            # read two connectivity lines (up to 20 ints)
                            if i + 2 > n:
                                break
                            conn_line1 = lines[i + 1].strip()
                            conn_line2 = lines[i + 2].strip() if (i + 2) < n else ""
                            nodes20 = parse_csv_ints(conn_line1) + parse_csv_ints(conn_line2)
                            nodes20 = [x for x in nodes20 if x != 0]
                            # Determine meshio cell type and node reorder based on topology
                            ctype = None
                            conn: List[int] = []
                            if topology == 0:
                                ctype = "line"
                                conn = nodes20[:2]
                            elif topology == 9:
                                ctype = "vertex"
                                conn = nodes20[:1]
                            elif topology == 2:
                                ctype = "triangle"
                                conn = nodes20[:3]
                            elif topology == 3:
                                # Tria6: reorder per C implementation
                                ctype = "triangle6"
                                if len(nodes20) >= 6:
                                    n = nodes20[:6]
                                    conn = [n[0], n[4], n[1], n[5], n[2], n[6 - 1]]  # n[6-1] == n[5]
                                else:
                                    conn = nodes20[:6]
                            elif topology == 4:
                                ctype = "quad"
                                conn = nodes20[:4]
                            elif topology == 5:
                                # Quad8: reorder per C implementation
                                ctype = "quad8"
                                if len(nodes20) >= 8:
                                    n = nodes20[:8]
                                    conn = [n[0], n[4], n[1], n[5], n[2], n[6], n[3], n[7]]
                                else:
                                    conn = nodes20[:8]
                            elif topology == 6:
                                ctype = "tetra"
                                conn = nodes20[:4]
                            elif topology == 10:
                                ctype = "tetra10"
                                conn = nodes20[:10]
                            elif topology == 7:
                                ctype = "wedge"
                                conn = nodes20[:6]
                            elif topology == 11:
                                # prism13 (wedge15 is typical in meshio; use wedge15 if 15, else fallback)
                                if len(nodes20) >= 15:
                                    ctype = "wedge15"
                                    conn = nodes20[:15]
                                else:
                                    ctype = "wedge"
                                    conn = nodes20[:6]
                            elif topology == 14:
                                ctype = "pyramid"
                                conn = nodes20[:5]
                            elif topology == 19:
                                ctype = "pyramid13"
                                conn = nodes20[:13]
                            elif topology == 8:
                                ctype = "hexahedron"
                                conn = nodes20[:8]
                            elif topology == 12:
                                ctype = "hexahedron20"
                                conn = nodes20[:20]
                            else:
                                # Fallback by length
                                if len(nodes20) == 3:
                                    ctype = "triangle"
                                    conn = nodes20[:3]
                                elif len(nodes20) == 4:
                                    ctype = "quad"
                                    conn = nodes20[:4]
                                elif len(nodes20) == 8:
                                    ctype = "hexahedron"
                                    conn = nodes20[:8]
                            if ctype and conn and len(conn) >= 1:
                                cells_by_type.setdefault(ctype, []).append((conn, matid, eid))
                            # advance past descriptor + two connectivity lines and skip aux lines until next -1 or data block end
                            i += 3
                            # Skip auxiliary lines (properties/zeros/float triplets)
                            # Writer emits 4 aux lines; skip up to 4 here
                            skip = 4
                            while skip > 0 and i < n:
                                if lines[i].strip() == "-1":
                                    break
                                i += 1
                                skip -= 1
                    else:
                        # other sections: fast-forward until next -1
                        while i < n and lines[i].strip() != "-1":
                            i += 1
            continue

    # Build mesh
    sorted_node_ids = sorted(nodes)
    id2idx = {nid: i for i, nid in enumerate(sorted_node_ids)}
    points = np.array([nodes[nid] for nid in sorted_node_ids], dtype=float)

    cells = []
    cell_data: Dict[str, List[np.ndarray]] = {}
    for ctype, conn_list in cells_by_type.items():
        # Keep only connectivity matching expected node count for this ctype
        expected = MESHIO_TO_FEMAP.get(ctype, (None, None))[1]
        conns_all = [conn for (conn, _mat, _eid) in conn_list]
        conns = [conn for conn in conns_all if expected is None or len(conn) == expected]
        mats = [int(_mat) for (_conn, _mat, _eid) in conn_list]
        eids = [int(_eid) for (_conn, _mat, _eid) in conn_list]
        data = np.array([[id2idx.get(n, 0) for n in conn] for conn in conns], dtype=int)
        cells.append((ctype, data))
        # Align cell_data lengths with filtered connectivity
        keep_idx = [i for i, conn in enumerate(conns_all) if expected is None or len(conn) == expected]
        cell_data.setdefault("matid", []).append(np.array([mats[i] for i in keep_idx], dtype=int))
        cell_data.setdefault("eid", []).append(np.array([eids[i] for i in keep_idx], dtype=int))

    return meshio.Mesh(
        points=points, cells=cells, point_data={"id": np.array(sorted_node_ids, int)}, cell_data=cell_data
    )

    # Build mesh
    sorted_node_ids = sorted(nodes)
    id2idx = {nid: i for i, nid in enumerate(sorted_node_ids)}
    points = np.array([nodes[nid] for nid in sorted_node_ids], dtype=float)

    cells = []
    for ctype, conn_list in cells_by_type.items():
        data = np.array([[id2idx[n] for n in conn] for conn in conn_list], dtype=int)
        cells.append((ctype, data))

    return meshio.Mesh(points=points, cells=cells, point_data={"id": np.array(sorted_node_ids, int)})


def file_header(out: List[str]) -> List[str]:
    # 403: Nodes (CSV style like samples)
    out.append("   -1\n")
    out.append("   100\n")
    out.append("<NULL>\n")
    out.append("4.41,\n")
    out.append("   -1\n")

    return out


def write_neu(path: str | Path, mesh: meshio.Mesh, version: str = "4.41") -> None:
    """Write Femap Neutral (.neu) in classic sections 403/404 per FEMAP_io.c."""
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

    # Helper: mapping meshio type -> (topology, femap_type)
    def _map_type(ctype: str) -> Tuple[int, int]:
        topo_map = {
            "line": 0,
            "triangle": 2,
            "triangle6": 3,
            "quad": 4,
            "quad8": 5,
            "tetra": 6,
            "tetra10": 10,
            "wedge": 7,
            "wedge15": 11,
            "pyramid": 14,
            "pyramid13": 19,
            "hexahedron": 8,
            "hexahedron20": 12,
            "vertex": 9,
        }
        femap_type_map = {
            0: 1,
            2: 17,
            4: 17,
            5: 18,
            6: 25,
            7: 25,
            14: 25,
            8: 25,
            9: 27,
            10: 26,
            11: 26,
            19: 26,
            12: 26,
        }
        topo = topo_map.get(ctype, None)
        if topo is None:
            raise ValueError(f"Unsupported cell type for Femap writer: {ctype}")
        femap_type = femap_type_map.get(topo, 0)
        return topo, femap_type

    out: List[str] = []

    # file header
    out = file_header(out)

    # 403: Nodes (CSV style like samples)
    out.append("   -1\n")
    out.append("   403\n")
    points = np.asarray(mesh.points, float)
    # Use provided point ids if available, else 1..N
    point_ids = None
    if isinstance(getattr(mesh, "point_data", None), dict) and "id" in mesh.point_data:
        pid = np.asarray(mesh.point_data["id"]).reshape(-1)
        if pid.size == len(points):
            point_ids = pid.astype(int)
    for i, (x, y, z) in enumerate(points):
        nid = int(point_ids[i]) if point_ids is not None else i + 1
        # CSV line: id,0,0,0,0, x, y, z,
        out.append(f"{nid},0,0,1,46,0,0,0,0,0,0,{x:.12e},{y:.12e},{z:.12e},\n")
    out.append("   -1\n")

    # 404: Elements (descriptor + up to two connectivity lines)
    out.append("   -1\n")
    out.append("   404\n")
    # Optional per-block element ids and material/property ids
    eid_blocks: List[np.ndarray] = []
    matid_blocks: List[np.ndarray] = []
    if isinstance(getattr(mesh, "cell_data", None), dict):
        if "eid" in mesh.cell_data:
            eid_blocks = [np.asarray(a, dtype=int) for a in mesh.cell_data.get("eid", [])]
        if "matid" in mesh.cell_data:
            matid_blocks = [np.asarray(a, dtype=int) for a in mesh.cell_data.get("matid", [])]

    eid_counter = 1
    for bidx, block in enumerate(cells):
        ctype, conn = block
        topo, femap_type = _map_type(ctype)
        conn_arr = np.asarray(conn, dtype=int)
        for eidx, e in enumerate(conn_arr):
            # element id
            eid = eid_counter
            if eid_blocks and bidx < len(eid_blocks):
                blk = eid_blocks[bidx]
                if blk is not None and blk.size == len(conn_arr):
                    eid = int(blk[eidx])
            eid_counter += 1
            # material/property id: use provided per-block data if present
            matid = 1
            if matid_blocks and bidx < len(matid_blocks):
                mblk = matid_blocks[bidx]
                if mblk is not None and mblk.size == len(conn_arr):
                    matid = int(mblk[eidx])
            # Map node indices to external ids if provided, else 1-based
            node_ids: List[int] = []
            for n in e:
                nid = int(point_ids[int(n)]) if point_ids is not None else int(n) + 1
                node_ids.append(nid)
            # Descriptor CSV: eid,124,matid,femap_type,topology,
            out.append(f"{eid},124,{matid},{femap_type},{topo},1,0,0,\n")
            # Connectivity: first up to 10, then remaining up to 10
            # Special formatting for tetra (topology 6): base triangle nodes, then 0, then apex
            if topo == 6 and len(node_ids) >= 4:
                base3 = node_ids[:3]
                apex = node_ids[3]
                fmt = base3 + [0, apex]
                n1 = fmt
                n2 = []
            # Special formatting for wedge (topology 7): lower tri (3), 0, upper tri (3), 0
            elif topo == 7 and len(node_ids) >= 6:
                lower3 = node_ids[:3]
                upper3 = node_ids[3:6]
                fmt = lower3 + [0] + upper3 + [0]
                n1 = fmt
                n2 = []
            else:
                n1 = node_ids[:10]
                n2 = node_ids[10:20]
            # Pad with zeros to indicate unused entries similar to reader filtering
            n1_pad = n1 + [0] * (10 - len(n1))
            n2_pad = n2 + [0] * (10 - len(n2))
            out.append(",".join(str(v) for v in n1_pad) + ",\n")
            out.append(",".join(str(v) for v in n2_pad) + ",\n")
            # Emit four auxiliary lines (zeros/float triplets) to match parser skipping
            out.append("0.,0.,0.,\n")
            out.append("0.,0.,0.,\n")
            out.append("0.,0.,0.,\n")
            out.append("0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,\n")
    out.append("   -1\n")

    # Write file
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(out), encoding="utf-8")


def read_neu_post(path: str | Path) -> List[dict]:
    """Read Femap Neutral post data (450 + 451) into ATLAS-like records.

    Accumulates per-component datasets inside each 451 section using temporary
    maps, then assigns per-id records as {'vector': [x,y,z], 'value': mag}.
    """
    path = Path(path)
    with path.open(encoding="utf-8", errors="ignore") as f:
        lines = [ln.rstrip() for ln in f]

    steps: List[dict] = []
    i = 0
    n = len(lines)

    def _new_step(step_num: int, time_val: float) -> dict:
        return {"step": step_num, "substep": 1, "time": time_val, "elements": {}, "nodes": {}}

    while i < n:
        s = lines[i].strip()
        i += 1
        if s == "-1" and i < n and lines[i].strip().isdigit():
            sec = int(lines[i].strip())
            i += 1
        elif s.isdigit():
            sec = int(s)
        else:
            continue

        if sec == 450:
            step_num = len(steps) + 1
            time_val = 0.0
            while i < n and lines[i].strip() != "-1":
                t = lines[i].strip()
                if "STEP:" in t and "Time:" in t:
                    try:
                        parts = t.replace(":", " ").split()
                        si = parts.index("STEP") + 1
                        ti = parts.index("Time") + 1
                        step_num = int(parts[si])
                        time_val = float(parts[ti])
                    except Exception:
                        pass
                i += 1
            if i < n and lines[i].strip() == "-1":
                i += 1
            steps.append(_new_step(step_num, time_val))
            continue

        if sec == 451 or sec == 1051:
            if not steps:
                steps.append(_new_step(1, 0.0))
            current = steps[-1]
            current_is_element: bool | None = None
            current_comp: int | None = None  # 0:X,1:Y,2:Z,3:|V|

            # Per-component accumulators for this section
            elems_comp: Dict[int, Dict[int, float]] = {0: {}, 1: {}, 2: {}, 3: {}}
            nodes_comp: Dict[int, Dict[int, float]] = {0: {}, 1: {}, 2: {}, 3: {}}
            acc: Dict[int, int, float] = {}  # dataset no, id, value

            # Read datasets until -1
            data_no = 0
            while i < n and lines[i].strip() != "-1":
                t = lines[i].strip()

                # check if element/node dataset
                parts = [x.strip() for x in t.split(",") if x.strip()]
                if int(parts[1]) > 60000:
                    current_is_element = True
                else:
                    current_is_element = False

                i += 1
                val_name = lines[i].strip()
                # Infer component from title suffix -1..-4
                title = val_name.lower()
                for k in (4, 3, 2, 1):
                    if title.endswith(f"-{k}"):
                        current_comp = 3 if k == 4 else (k - 1)
                        break
                i += 6  # skip 0,0,0,0,0,0,0,0,0,0,0, line and two float triplet lines

                t = lines[i].strip()
                parts = [x.strip() for x in t.split(",") if x.strip()]

                # 451 data record: id,value
                if sec == 451:
                    while parts[0] != "-1":
                        try:
                            rid = int(parts[0])
                            val = float(parts[1])
                            acc.setdefault(data_no, {})[rid] = val
                        except Exception:
                            pass
                        i += 1
                        t = lines[i].strip()
                        parts = [x.strip() for x in t.split(",") if x.strip()]
                    data_no += 1
                    i += 1

                # 1051 data record: start,end,values...
                elif sec == 1051:
                    while parts[0] != "-1":
                        start_id = int(parts[0])
                        end_id = int(parts[1])
                        values: List[float] = []
                        for vstr in parts[2:]:
                            try:
                                v = float(vstr)
                                values.append(v)
                            except Exception:
                                pass
                        # Read continuation lines until we have all values
                        while len(values) < (end_id - start_id + 1) and i + 1 < n:
                            i += 1
                            t = lines[i].strip()
                            parts = [x.strip() for x in t.split(",") if x.strip()]
                            for vstr in parts:
                                try:
                                    v = float(vstr)
                                    values.append(v)
                                except Exception:
                                    pass
                        # Assign values to rids in range
                        for idx, rid in enumerate(range(start_id, end_id + 1)):
                            if idx < len(values):
                                acc.setdefault(data_no, {})[rid] = values[idx]
                        i += 1
                        t = lines[i].strip()
                        parts = [x.strip() for x in t.split(",") if x.strip()]
                    data_no += 1
                    i += 1

                # Assign accumulated values to elements/nodes as arrays per rid
                # Build per-id arrays from acc: {data_no: {rid: val}}
                vals_by_rid: Dict[int, List[float]] = {}
                for no_map in acc.values():
                    for rid, val in no_map.items():
                        vals_by_rid.setdefault(rid, []).append(float(val))

                # Decide target based on current_is_element
                if current_is_element is True:
                    for rid, arr in vals_by_rid.items():
                        elems_comp[current_comp][rid] = np.asarray(arr, float)
                    acc = {}
                else:
                    for rid, arr in vals_by_rid.items():
                        nodes_comp[current_comp][rid] = np.asarray(arr, float)
                    acc = {}

            # Build per-id records
            def _last_scalar(v: float | np.ndarray) -> float:
                arr = np.asarray(v, float).reshape(-1)
                return float(arr[-1]) if arr.size > 0 else 0.0

            ids_e = set().union(*[set(d.keys()) for d in elems_comp.values()])
            for eid in sorted(ids_e):
                vec = [_last_scalar(elems_comp[c].get(eid, 0.0)) for c in (0, 1, 2)]
                mag = _last_scalar(elems_comp[3].get(eid, 0.0))
                if mag == 0.0 and any(abs(x) > 0.0 for x in vec):
                    mag = float(np.linalg.norm(np.asarray(vec, float)))
                current["elements"][eid] = {"vector": vec, "value": mag}

            ids_n = set().union(*[set(d.keys()) for d in nodes_comp.values()])
            for nid in sorted(ids_n):
                vec = [_last_scalar(nodes_comp[c].get(nid, 0.0)) for c in (0, 1, 2)]
                mag = _last_scalar(nodes_comp[3].get(nid, 0.0))
                if mag == 0.0 and any(abs(x) > 0.0 for x in vec):
                    mag = float(np.linalg.norm(np.asarray(vec, float)))
                current["nodes"][nid] = {"vector": vec, "value": mag}

            if i < n and lines[i].strip() == "-1":
                i += 1
            continue

        # Fast-forward other sections
        while i < n and lines[i].strip() != "-1":
            i += 1
        if i < n and lines[i].strip() == "-1":
            i += 1

    return steps


def write_neu_post(
    path: str | Path, steps: List[dict], mode: str | None = None, style: str = "451", title_prefix: str = "BMAG"
) -> None:
    """Write Femap Neutral post data using 450 + 451 (default) or 1051 sections.

    - 450: step header with STEP and Time lines
    - 451: datasets with per-line "id, value,"
    - 1051: datasets with range lines "start,end,values..."
    """
    path = Path(path)
    out: List[str] = []

    # file header
    out = file_header(out)

    def f13(v: float) -> str:
        return f"{v:13.5e}"

    def _contiguous_runs(sorted_ids: List[int]) -> List[Tuple[int, int, List[int]]]:
        if not sorted_ids:
            return []
        runs: List[Tuple[int, int, List[int]]] = []
        start = sorted_ids[0]
        prev = start
        buf: List[int] = [start]
        for k in sorted_ids[1:]:
            if k == prev + 1:
                buf.append(k)
                prev = k
            else:
                runs.append((start, prev, buf[:]))
                start = prev = k
                buf = [k]
        runs.append((start, prev, buf[:]))
        return runs

    def _emit_1051_values(start_id: int, end_id: int, vals: List[float]) -> List[str]:
        lines: List[str] = []
        idx = 0
        first = vals[idx : idx + 8]
        idx += len(first)
        lines.append(f"{start_id},{end_id}," + ",".join(f"{v:.5e}" for v in first) + "\n")
        while idx < len(vals):
            chunk = vals[idx : idx + 10]
            idx += len(chunk)
            lines.append(",".join(f"{v:.5e}" for v in chunk) + "\n")
        return lines

    for st in steps:
        step = int(st.get("step", 1))
        time = float(st.get("time", 0.0))
        # 450 header
        out.append("   -1\n")
        out.append("   450\n")
        out.append(f"{step},\n")
        out.append(f"STEP:{step} Time:{f13(time)}\n")
        out.append(f"0,3,\n{f13(time)},\n1,\n<NULL>\n")
        out.append("   -1\n")

        # Data section
        if style == "1051":
            out.append("   -1\n")
            out.append("  1051\n")
        else:
            out.append("   -1\n")
            out.append("   451\n")

        elems: Dict[int, Dict[str, np.ndarray | float]] = st.get("elements", {}) or {}
        nodes: Dict[int, Dict[str, np.ndarray | float]] = st.get("nodes", {}) or {}

        # Elements: DSIDs 60031..60034, titles BMAG-elem-1..4
        if elems:
            # Preserve original element ID order as provided in steps
            ids_sorted_e = list(elems.keys())
            first_e = elems[ids_sorted_e[0]]
            vec_e = np.asarray(first_e.get("vector", [0.0, 0.0, 0.0]), float).reshape(-1)
            has_vec_e = bool(np.any(vec_e))
            elem_titles = [f"{title_prefix}-elem-{i}" for i in range(1, 5)]
            elem_dsids = [60031, 60032, 60033, 60034]
            # Always emit four datasets (X,Y,Z,|V|) to match sample 451 layout
            datasets_e = [(elem_dsids[c], elem_titles[c], c) for c in range(4)]
            for dsid, title, comp in datasets_e:
                if style == "1051":
                    # Header
                    out.append(f"{step}, {dsid},1,\n")
                    out.append(f"{title}\n")
                    out.append("0.,-1.,0.,\n")
                    out.append(f"{dsid},0,0,0,0,0,0,0,0,0,\n")
                    out.append("0,0,0,0,0,0,0,0,0,0,\n")
                    out.append("0,0,3,8,\n")
                    out.append("0,1,1,\n")
                    # Data
                    for s_id, e_id, run_ids in _contiguous_runs(ids_sorted_e):
                        values: List[float] = []
                        for eid in run_ids:
                            rec = elems[eid]
                            if has_vec_e and comp in (0, 1, 2):
                                v = np.asarray(rec.get("vector", [0.0, 0.0, 0.0]), float)
                                val = float(v[comp]) if comp < len(v) else 0.0
                            elif has_vec_e and comp == 3:
                                v = np.asarray(rec.get("vector", [0.0, 0.0, 0.0]), float)
                                val = float(np.linalg.norm(v))
                            else:
                                val = float(rec.get("value", 0.0))
                            values.append(val)
                        out.extend(_emit_1051_values(s_id, e_id, values))
                    out.append("-1,0.,\n")
                else:
                    # 451 header
                    out.append(f"{step}, {dsid},1,\n")
                    out.append(f"{title}\n")
                    out.append("0.,-1.,0.,\n")
                    out.append(f"{dsid},0,0,0,0,0,0,0,0,0,\n")
                    out.append("0,0,0,0,0,0,0,0,0,0,\n")
                    out.append("0,0,3,8,\n")
                    out.append("0,1,1,\n")
                    for eid in ids_sorted_e:
                        rec = elems[eid]
                        if has_vec_e and comp in (0, 1, 2):
                            v = np.asarray(rec.get("vector", [0.0, 0.0, 0.0]), float)
                            val = float(v[comp]) if comp < len(v) else 0.0
                        elif has_vec_e and comp == 3:
                            v = np.asarray(rec.get("vector", [0.0, 0.0, 0.0]), float)
                            val = float(np.linalg.norm(v))
                        else:
                            val = float(rec.get("value", 0.0))
                        out.append(f"{eid}, {f13(val)},\n")
                    out.append("-1,0.,\n")

        # Nodes: DSIDs 31..34, titles BMAG-node-1..4
        if nodes:
            # Preserve original node ID order as provided in steps
            ids_sorted_n = list(nodes.keys())
            first_n = nodes[ids_sorted_n[0]]
            vec_n = np.asarray(first_n.get("vector", [0.0, 0.0, 0.0]), float).reshape(-1)
            has_vec_n = bool(np.any(vec_n))
            node_titles = [f"{title_prefix}-node-{i}" for i in range(1, 5)]
            node_dsids = [31, 32, 33, 34]
            # Always emit four datasets (X,Y,Z,|V|) to match sample 451 layout
            datasets_n = [(node_dsids[c], node_titles[c], c) for c in range(4)]
            for dsid, title, comp in datasets_n:
                if style == "1051":
                    # Header
                    out.append(f"{step}, {dsid},1,\n")
                    out.append(f"{title}\n")
                    out.append("0.,-1.,0.,\n")
                    out.append(f"{dsid},0,0,0,0,0,0,0,0,0,\n")
                    out.append("0,0,0,0,0,0,0,0,0,0,\n")
                    out.append("0,0,3,7,\n")
                    out.append("0,1,1,\n")
                    # Data
                    for s_id, e_id, run_ids in _contiguous_runs(ids_sorted_n):
                        values: List[float] = []
                        for nid in run_ids:
                            rec = nodes[nid]
                            if has_vec_n and comp in (0, 1, 2):
                                v = np.asarray(rec.get("vector", [0.0, 0.0, 0.0]), float)
                                val = float(v[comp]) if comp < len(v) else 0.0
                            elif has_vec_n and comp == 3:
                                v = np.asarray(rec.get("vector", [0.0, 0.0, 0.0]), float)
                                val = float(np.linalg.norm(v))
                            else:
                                val = float(rec.get("value", 0.0))
                            values.append(val)
                        out.extend(_emit_1051_values(s_id, e_id, values))
                    out.append("-1,0.,\n")
                else:
                    # 451 header
                    out.append(f"{step}, {dsid},1,\n")
                    out.append(f"{title}\n")
                    out.append("0.,-1.,0.,\n")
                    out.append(f"{dsid},0,0,0,0,0,0,0,0,0,\n")
                    out.append("0,0,0,0,0,0,0,0,0,0,\n")
                    out.append("0,0,3,7,\n")
                    out.append("0,1,1,\n")
                    for nid in ids_sorted_n:
                        rec = nodes[nid]
                        if has_vec_n and comp in (0, 1, 2):
                            v = np.asarray(rec.get("vector", [0.0, 0.0, 0.0]), float)
                            val = float(v[comp]) if comp < len(v) else 0.0
                        elif has_vec_n and comp == 3:
                            v = np.asarray(rec.get("vector", [0.0, 0.0, 0.0]), float)
                            val = float(np.linalg.norm(v))
                        else:
                            val = float(rec.get("value", 0.0))
                        out.append(f"{nid}, {f13(val)},\n")
                    out.append("-1,0.,\n")

        # End of dataset section
        out.append("   -1\n")

    # Ensure output directory exists before writing
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "".join(out)
    try:
        with path.open("w", encoding="utf-8", errors="ignore") as fh:
            fh.write(content)
    except Exception:
        # Fallback in case of environment-specific issues
        path.write_text(content, encoding="utf-8")
    # Final guard: ensure file exists for downstream reads
    if not path.exists():
        path.touch()
