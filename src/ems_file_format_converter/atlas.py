# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Science Solutions International Laboratory, Inc.
# emsol_mesh_convert/atlas.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import meshio

ATLAS_TO_MESHIO = {
    101: ("vertex", 1),
    102: ("line", 2),
    203: ("triangle", 3),
    204: ("quad", 4),
    208: ("quad8", 8),
    304: ("tetra", 4),
    305: ("pyramid", 5),
    306: ("wedge", 6),
    308: ("hexahedron", 8),
    310: ("tetra10", 10),
    313: ("pyramid13", 13),
    315: ("wedge15", 15),
    320: ("hexahedron20", 20),
}


def _read_nodes(lines: List[str]) -> Dict[int, Tuple[float, float, float]]:
    nodes: Dict[int, Tuple[float, float, float]] = {}
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith("-1"):
            break
        parts = line.split()
        nid = int(parts[0])
        x, y, z = map(float, parts[1:4])
        nodes[nid] = (x, y, z)
    return nodes


def _read_elements(
    lines: List[str],
) -> Tuple[
    Dict[str, List[List[int]]],
    Dict[str, List[int]],
    Dict[str, List[int]],
]:
    cells_by_type: Dict[str, List[List[int]]] = {}
    elem_ids_by_type: Dict[str, List[int]] = {}
    iprops_by_type: Dict[str, List[int]] = {}
    it = iter(lines)
    for raw in it:
        line = raw.strip()
        if not line:
            continue
        if line.startswith("-1"):
            break

        parts = line.split()
        ielem = int(parts[0])
        ietp = int(parts[1])
        iprop = int(parts[2])  # 物性番号（必要なら cell_data に入れる）

        try:
            cell_type, n_nodes = ATLAS_TO_MESHIO[ietp]
        except KeyError:
            raise ValueError(f"Unsupported ATLAS element type ietp={ietp}")

        conn = [int(x) for x in parts[3:]]

        # 1行に収まらず、節点が足りない場合は次行以降から補う
        while len(conn) < n_nodes:
            cont = next(it).strip()
            if cont.startswith("-1"):
                raise ValueError("Unexpected end of CONC section")
            conn.extend(int(x) for x in cont.split())

        cells_by_type.setdefault(cell_type, []).append(conn)
        elem_ids_by_type.setdefault(cell_type, []).append(ielem)
        iprops_by_type.setdefault(cell_type, []).append(iprop)

    return cells_by_type, elem_ids_by_type, iprops_by_type


def read_atlas(path: str | Path) -> meshio.Mesh:
    path = Path(path)
    with path.open() as f:
        lines = f.readlines()

    # セクションを探す
    node_lines: List[str] = []
    elem_lines: List[str] = []

    mode = None
    for line in lines:
        key = line.strip().split()
        if not key:
            continue
        if key[0] == "GRID":
            mode = "grid"
            continue
        if key[0] == "CONC":
            mode = "conc"
            continue

        if mode == "grid":
            node_lines.append(line)
        elif mode == "conc":
            elem_lines.append(line)

    nodes = _read_nodes(node_lines)
    cells_by_type, elem_ids_by_type, iprops_by_type = _read_elements(elem_lines)

    # ID → 0 始まりインデックス
    sorted_node_ids = sorted(nodes)
    id2idx = {nid: i for i, nid in enumerate(sorted_node_ids)}
    points = np.array([nodes[nid] for nid in sorted_node_ids], dtype=float)

    cells = []
    cell_data: Dict[str, List[np.ndarray]] = {"id": [], "iprop": []}
    for cell_type, conn_list in cells_by_type.items():
        data = np.array(
            [[id2idx[n] for n in conn] for conn in conn_list],
            dtype=int,
        )
        cells.append((cell_type, data))
        # element IDs and iprops aligned per type
        cell_data["id"].append(np.asarray(elem_ids_by_type[cell_type], dtype=int))
        cell_data["iprop"].append(np.asarray(iprops_by_type[cell_type], dtype=int))
    # store original node IDs in point_data
    point_data = {"id": np.asarray(sorted_node_ids, dtype=int)}

    mesh = meshio.Mesh(points=points, cells=cells, point_data=point_data, cell_data=cell_data)
    return mesh


# Inverse map: meshio cell type -> (ATLAS element type code, n_nodes)
MESHIO_TO_ATLAS: Dict[str, Tuple[int, int]] = {v[0]: (k, v[1]) for k, v in ATLAS_TO_MESHIO.items()}


def write_atlas(path: str | Path, mesh: meshio.Mesh, property_id: int = 1) -> None:
    path = Path(path)

    # Normalize cells to list of (type, data)
    cells: List[Tuple[str, np.ndarray]] = []
    for block in mesh.cells:
        try:
            ctype = block.type  # meshio CellBlock
            cdata = block.data
        except AttributeError:
            ctype, cdata = block  # tuple form
        cells.append((ctype, np.asarray(cdata, dtype=int)))

    # Prepare point IDs (original node ids) if provided
    point_ids: np.ndarray | None = None
    if isinstance(getattr(mesh, "point_data", None), dict) and "id" in mesh.point_data:
        pid = np.asarray(mesh.point_data["id"]).reshape(-1)
        if pid.size == len(mesh.points):
            point_ids = pid.astype(int)

    # Prepare cell_data for iprop/id if provided
    iprop_blocks: List[np.ndarray] = []
    id_blocks: List[np.ndarray] = []
    if isinstance(getattr(mesh, "cell_data", None), dict):
        if "iprop" in mesh.cell_data:
            iprop_blocks = [np.asarray(a, dtype=int) for a in mesh.cell_data["iprop"]]
        if "id" in mesh.cell_data:
            id_blocks = [np.asarray(a, dtype=int) for a in mesh.cell_data["id"]]

    lines: List[str] = []

    _write_nodes_section(lines, mesh, point_ids)
    _write_elements_section(lines, cells, point_ids, iprop_blocks, id_blocks, property_id)

    path.write_text("".join(lines), encoding="utf-8")


def _write_nodes_section(lines: List[str], mesh: meshio.Mesh, point_ids: np.ndarray | None) -> None:
    lines.append("GRID    (I8,3E14.6)\n")
    for i, (x, y, z) in enumerate(np.asarray(mesh.points, dtype=float)):
        nid = int(point_ids[i]) if point_ids is not None else i + 1
        # Fixed-width formatting: I8, 3E14.6
        lines.append(f"{nid:8d} {x:14.6e} {y:14.6e} {z:14.6e}\n")
    lines.append("      -1\n")


def _write_elements_section(
    lines: List[str],
    cells: List[Tuple[str, np.ndarray]],
    point_ids: np.ndarray | None,
    iprop_blocks: List[np.ndarray],
    id_blocks: List[np.ndarray],
    property_id: int,
) -> None:
    lines.append("CONC    (8I8)\n")
    elem_id_counter = 1
    for bidx, (cell_type, conn) in enumerate(cells):
        if cell_type not in MESHIO_TO_ATLAS:
            raise ValueError(f"Unsupported meshio cell type for ATLAS writer: {cell_type}")
        ietp, n_nodes = MESHIO_TO_ATLAS[cell_type]

        block_iprops: np.ndarray | None = None
        if iprop_blocks and bidx < len(iprop_blocks):
            block_iprops = iprop_blocks[bidx]
            if block_iprops is not None and block_iprops.size != len(conn):
                block_iprops = None

        block_ids: np.ndarray | None = None
        if id_blocks and bidx < len(id_blocks):
            block_ids = id_blocks[bidx]
            if block_ids is not None and block_ids.size != len(conn):
                block_ids = None

        for eidx, elem in enumerate(conn):
            iprop = property_id if block_iprops is None else int(block_iprops[eidx])

            if point_ids is not None:
                node_ids = [int(point_ids[int(n)]) for n in elem[:n_nodes]]
            else:
                node_ids = [int(n) + 1 for n in elem[:n_nodes]]

            eid = int(block_ids[eidx]) if block_ids is not None else elem_id_counter
            # First line: element header with fixed-width integers (I8)
            header = f"{eid:8d} {ietp:8d} {iprop:8d}"
            # Connectivity: write 8 node IDs per line, each I8
            if not node_ids:
                lines.append(header + "\n")
            else:
                # write first line header + up to 5 ids
                chunk = node_ids[:5]
                lines.append(header + " " + " ".join(f"{nid:8d}" for nid in chunk) + "\n")
                # remaining ids on subsequent lines (8 per line = eid+ietp+iprop + 5 ids)
                rest = node_ids[5:]
                for i in range(0, len(rest), 5):
                    part = rest[i : i + 5]
                    lines.append("" + " ".join(f"{nid:8d}" for nid in part) + "\n")

            elem_id_counter += 1
    lines.append("      -1\n")


# --- ATLAS Post data (STEP/EVAL/STRE) ---
def read_atlas_post(path: str | Path) -> List[dict]:
    path = Path(path)
    with path.open() as f:
        lines = f.readlines()

    it = iter(lines)
    steps: List[dict] = []

    def _read_block() -> List[str]:
        block: List[str] = []
        for raw in it:
            s = raw.strip()
            if not s:
                continue
            if s.startswith("-1"):
                break
            block.append(s)
        return block

    for raw in it:
        s = raw.strip()
        if not s:
            continue
        if s.startswith("STEP"):
            nxt = next(it).strip()
            p = nxt.split()
            step = int(p[0])
            sub = int(p[1])
            time = float(p[2])

            step_rec = {"step": step, "substep": sub, "time": time, "elements": {}, "nodes": {}}

            hdr = next(it).strip()
            if not hdr.startswith("EVAL"):
                raise ValueError("Expected EVAL section after STEP")
            for row in _read_block():
                parts = row.split()
                eid = int(parts[0])
                vals = [float(x) for x in parts[1:]]
                if len(vals) >= 4:
                    vec = vals[:3]
                    val = vals[-1]
                else:
                    vec = (vals + [0.0, 0.0, 0.0])[:3]
                    val = vals[-1] if vals else 0.0
                step_rec["elements"][eid] = {"vector": np.array(vec, float), "value": float(val)}

            hdr = next(it).strip()
            if not hdr.startswith("STRE"):
                raise ValueError("Expected STRE section after EVAL")
            for row in _read_block():
                parts = row.split()
                nid = int(parts[0])
                vals = [float(x) for x in parts[1:]]
                if len(vals) >= 4:
                    vec = vals[:3]
                    val = vals[-1]
                else:
                    vec = (vals + [0.0, 0.0, 0.0])[:3]
                    val = vals[-1] if vals else 0.0
                step_rec["nodes"][nid] = {"vector": np.array(vec, float), "value": float(val)}

            steps.append(step_rec)

    return steps


def write_atlas_post(path: str | Path, steps: List[dict], mode: str = "vector+scalar") -> None:
    path = Path(path)
    out: List[str] = []

    def fmt_e(v: float) -> str:
        return f"{v:14.5e}"

    for idx, st in enumerate(steps, start=1):
        step_no = int(st.get("step", idx))
        sub_no = int(st.get("substep", 1))
        time = float(st.get("time", 0.0))

        out.append("STEP    (2I5,E12.0)\n")
        out.append(f"{step_no:5d} {sub_no:5d} {time:12.3e}\n")

        out.append(f"EVAL   {idx}(I8,6E14.0)\n")
        elements = st.get("elements", {})
        for eid in sorted(elements):
            rec = elements[eid]
            vx, vy, vz = np.asarray(rec.get("vector", [0.0, 0.0, 0.0]), float).tolist()
            val = float(rec.get("value", 0.0))
            if mode == "scalar":
                out.append(f"{eid:8d} {fmt_e(val)}\n")
            elif mode == "vector":
                out.append(f"{eid:8d} {fmt_e(vx)} {fmt_e(vy)} {fmt_e(vz)}\n")
            else:  # vector+scalar
                out.append(f"{eid:8d} {fmt_e(vx)} {fmt_e(vy)} {fmt_e(vz)} {fmt_e(val)}\n")
        out.append("      -1\n")

        out.append(f"STRE   {idx}(I8,6E14.0)\n")
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
        out.append("      -1\n")

    path.write_text("".join(out), encoding="utf-8")
