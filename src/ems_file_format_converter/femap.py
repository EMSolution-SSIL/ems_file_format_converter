# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Science Solutions International Laboratory, Inc.
from __future__ import annotations

from pathlib import Path
import re
from time import monotonic
from typing import Dict, List, Tuple

import numpy as np
import meshio

from .post_components import get_component, max_component_index_in_record, record_from_components

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

TITLE_PREFIX_BY_NAME: Dict[str, str] = {
    "magnetic": "BMAG",
    "current": "CURR",
    "disp": "DISP",
    "electric": "ELEC",
    "surface_current": "SCUR",
    "force_j_b": "LFOR",
    "force": "NFOR",
    "heat": "HEAT",
    "magnet": "MAGNET",
    "iron_loss": "IRON_LOSS",
}


def infer_title_prefix_from_filename(path: str | Path, default: str = "BMAG") -> str:
    """Infer FEMAP post title prefix from a filename.

    Examples: magnetic_*.neu -> BMAG, current_*.neu -> CURR.
    Returns ``default`` when no known keyword is found.
    """
    stem = Path(path).stem.lower()
    normalized = re.sub(r"[^a-z0-9]+", "_", stem).strip("_")

    items = sorted(TITLE_PREFIX_BY_NAME.items(), key=lambda kv: len(kv[0]), reverse=True)
    for key, prefix in items:
        key_norm = re.sub(r"[^a-z0-9]+", "_", key.lower()).strip("_")
        if re.search(rf"(^|_){re.escape(key_norm)}($|_)", normalized):
            return prefix

    for key, prefix in items:
        key_norm = re.sub(r"[^a-z0-9]+", "_", key.lower()).strip("_")
        if key_norm and key_norm in normalized:
            return prefix

    return default


def _parse_classic_node_record(record: str) -> Tuple[int, float, float, float] | None:
    """Parse a section-403 node record from Femap/EMSolution Neutral files.

    The standard coordinate fields are CSV columns 12--14 (zero-based indices
    11--13).  Femap 10.3 can append two status fields after Z, while EMSolution
    4.41 ends the record at Z.  Reading the final three numeric values therefore
    collapses 10.3 coordinates to the trailing status values.

    A final-three-values fallback is retained for older compact records that do
    not contain the standard 14 columns.
    """

    fields = [field.strip() for field in record.split(",")]
    if not fields or not fields[0]:
        return None
    try:
        node_id = int(fields[0])
        if len(fields) >= 14:
            x, y, z = (float(value) for value in fields[11:14])
            return node_id, x, y, z
    except ValueError:
        return None

    numeric_values: List[float] = []
    for value in fields[1:]:
        if not value:
            continue
        try:
            numeric_values.append(float(value))
        except ValueError:
            continue
    if len(numeric_values) < 3:
        return None
    x, y, z = numeric_values[-3:]
    return node_id, x, y, z


def read_mesh(
    path: str | Path,
    *,
    progress: bool = False,
    progress_interval: int = 500_000,
) -> meshio.Mesh:
    """Read Femap Neutral (.neu) mesh supporting v4.1/v10.3 samples.

    Handles classic neutral sections:
      - 403: nodes (CSV: id, ..., x, y, z)
      - 404: elements (descriptor line with count, followed by connectivity line)
    """
    path = Path(path)
    if progress_interval <= 0:
        raise ValueError("progress_interval must be positive")
    reporter = _MeshReadProgress(path, progress_interval) if progress else None
    nodes: Dict[int, Tuple[float, float, float]] = {}
    # Store per-type elements as tuples: (connectivity, matid, eid)
    cells_by_type: Dict[str, List[Tuple[List[int], int, int]]] = {}
    if _is_simple_mesh_format(path):
        _read_simple_mesh(path, nodes, cells_by_type, reporter)
    else:
        _read_classic_mesh(path, nodes, cells_by_type, reporter)

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
        cell_data.setdefault("property_id", []).append(np.array([mats[i] for i in keep_idx], dtype=int))
        cell_data.setdefault("element_id", []).append(np.array([eids[i] for i in keep_idx], dtype=int))

    mesh = meshio.Mesh(
        points=points, cells=cells, point_data={"id": np.array(sorted_node_ids, int)}, cell_data=cell_data
    )
    if reporter is not None:
        reporter.finish()
    return mesh


class _MeshReadProgress:
    def __init__(self, path: Path, interval: int) -> None:
        self.path = path
        self.interval = interval
        self.nodes = 0
        self.elements = 0
        self._next_nodes = interval
        self._next_elements = interval
        self._started = monotonic()
        size_gib = path.stat().st_size / (1024**3)
        print(f"Femap read: start {path} ({size_gib:.2f} GiB)", flush=True)

    def node(self) -> None:
        self.nodes += 1
        if self.nodes >= self._next_nodes:
            self._report("nodes")
            self._next_nodes += self.interval

    def element(self) -> None:
        self.elements += 1
        if self.elements >= self._next_elements:
            self._report("elements")
            self._next_elements += self.interval

    def finish(self) -> None:
        self._report("complete")

    def _report(self, phase: str) -> None:
        elapsed = monotonic() - self._started
        print(
            f"Femap read: phase={phase} nodes={self.nodes:,} elements={self.elements:,} "
            f"elapsed={elapsed:.1f}s",
            flush=True,
        )


def _is_simple_mesh_format(path: Path) -> bool:
    """Identify the compact dollar-delimited format without scanning the file."""

    with path.open(encoding="utf-8", errors="ignore") as stream:
        for line_index, line in enumerate(stream):
            if line.strip().lower() == "$ nodes":
                return True
            if line_index >= 4095:
                break
    return False


def _parse_csv_ints(value: str) -> List[int]:
    return [
        int(token)
        for token in value.replace(" ", "").split(",")
        if token and token.replace("-", "").isdigit()
    ]


def _read_simple_mesh(
    path: Path,
    nodes: Dict[int, Tuple[float, float, float]],
    cells_by_type: Dict[str, List[Tuple[List[int], int, int]]],
    reporter: _MeshReadProgress | None,
) -> None:
    mode = None
    with path.open(encoding="utf-8", errors="ignore") as stream:
        for line in stream:
            value = line.strip()
            if not value:
                continue
            if value.startswith("$"):
                lowered = value.lower()
                if "nodes" in lowered:
                    mode = "nodes"
                elif "elements" in lowered:
                    mode = "elements"
                continue
            if mode == "nodes":
                parts = value.split()
                if len(parts) < 4:
                    continue
                try:
                    node_id = int(parts[0])
                    x, y, z = map(float, parts[1:4])
                except ValueError:
                    continue
                nodes[node_id] = (x, y, z)
                if reporter is not None:
                    reporter.node()
            elif mode == "elements":
                parsed = _parse_simple_element(value)
                if parsed is None:
                    continue
                cell_type, connectivity, material_id, element_id = parsed
                cells_by_type.setdefault(cell_type, []).append(
                    (connectivity, material_id, element_id)
                )
                if reporter is not None:
                    reporter.element()


def _parse_simple_element(value: str) -> Tuple[str, List[int], int, int] | None:
    parts = value.split()
    if len(parts) < 3:
        return None
    try:
        element_id = int(parts[0])
    except ValueError:
        return None
    element_token = parts[1].upper()
    if element_token not in FEMAP_TO_MESHIO:
        return None
    cell_type, expected = FEMAP_TO_MESHIO[element_token]
    integers = [int(token) for token in parts[2:] if token.isdigit()]
    connectivity = integers[-expected:] if len(integers) >= expected else integers
    if len(connectivity) != expected:
        return None
    return cell_type, connectivity, 0, element_id


def _read_classic_mesh(
    path: Path,
    nodes: Dict[int, Tuple[float, float, float]],
    cells_by_type: Dict[str, List[Tuple[List[int], int, int]]],
    reporter: _MeshReadProgress | None,
) -> None:
    with path.open(encoding="utf-8", errors="ignore") as stream:
        while True:
            line = stream.readline()
            if not line:
                return
            if line.strip() != "-1":
                continue
            section_line = stream.readline()
            if not section_line:
                return
            section_value = section_line.strip()
            if not section_value.isdigit():
                continue
            section = int(section_value)
            if section == 403:
                _read_classic_nodes(stream, nodes, reporter)
            elif section == 404:
                _read_classic_elements(stream, cells_by_type, reporter)
            else:
                _skip_classic_section(stream)


def _read_classic_nodes(
    stream,
    nodes: Dict[int, Tuple[float, float, float]],
    reporter: _MeshReadProgress | None,
) -> None:
    for line in stream:
        value = line.strip()
        if value == "-1":
            return
        parsed = _parse_classic_node_record(value)
        if parsed is None:
            continue
        node_id, x, y, z = parsed
        nodes[node_id] = (x, y, z)
        if reporter is not None:
            reporter.node()


def _read_classic_elements(
    stream,
    cells_by_type: Dict[str, List[Tuple[List[int], int, int]]],
    reporter: _MeshReadProgress | None,
) -> None:
    while True:
        descriptor = stream.readline()
        if not descriptor or descriptor.strip() == "-1":
            return
        descriptor_values = _parse_csv_ints(descriptor.strip())
        if len(descriptor_values) < 5:
            continue
        connectivity_line_1 = stream.readline()
        connectivity_line_2 = stream.readline()
        if not connectivity_line_1 or not connectivity_line_2:
            return
        nodes20 = _parse_csv_ints(connectivity_line_1) + _parse_csv_ints(connectivity_line_2)
        nodes20 = [node_id for node_id in nodes20 if node_id != 0]
        element_id = descriptor_values[0]
        material_id = descriptor_values[2]
        topology = descriptor_values[4]
        cell_type, connectivity = _classic_connectivity(topology, nodes20)
        if cell_type is not None and connectivity:
            cells_by_type.setdefault(cell_type, []).append(
                (connectivity, material_id, element_id)
            )
            if reporter is not None:
                reporter.element()
        for _ in range(4):
            auxiliary = stream.readline()
            if not auxiliary or auxiliary.strip() == "-1":
                return


def _classic_connectivity(topology: int, nodes20: List[int]) -> Tuple[str | None, List[int]]:
    mapping = {
        0: ("line", 2),
        9: ("vertex", 1),
        2: ("triangle", 3),
        4: ("quad", 4),
        6: ("tetra", 4),
        10: ("tetra10", 10),
        7: ("wedge", 6),
        14: ("pyramid", 5),
        19: ("pyramid13", 13),
        8: ("hexahedron", 8),
        12: ("hexahedron20", 20),
    }
    if topology == 3:
        values = nodes20[:6]
        if len(values) < 6:
            return "triangle6", values
        return "triangle6", [values[0], values[4], values[1], values[5], values[2], values[5]]
    if topology == 5:
        values = nodes20[:8]
        if len(values) < 8:
            return "quad8", values
        return "quad8", [values[0], values[4], values[1], values[5], values[2], values[6], values[3], values[7]]
    if topology == 11:
        return ("wedge15", nodes20[:15]) if len(nodes20) >= 15 else ("wedge", nodes20[:6])
    if topology in mapping:
        cell_type, count = mapping[topology]
        return cell_type, nodes20[:count]
    fallback = {3: "triangle", 4: "quad", 8: "hexahedron"}.get(len(nodes20))
    return fallback, nodes20 if fallback is not None else []


def _skip_classic_section(stream) -> None:
    for line in stream:
        if line.strip() == "-1":
            return


def file_header(out: List[str]) -> List[str]:
    # 403: Nodes (CSV style like samples)
    out.append("   -1\n")
    out.append("   100\n")
    out.append("<NULL>\n")
    out.append("4.41,\n")
    out.append("   -1\n")

    return out


def write_mesh(path: str | Path, mesh: meshio.Mesh, version: str = "4.41") -> None:
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
        if "element_id" in mesh.cell_data:
            eid_blocks = [np.asarray(a, dtype=int) for a in mesh.cell_data.get("element_id", [])]
        if "property_id" in mesh.cell_data:
            matid_blocks = [np.asarray(a, dtype=int) for a in mesh.cell_data.get("property_id", [])]

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


def read_post(path: str | Path) -> List[dict]:
    """Read Femap Neutral post data (450 + 451) into ATLAS-like records.

    Accumulates per-component datasets inside each 451 section using temporary
    maps, then assigns per-id records as {'component1': ..., 'component2': ..., ...}.
    """
    path = Path(path)
    with path.open(encoding="utf-8", errors="ignore") as f:
        lines = [ln.rstrip() for ln in f]

    steps: List[dict] = []
    i = 0
    n = len(lines)

    comp_re = re.compile(r"-(\d+)\s*$")

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
            current_comp: int | None = None  # 1-based component index from dataset title suffix

            # Per-component accumulators for this section (component index -> {id -> value})
            elems_comp: Dict[int, Dict[int, float]] = {}
            nodes_comp: Dict[int, Dict[int, float]] = {}
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
                # Infer component index from title suffix "-<n>" (1-based).
                # If missing, fall back to sequential numbering to avoid dropping data.
                m = comp_re.search(val_name)
                current_comp = int(m.group(1)) if m else (data_no + 1)
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

                # Assign accumulated values for this dataset into the component map.
                # acc is {dataset_no -> {id -> val}}; we clear it after each dataset.
                no_map = {}
                for mapp in acc.values():
                    no_map.update({int(rid): float(v) for rid, v in mapp.items()})
                acc = {}

                if current_comp is None:
                    continue
                if current_is_element is True:
                    comp_map = elems_comp.setdefault(int(current_comp), {})
                else:
                    comp_map = nodes_comp.setdefault(int(current_comp), {})
                for rid, v in no_map.items():
                    comp_map[int(rid)] = float(v)

            # Build per-id records (component1..componentN)
            ids_e = set().union(*[set(d.keys()) for d in elems_comp.values()]) if elems_comp else set()
            max_c_e = max(elems_comp.keys()) if elems_comp else 0
            for eid in sorted(ids_e):
                vals = [elems_comp.get(c, {}).get(eid, 0.0) for c in range(1, max_c_e + 1)]
                current["elements"][eid] = record_from_components(vals)

            ids_n = set().union(*[set(d.keys()) for d in nodes_comp.values()]) if nodes_comp else set()
            max_c_n = max(nodes_comp.keys()) if nodes_comp else 0
            for nid in sorted(ids_n):
                vals = [nodes_comp.get(c, {}).get(nid, 0.0) for c in range(1, max_c_n + 1)]
                current["nodes"][nid] = record_from_components(vals)

            if i < n and lines[i].strip() == "-1":
                i += 1
            continue

        # Fast-forward other sections
        while i < n and lines[i].strip() != "-1":
            i += 1
        if i < n and lines[i].strip() == "-1":
            i += 1

    return steps


def write_post(
    path: str | Path,
    steps: List[dict],
    mode: str | None = None,
    style: str = "451",
    title_prefix: str | None = None,
) -> None:
    """Write Femap Neutral post data using 450 + 451 (default) or 1051 sections.

    - 450: step header with STEP and Time lines
    - 451: datasets with per-line "id, value,"
    - 1051: datasets with range lines "start,end,values..."
    """
    path = Path(path)
    resolved_title_prefix = (
        title_prefix.strip().upper() if isinstance(title_prefix, str) and title_prefix.strip()
        else infer_title_prefix_from_filename(path)
    )
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

        elems: Dict[int, dict] = st.get("elements", {}) or {}
        nodes: Dict[int, dict] = st.get("nodes", {}) or {}

        mode_norm = (mode or "components").lower()
        if mode_norm not in ("scalar", "vector", "vector+scalar", "components", "all"):
            raise ValueError(f"Unsupported mode: {mode_norm}")
        if mode_norm == "all":
            mode_norm = "components"

        def _target_ncomp(bucket: Dict[int, dict]) -> int:
            if mode_norm == "scalar":
                return 1
            if mode_norm == "vector":
                return 3
            if mode_norm == "vector+scalar":
                return 4
            mx = 0
            for rec in bucket.values():
                if isinstance(rec, dict) and ("vector" in rec or "value" in rec):
                    raise TypeError("Femap post records must use componentN keys; legacy vector/value records are not supported")
                if isinstance(rec, dict):
                    mx = max(mx, max_component_index_in_record(rec))
            return mx

        ncomp_e = _target_ncomp(elems) if elems else 0
        ncomp_n = _target_ncomp(nodes) if nodes else 0

        # Elements: DSIDs 60031.., titles <prefix>-elem-<comp>
        if elems and ncomp_e > 0:
            # Preserve original element ID order as provided in steps
            ids_sorted_e = list(elems.keys())
            datasets_e = [(60030 + c, f"{resolved_title_prefix}-elem-{c}", c) for c in range(1, ncomp_e + 1)]
            for dsid, title, comp_idx in datasets_e:
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
                            val = get_component(rec, comp_idx, 0.0)
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
                        val = get_component(rec, comp_idx, 0.0)
                        out.append(f"{eid}, {f13(val)},\n")
                    out.append("-1,0.,\n")

        # Nodes: DSIDs 31.., titles <prefix>-node-<comp>
        if nodes and ncomp_n > 0:
            # Preserve original node ID order as provided in steps
            ids_sorted_n = list(nodes.keys())
            datasets_n = [(30 + c, f"{resolved_title_prefix}-node-{c}", c) for c in range(1, ncomp_n + 1)]
            for dsid, title, comp_idx in datasets_n:
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
                            val = get_component(rec, comp_idx, 0.0)
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
                        val = get_component(rec, comp_idx, 0.0)
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


# Backward-compatible aliases
read_neu = read_mesh
write_neu = write_mesh
read_neu_post = read_post
write_neu_post = write_post
