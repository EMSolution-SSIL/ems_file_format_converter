# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Science Solutions International Laboratory, Inc.
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import re
from typing import Iterable

import meshio
import numpy as np


_CELL_TYPE_INFO = {
    "vertex": (15, 0, 1),
    "line": (1, 1, 2),
    "triangle": (2, 2, 3),
    "quad": (3, 2, 4),
    "tetra": (4, 3, 4),
    "hexahedron": (5, 3, 8),
    "wedge": (6, 3, 6),
    "pyramid": (7, 3, 5),
    "line3": (8, 1, 3),
    "triangle6": (9, 2, 6),
    "quad9": (10, 2, 9),
    "tetra10": (11, 3, 10),
    "hexahedron27": (12, 3, 27),
    "wedge18": (13, 3, 18),
    "pyramid14": (14, 3, 14),
    "quad8": (16, 2, 8),
    "hexahedron20": (17, 3, 20),
    "wedge15": (18, 3, 15),
    "pyramid13": (19, 3, 13),
}
_GMSH_TYPE_INFO = {gmsh_type: (name, dim, count) for name, (gmsh_type, dim, count) in _CELL_TYPE_INFO.items()}


def _section_lines(text: str, name: str) -> list[str] | None:
    lines = text.splitlines()
    start_marker = f"${name}"
    end_marker = f"$End{name}"
    try:
        start = lines.index(start_marker) + 1
        end = lines.index(end_marker, start)
    except ValueError:
        return None
    return lines[start:end]


def _mesh_format(text: str) -> tuple[str, int]:
    lines = _section_lines(text, "MeshFormat")
    if not lines:
        raise ValueError("Gmsh file does not contain a $MeshFormat section")
    values = lines[0].split()
    if len(values) != 3:
        raise ValueError("invalid Gmsh $MeshFormat header")
    return values[0], int(values[1])


def _parse_node_and_element_tags(
    text: str,
) -> tuple[np.ndarray | None, np.ndarray | None, list[tuple[str, np.ndarray]] | None]:
    version, binary = _mesh_format(text)
    if binary or not version.startswith("4."):
        return None, None, None

    node_lines = _section_lines(text, "Nodes")
    element_lines = _section_lines(text, "Elements")
    if node_lines is None or element_lines is None:
        return None, None, None

    node_tokens = iter(" ".join(node_lines).split())
    node_block_count = int(next(node_tokens))
    node_count = int(next(node_tokens))
    next(node_tokens)
    next(node_tokens)
    node_tags: list[int] = []
    node_dim_tags: list[tuple[int, int]] = []
    for _ in range(node_block_count):
        entity_dim = int(next(node_tokens))
        entity_tag = int(next(node_tokens))
        parametric = int(next(node_tokens))
        block_count = int(next(node_tokens))
        block_tags = [int(next(node_tokens)) for _ in range(block_count)]
        node_tags.extend(block_tags)
        node_dim_tags.extend([(entity_dim, entity_tag)] * block_count)
        values_per_node = 3 + (entity_dim if parametric else 0)
        for _ in range(block_count * values_per_node):
            next(node_tokens)
    if len(node_tags) != node_count:
        raise ValueError("Gmsh node count does not match the $Nodes header")

    element_tokens = iter(" ".join(element_lines).split())
    element_block_count = int(next(element_tokens))
    element_count = int(next(element_tokens))
    next(element_tokens)
    next(element_tokens)
    element_blocks: list[tuple[str, np.ndarray]] = []
    parsed_count = 0
    for _ in range(element_block_count):
        next(element_tokens)
        next(element_tokens)
        gmsh_type = int(next(element_tokens))
        block_count = int(next(element_tokens))
        if gmsh_type not in _GMSH_TYPE_INFO:
            return np.asarray(node_tags), np.asarray(node_dim_tags), None
        cell_type, _, nodes_per_element = _GMSH_TYPE_INFO[gmsh_type]
        tags = np.empty(block_count, dtype=np.int64)
        for index in range(block_count):
            tags[index] = int(next(element_tokens))
            for _ in range(nodes_per_element):
                next(element_tokens)
        element_blocks.append((cell_type, tags))
        parsed_count += block_count
    if parsed_count != element_count:
        raise ValueError("Gmsh element count does not match the $Elements header")
    return np.asarray(node_tags), np.asarray(node_dim_tags), element_blocks


def read_mesh(path: str | Path) -> meshio.Mesh:
    """Read Gmsh and retain actual node/element tags for topology editing."""
    mesh = meshio.read(path)
    raw = Path(path).read_bytes()
    if b"\x00" in raw[:256]:
        return mesh
    try:
        text = raw.decode("utf-8")
        node_tags, node_dim_tags, element_blocks = _parse_node_and_element_tags(text)
    except (UnicodeDecodeError, ValueError, StopIteration):
        return mesh

    if node_tags is not None and len(node_tags) == len(mesh.points):
        mesh.point_data["gmsh:node_tags"] = node_tags
        if "gmsh:dim_tags" not in mesh.point_data and node_dim_tags is not None:
            mesh.point_data["gmsh:dim_tags"] = node_dim_tags

    if element_blocks is not None and len(element_blocks) == len(mesh.cells):
        aligned = all(
            cell.type == parsed_type and len(cell.data) == len(tags)
            for cell, (parsed_type, tags) in zip(mesh.cells, element_blocks, strict=True)
        )
        if aligned:
            mesh.cell_data["gmsh:element_tags"] = [tags for _, tags in element_blocks]
    return mesh


def _cell_data_arrays(
    mesh: meshio.Mesh,
    keys: Iterable[str],
    default: int = 0,
) -> list[np.ndarray]:
    for key in keys:
        if key in mesh.cell_data:
            arrays = mesh.cell_data[key]
            if len(arrays) != len(mesh.cells):
                raise ValueError(f"cell_data[{key!r}] does not match mesh cell blocks")
            result = []
            for block, values in zip(mesh.cells, arrays, strict=True):
                array = np.asarray(values, dtype=np.int64).reshape(-1)
                if len(array) != len(block.data):
                    raise ValueError(f"cell_data[{key!r}] block length does not match cells")
                result.append(array.copy())
            return result
    return [np.full(len(block.data), default, dtype=np.int64) for block in mesh.cells]


def _allocate_missing_tags(values: np.ndarray) -> np.ndarray:
    tags = np.asarray(values, dtype=np.int64).copy()
    positive = tags[tags > 0]
    if len(np.unique(positive)) != len(positive):
        raise ValueError("Gmsh tags must be unique")
    used = set(int(tag) for tag in positive)
    next_tag = max(used, default=0) + 1
    for index in np.flatnonzero(tags <= 0):
        while next_tag in used:
            next_tag += 1
        tags[index] = next_tag
        used.add(next_tag)
        next_tag += 1
    return tags


def _point_tags(mesh: meshio.Mesh) -> np.ndarray:
    for key in ("gmsh:node_tags", "id"):
        if key in mesh.point_data:
            values = np.asarray(mesh.point_data[key], dtype=np.int64).reshape(-1)
            if len(values) != len(mesh.points):
                raise ValueError(f"point_data[{key!r}] length does not match mesh points")
            return _allocate_missing_tags(values)
    return np.arange(1, len(mesh.points) + 1, dtype=np.int64)


def _element_tags(mesh: meshio.Mesh) -> list[np.ndarray]:
    arrays = _cell_data_arrays(mesh, ("gmsh:element_tags", "element_id"))
    sizes = [len(array) for array in arrays]
    flat = _allocate_missing_tags(np.concatenate(arrays) if arrays else np.empty(0, dtype=int))
    result = []
    start = 0
    for size in sizes:
        result.append(flat[start : start + size])
        start += size
    return result


def _normalized_entity_tags(
    mesh: meshio.Mesh, physical: list[np.ndarray]
) -> list[np.ndarray]:
    geometrical = _cell_data_arrays(mesh, ("gmsh:geometrical",))
    used_by_dimension: dict[int, set[int]] = defaultdict(set)
    for block, values in zip(mesh.cells, geometrical, strict=True):
        _, dimension, _ = _CELL_TYPE_INFO.get(block.type, (None, None, None))
        if dimension is None:
            raise ValueError(f"unsupported Gmsh cell type: {block.type}")
        used_by_dimension[dimension].update(int(value) for value in values if value > 0)

    defaults: dict[int, int] = {}
    result = []
    for block, physical_values, geometry_values in zip(
        mesh.cells, physical, geometrical, strict=True
    ):
        _, dimension, _ = _CELL_TYPE_INFO[block.type]
        normalized = geometry_values.copy()
        for index in np.flatnonzero(normalized <= 0):
            physical_tag = int(physical_values[index])
            if physical_tag > 0:
                normalized[index] = physical_tag
                used_by_dimension[dimension].add(physical_tag)
            else:
                if dimension not in defaults:
                    candidate = 1
                    while candidate in used_by_dimension[dimension]:
                        candidate += 1
                    defaults[dimension] = candidate
                    used_by_dimension[dimension].add(candidate)
                normalized[index] = defaults[dimension]
        result.append(normalized)
    return result


def _derive_node_dim_tags(
    mesh: meshio.Mesh,
    geometrical: list[np.ndarray],
) -> np.ndarray:
    existing = mesh.point_data.get("gmsh:dim_tags")
    if existing is None:
        result = np.zeros((len(mesh.points), 2), dtype=np.int64)
    else:
        result = np.asarray(existing, dtype=np.int64).copy()
        if result.shape != (len(mesh.points), 2):
            raise ValueError("point_data['gmsh:dim_tags'] must have shape (num_points, 2)")

    incident: list[set[tuple[int, int]]] = [set() for _ in range(len(mesh.points))]
    for block, entity_values in zip(mesh.cells, geometrical, strict=True):
        _, dimension, _ = _CELL_TYPE_INFO[block.type]
        for connectivity, entity_tag in zip(block.data, entity_values, strict=True):
            for point_index in connectivity:
                incident[int(point_index)].add((dimension, int(entity_tag)))

    max_entity_by_dimension: dict[int, int] = defaultdict(int)
    for dimension, entity_tag in result:
        if entity_tag > 0:
            max_entity_by_dimension[int(dimension)] = max(
                max_entity_by_dimension[int(dimension)], int(entity_tag)
            )
    for point_index, candidates in enumerate(incident):
        if result[point_index, 1] > 0:
            continue
        if candidates:
            result[point_index] = min(candidates)
        else:
            max_entity_by_dimension[0] += 1
            result[point_index] = (0, max_entity_by_dimension[0])
    if np.any(result[:, 0] < 0) or np.any(result[:, 0] > 3) or np.any(result[:, 1] <= 0):
        raise ValueError("invalid Gmsh node dimension/entity tags")
    return result


def _physical_name_lines(field_data: dict[str, np.ndarray]) -> list[str]:
    entries = []
    seen = set()
    for name, values in field_data.items():
        array = np.asarray(values).reshape(-1)
        if len(array) < 2:
            raise ValueError(f"field_data[{name!r}] must contain physical tag and dimension")
        physical_tag, dimension = int(array[0]), int(array[1])
        key = (dimension, physical_tag)
        if key in seen:
            raise ValueError(f"duplicate physical name for dimension/tag {key}")
        seen.add(key)
        escaped_name = str(name).replace("\\", "\\\\").replace('"', '\\"')
        entries.append((dimension, physical_tag, escaped_name))
    entries.sort()
    if not entries:
        return []
    return ["$PhysicalNames", str(len(entries))] + [
        f'{dimension} {tag} "{name}"' for dimension, tag, name in entries
    ] + ["$EndPhysicalNames"]


def write_mesh(path: str | Path, mesh: meshio.Mesh) -> None:
    """Write topology-changing meshes as Gmsh 4.1 ASCII with stable tags."""
    points = np.asarray(mesh.points, dtype=float)
    if points.ndim != 2 or points.shape[1] not in {2, 3}:
        raise ValueError("Gmsh points must be an N x 2 or N x 3 array")
    if points.shape[1] == 2:
        points = np.column_stack([points, np.zeros(len(points))])
    for block in mesh.cells:
        if block.type not in _CELL_TYPE_INFO:
            raise ValueError(f"unsupported Gmsh cell type: {block.type}")
        connectivity = np.asarray(block.data)
        if connectivity.size and (np.min(connectivity) < 0 or np.max(connectivity) >= len(points)):
            raise ValueError("cell connectivity contains an invalid point index")

    physical = _cell_data_arrays(mesh, ("gmsh:physical", "property_id"))
    if any(np.any(values < 0) for values in physical):
        raise ValueError("physical tags must be non-negative")
    geometrical = _normalized_entity_tags(mesh, physical)
    node_tags = _point_tags(mesh)
    element_tags = _element_tags(mesh)
    node_dim_tags = _derive_node_dim_tags(mesh, geometrical)

    entity_points: dict[tuple[int, int], set[int]] = defaultdict(set)
    entity_physical: dict[tuple[int, int], set[int]] = defaultdict(set)
    element_groups: dict[tuple[int, int, str], list[tuple[int, np.ndarray]]] = defaultdict(list)
    for block, physical_values, entity_values, tags in zip(
        mesh.cells, physical, geometrical, element_tags, strict=True
    ):
        _, dimension, _ = _CELL_TYPE_INFO[block.type]
        for connectivity, physical_tag, entity_tag, element_tag in zip(
            block.data, physical_values, entity_values, tags, strict=True
        ):
            entity_key = (dimension, int(entity_tag))
            entity_points[entity_key].update(int(index) for index in connectivity)
            if physical_tag > 0:
                entity_physical[entity_key].add(int(physical_tag))
            element_groups[(dimension, int(entity_tag), block.type)].append(
                (int(element_tag), np.asarray(connectivity, dtype=np.int64))
            )

    for point_index, (dimension, entity_tag) in enumerate(node_dim_tags):
        entity_points[(int(dimension), int(entity_tag))].add(point_index)
    for key, tags in entity_physical.items():
        if len(tags) > 1:
            raise ValueError(
                f"Gmsh entity {key} has multiple physical tags; split it into distinct entities"
            )

    lines = ["$MeshFormat", "4.1 0 8", "$EndMeshFormat"]
    lines.extend(_physical_name_lines(mesh.field_data))

    entities_by_dimension = {
        dimension: sorted(tag for dim, tag in entity_points if dim == dimension)
        for dimension in range(4)
    }
    lines.extend(
        [
            "$Entities",
            " ".join(str(len(entities_by_dimension[dimension])) for dimension in range(4)),
        ]
    )
    for dimension in range(4):
        for entity_tag in entities_by_dimension[dimension]:
            key = (dimension, entity_tag)
            indices = np.asarray(sorted(entity_points[key]), dtype=int)
            entity_coords = points[indices]
            physical_tags = sorted(entity_physical.get(key, set()))
            physical_part = " ".join(str(tag) for tag in physical_tags)
            if dimension == 0:
                coord = entity_coords[0]
                line = f"{entity_tag} {coord[0]:.17g} {coord[1]:.17g} {coord[2]:.17g} {len(physical_tags)}"
            else:
                minimum = np.min(entity_coords, axis=0)
                maximum = np.max(entity_coords, axis=0)
                line = (
                    f"{entity_tag} {minimum[0]:.17g} {minimum[1]:.17g} {minimum[2]:.17g} "
                    f"{maximum[0]:.17g} {maximum[1]:.17g} {maximum[2]:.17g} {len(physical_tags)}"
                )
            if physical_part:
                line += " " + physical_part
            if dimension > 0:
                line += " 0"
            lines.append(line)
    lines.append("$EndEntities")

    node_groups: dict[tuple[int, int], list[int]] = defaultdict(list)
    for point_index, dim_tag in enumerate(node_dim_tags):
        node_groups[(int(dim_tag[0]), int(dim_tag[1]))].append(point_index)
    lines.extend(
        [
            "$Nodes",
            f"{len(node_groups)} {len(points)} {int(np.min(node_tags)) if len(node_tags) else 0} "
            f"{int(np.max(node_tags)) if len(node_tags) else 0}",
        ]
    )
    for (dimension, entity_tag), indices in sorted(node_groups.items()):
        lines.append(f"{dimension} {entity_tag} 0 {len(indices)}")
        lines.extend(str(int(node_tags[index])) for index in indices)
        lines.extend(
            f"{points[index, 0]:.17g} {points[index, 1]:.17g} {points[index, 2]:.17g}"
            for index in indices
        )
    lines.append("$EndNodes")

    all_element_tags = np.concatenate(element_tags) if element_tags else np.empty(0, dtype=int)
    lines.extend(
        [
            "$Elements",
            f"{len(element_groups)} {len(all_element_tags)} "
            f"{int(np.min(all_element_tags)) if len(all_element_tags) else 0} "
            f"{int(np.max(all_element_tags)) if len(all_element_tags) else 0}",
        ]
    )
    for (dimension, entity_tag, cell_type), elements in sorted(element_groups.items()):
        gmsh_type, _, _ = _CELL_TYPE_INFO[cell_type]
        lines.append(f"{dimension} {entity_tag} {gmsh_type} {len(elements)}")
        for element_tag, connectivity in elements:
            connectivity_tags = " ".join(str(int(node_tags[index])) for index in connectivity)
            lines.append(f"{element_tag} {connectivity_tags}")
    lines.append("$EndElements")

    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
