from __future__ import annotations

import re
from typing import Any, Mapping, Sequence


_COMP_RE = re.compile(r"^component(\d+)$")


def record_from_components(vals: Sequence[float]) -> dict[str, float]:
    """Build a per-ID record from a positional component sequence.

    Components are 1-based: component1, component2, ...
    """
    rec: dict[str, float] = {}
    for i, v in enumerate(vals, start=1):
        rec[f"component{i}"] = float(v)
    return rec


def components_from_record(rec: Mapping[str, Any]) -> list[float]:
    """Extract components from a record, ordered by component index."""
    found: list[tuple[int, float]] = []
    for k, v in rec.items():
        m = _COMP_RE.match(str(k))
        if not m:
            continue
        idx = int(m.group(1))
        try:
            fv = float(v)
        except Exception:
            fv = 0.0
        found.append((idx, fv))
    found.sort(key=lambda t: t[0])
    return [v for _, v in found]


def get_component(rec: Mapping[str, Any], idx: int, default: float = 0.0) -> float:
    """Read 1-based component index from a record."""
    try:
        return float(rec.get(f"component{idx}", default))
    except Exception:
        return float(default)


def max_component_index_in_record(rec: Mapping[str, Any]) -> int:
    mx = 0
    for k in rec.keys():
        m = _COMP_RE.match(str(k))
        if not m:
            continue
        mx = max(mx, int(m.group(1)))
    return mx


def max_components_in_step(step: Mapping[str, Any]) -> int:
    mx = 0
    for bucket_key in ("elements", "nodes"):
        bucket = step.get(bucket_key) or {}
        if not isinstance(bucket, Mapping):
            continue
        for rec in bucket.values():
            if isinstance(rec, Mapping):
                mx = max(mx, max_component_index_in_record(rec))
    return mx

