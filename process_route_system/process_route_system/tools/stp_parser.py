"""
tools/stp_parser.py — STP File Analysis Tool for the Feature Extraction Agent (FEA).

Parses an ISO 10303 (STEP/STP) file and extracts:
  - Face areas, normal vectors, vertex coordinates
  - Machining feature candidates (holes, slots, pockets, flat surfaces)

The paper uses a deep-learning GNN for semantic segmentation; this module
provides a rule-based geometric fallback that is fully self-contained and
reproduces the OUTPUT FORMAT the agents expect.  Swap in your GNN model by
replacing the `_gnn_segment` stub.

Dependencies: numpy (stdlib geometry only — no OCC / ifcopenshell required).
"""

import re
import math
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ── Data classes ───────────────────────────────────────────────────────────────

class Face:
    def __init__(self, face_id: int, area: float,
                 normal: Tuple[float, float, float],
                 vertices: List[Tuple[float, float, float]]):
        self.face_id  = face_id
        self.area     = area
        self.normal   = normal
        self.vertices = vertices

    def to_dict(self) -> Dict:
        return {
            "face_id":  self.face_id,
            "area":     round(self.area, 2),
            "normal":   [round(v, 4) for v in self.normal],
            "vertices": [[round(c, 4) for c in vtx] for vtx in self.vertices],
        }


class MachiningFeature:
    """
    Recognised machining feature with its associated faces and bottom face.
    Feature types follow the paper's vocabulary:
      through_hole | blind_hole | rectangular_through_slot |
      rectangular_pocket | flat_surface | chamfer | fillet
    """
    def __init__(self, feature_id: int, feature_type: str,
                 face_ids: List[int], bottom_face_id: Optional[int],
                 description: str):
        self.feature_id     = feature_id
        self.feature_type   = feature_type
        self.face_ids       = face_ids
        self.bottom_face_id = bottom_face_id
        self.description    = description

    def to_dict(self) -> Dict:
        return {
            "feature_id":     self.feature_id,
            "feature_type":   self.feature_type,
            "face_ids":       self.face_ids,
            "bottom_face_id": self.bottom_face_id,
            "description":    self.description,
        }


# ── STP token extraction ───────────────────────────────────────────────────────

def _parse_cartesian_points(text: str) -> Dict[int, Tuple[float, float, float]]:
    """Extract all CARTESIAN_POINT entities → {id: (x, y, z)}."""
    pattern = r"#(\d+)\s*=\s*CARTESIAN_POINT\s*\(\s*'[^']*'\s*,\s*\(([^)]+)\)\s*\)"
    points: Dict[int, Tuple[float, float, float]] = {}
    for m in re.finditer(pattern, text):
        pid    = int(m.group(1))
        coords = [float(c.strip()) for c in m.group(2).split(",")]
        if len(coords) == 3:
            points[pid] = (coords[0], coords[1], coords[2])
    return points


def _parse_direction_vectors(text: str) -> Dict[int, Tuple[float, float, float]]:
    """Extract DIRECTION entities → {id: (dx, dy, dz)}."""
    pattern = r"#(\d+)\s*=\s*DIRECTION\s*\(\s*'[^']*'\s*,\s*\(([^)]+)\)\s*\)"
    dirs: Dict[int, Tuple[float, float, float]] = {}
    for m in re.finditer(pattern, text):
        did    = int(m.group(1))
        coords = [float(c.strip()) for c in m.group(2).split(",")]
        if len(coords) == 3:
            dirs[did] = (coords[0], coords[1], coords[2])
    return dirs


def _parse_advanced_faces(text: str,
                           points: Dict[int, Tuple],
                           directions: Dict[int, Tuple]) -> List[Face]:
    """
    Attempt to reconstruct ADVANCED_FACE geometry.
    Returns a list of Face objects with approximate areas and normals.
    """
    # Collect PLANE entities (each has an axis placement → normal direction)
    plane_normals: Dict[int, Tuple[float, float, float]] = {}
    plane_pat = r"#(\d+)\s*=\s*PLANE\s*\(\s*'[^']*'\s*,\s*#(\d+)\s*\)"
    axis_pat  = (r"#(\d+)\s*=\s*AXIS2_PLACEMENT_3D\s*\(\s*'[^']*'\s*,"
                 r"\s*#(\d+)\s*,\s*#(\d+)\s*,\s*#(\d+)\s*\)")
    axis_map: Dict[int, Dict] = {}
    for m in re.finditer(axis_pat, text):
        aid = int(m.group(1))
        axis_map[aid] = {
            "origin": int(m.group(2)),
            "axis":   int(m.group(3)),
            "ref":    int(m.group(4)),
        }
    for m in re.finditer(plane_pat, text):
        pid  = int(m.group(1))
        aid  = int(m.group(2))
        if aid in axis_map:
            normal_id = axis_map[aid]["axis"]
            if normal_id in directions:
                plane_normals[pid] = directions[normal_id]

    # Collect ADVANCED_FACE entries
    af_pat = r"#(\d+)\s*=\s*ADVANCED_FACE\s*\(\s*'[^']*'\s*,\s*\(([^)]+)\)\s*,\s*#(\d+)\s*,\s*\.([A-Z]+)\.\s*\)"
    faces: List[Face] = []
    for m in re.finditer(af_pat, text):
        fid       = int(m.group(1))
        surface_id = int(m.group(3))
        normal = plane_normals.get(surface_id, (0.0, 0.0, 1.0))

        # Gather referenced point ids from the bound list (rough proxy for vertices)
        bound_refs = [int(x.strip().lstrip('#'))
                      for x in m.group(2).split(',')
                      if x.strip().startswith('#')]
        verts = [points[r] for r in bound_refs if r in points]
        area  = _estimate_area(verts) if len(verts) >= 3 else 0.0

        faces.append(Face(fid, area, normal, verts))

    return faces


def _estimate_area(verts: List[Tuple[float, float, float]]) -> float:
    """Shoelace / cross-product area estimate for a planar polygon."""
    if len(verts) < 3:
        return 0.0
    v = np.array(verts)
    # Fan triangulation from first vertex
    total = 0.0
    for i in range(1, len(v) - 1):
        a = v[i]   - v[0]
        b = v[i+1] - v[0]
        total += np.linalg.norm(np.cross(a, b))
    return total / 2.0


# ── Geometric feature recognition ─────────────────────────────────────────────

def _classify_features(faces: List[Face]) -> List[MachiningFeature]:
    """
    Rule-based geometric feature classifier.
    Groups faces by shared normal direction and proximity to identify:
      - Through holes  (cylindrical faces with normals in XY plane)
      - Rectangular slots / pockets / flat surfaces
    """
    features: List[MachiningFeature] = []
    used: set = set()

    # Group faces by quantised normal (rounded to 1 dp)
    normal_groups: Dict[Tuple, List[Face]] = {}
    for f in faces:
        key = tuple(round(n, 1) for n in f.normal)
        normal_groups.setdefault(key, []).append(f)

    feat_id = 0

    # Heuristic: faces with near-zero Z-normal component that form a ring → hole
    for key, group in normal_groups.items():
        nx, ny, nz = key
        if abs(nz) < 0.1 and len(group) >= 2:
            # Likely cylindrical wall → through hole candidate
            ids = [f.face_id for f in group]
            features.append(MachiningFeature(
                feat_id, "through_hole", ids, ids[0],
                f"Through hole — {len(ids)} cylindrical faces, "
                f"normal ≈ ({nx},{ny},{nz})"
            ))
            used.update(ids)
            feat_id += 1

    # Faces with Z-normal ≈ ±1 → flat surfaces / pockets
    for key, group in normal_groups.items():
        nx, ny, nz = key
        if abs(nz) > 0.9 and len(group) >= 1:
            unused = [f for f in group if f.face_id not in used]
            if not unused:
                continue
            ids = [f.face_id for f in unused]
            # Large single face → flat surface; multiple → pocket / slot
            if len(unused) == 1 and unused[0].area > 1000:
                ftype = "flat_surface"
                desc  = f"Flat surface (area ≈ {unused[0].area:.1f})"
            elif len(unused) >= 4:
                ftype = "rectangular_pocket"
                desc  = f"Rectangular pocket — {len(ids)} faces"
            else:
                ftype = "rectangular_through_slot"
                desc  = f"Rectangular through slot — {len(ids)} faces"
            features.append(MachiningFeature(feat_id, ftype, ids, ids[-1], desc))
            used.update(ids)
            feat_id += 1

    # Remaining unclassified faces → generic surface
    remaining = [f for f in faces if f.face_id not in used]
    if remaining:
        ids = [f.face_id for f in remaining]
        features.append(MachiningFeature(feat_id, "flat_surface", ids, None,
                                          f"Flat/general surface — {len(ids)} faces"))

    return features


# ── GNN stub (replace with your trained model) ────────────────────────────────

def _gnn_segment(face_dicts: List[Dict]) -> List[MachiningFeature]:
    """
    Stub for the deep-learning GNN segmentation described in the paper.

    Replace this function body with your actual GNN inference:
      1. Convert face_dicts to a graph (nodes = faces, edges = adjacency)
      2. Run your trained model
      3. Return MachiningFeature list from predicted labels

    Until replaced, this function returns an empty list, causing the caller to
    fall back to rule-based classification.
    """
    logger.warning("GNN model not loaded — falling back to rule-based classifier.")
    return []


# ── Public API ─────────────────────────────────────────────────────────────────

def parse_stp_file(stp_path: str, use_gnn: bool = False) -> Dict[str, Any]:
    """
    Main entry point for the STP File Analysis Tool.

    Args:
        stp_path : Path to the .stp / .step file.
        use_gnn  : If True, attempt GNN-based segmentation (requires model).

    Returns:
        {
          "faces":    [ {face_id, area, normal, vertices}, ... ],
          "features": [ {feature_id, feature_type, face_ids,
                         bottom_face_id, description}, ... ],
          "summary":  "Human-readable summary string"
        }
    """
    path = Path(stp_path)
    if not path.exists():
        raise FileNotFoundError(f"STP file not found: {stp_path}")

    text = path.read_text(encoding="utf-8", errors="ignore")
    logger.info(f"Parsing STP file: {path.name}  ({len(text):,} chars)")

    points    = _parse_cartesian_points(text)
    directions = _parse_direction_vectors(text)
    faces     = _parse_advanced_faces(text, points, directions)
    logger.info(f"  Extracted {len(faces)} ADVANCED_FACE entities")

    face_dicts = [f.to_dict() for f in faces]

    # Feature recognition
    if use_gnn:
        features = _gnn_segment(face_dicts)

    if not use_gnn or not features:
        features = _classify_features(faces)

    logger.info(f"  Recognised {len(features)} machining features")

    # Summary text (mirrors Fig. 9 output style in paper)
    feature_summary_lines = []
    for feat in features:
        line = (f"{feat.feature_type} [{','.join(str(i) for i in feat.face_ids)}]"
                + (f" [{feat.bottom_face_id}]" if feat.bottom_face_id else ""))
        feature_summary_lines.append(line)

    summary = (
        f"This part has {len(faces)} faces.\n"
        f"This part has {len(features)} feature sets, which are:\n"
        + "\n".join(feature_summary_lines)
    )

    return {
        "faces":    face_dicts,
        "features": [f.to_dict() for f in features],
        "summary":  summary,
    }


# ── CLI usage ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys, pprint
    if len(sys.argv) < 2:
        print("Usage: python stp_parser.py <path/to/part.stp>")
        sys.exit(1)
    result = parse_stp_file(sys.argv[1])
    print(result["summary"])
    print(f"\n{len(result['features'])} features detected:")
    for feat in result["features"]:
        pprint.pprint(feat)
