"""
dxf_to_panels.py

Extract RigidPanel geometry and interior crease edges from a flat DXF
crease pattern (a series of LINE/LWPOLYLINE segments meeting at shared
vertices -- no layers, no pre-closed panel polygons required).

Method: treat the DXF's line segments as a planar straight-line graph and
recover panels as its enclosed faces via shapely.ops.polygonize. An edge
shared by exactly two resulting faces is an interior crease (a candidate
kinematic-coupling spoke between those two panels); an edge bordering only
one face is the pattern's outer boundary, not a crease.

Verified on Flasher_gore_reference_cleaned.dxf: 17 raw lines -> 6 panels,
6 interior creases, every input line accounted for in exactly one polygon
edge (no gaps, no leftover dangling segments) -- not a degenerate file.
"""
import numpy as np
import ezdxf
from shapely.ops import polygonize
from shapely.geometry import LineString

from RigidBodyModel import RigidPanel

_ROUND = 6


def _pt(p):
    return (round(p[0], _ROUND), round(p[1], _ROUND))


def _segments_from_entities(entities, segs):
    for e in entities:
        if e.dxftype() == 'LINE':
            s, en = e.dxf.start, e.dxf.end
            segs.append((_pt((s.x, s.y)), _pt((en.x, en.y))))
        elif e.dxftype() == 'LWPOLYLINE':
            pts = [_pt((x, y)) for x, y, *_ in e.get_points()]
            segs.extend(zip(pts, pts[1:]))
            if e.closed and len(pts) > 1:
                segs.append((pts[-1], pts[0]))
        elif e.dxftype() == 'INSERT':
            # Block reference (e.g. a mirrored/rotated copy of a neighbor
            # gore's boundary panels) -- virtual_entities() returns the
            # block's content already transformed by the insert's
            # position/rotation/scale, recursing through nested blocks.
            _segments_from_entities(e.virtual_entities(), segs)


def _read_segments(dxf_path):
    doc = ezdxf.readfile(dxf_path)
    msp = doc.modelspace()
    segs = []
    _segments_from_entities(msp, segs)
    return segs


def _snap_points(segs, tol):
    """
    Merge endpoints within `tol` of each other to a single representative
    point (simple O(n^2) union-find -- fine for the small point counts
    these crease patterns have). Needed when a DXF's lines were drawn/
    copied at slightly different precision than the geometry they're
    meant to coincide with (e.g. a mirrored block insert landing a hair
    off the original vertex) -- without this, polygonize() sees a tiny
    gap instead of a shared vertex and fails to close the face.
    """
    if tol <= 0:
        return segs

    pts = sorted({p for seg in segs for p in seg})
    parent = list(range(len(pts)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            dx = pts[i][0] - pts[j][0]
            dy = pts[i][1] - pts[j][1]
            if (dx * dx + dy * dy) ** 0.5 <= tol:
                union(i, j)

    # representative = first point (in sorted order) in each cluster
    cluster_rep = {}
    for i, p in enumerate(pts):
        cluster_rep.setdefault(find(i), p)
    remap = {p: cluster_rep[find(i)] for i, p in enumerate(pts)}

    return [(remap[a], remap[b]) for a, b in segs]


def load_gore_geometry(dxf_path, thickness=None, thickness_fraction=0.05,
                        snap_tol=None):
    """
    Returns (panels, creases, thickness).

    panels:  list[RigidPanel], ids 0..n_panels-1 (order = polygonize's
             face order -- arbitrary but stable across calls), z=0 (flat/
             unfolded state), thickness as resolved below.
    creases: list of dicts {panel_a, panel_b, p_start, p_end} -- panel_a/b
             are ids into `panels`, p_start/p_end are (3,) arrays, the two
             endpoints of the shared edge (one interior crease = one
             candidate coupling spoke between those two panels).
    thickness: the value actually used -- either the caller's `thickness`,
             or thickness_fraction * mean crease-edge length if not given
             (so panel thickness scales with whatever units the DXF uses,
             rather than an arbitrary fixed constant).
    snap_tol: merge endpoints within this distance before polygonizing
             (see _snap_points) -- needed for DXFs assembled from
             multiple pieces (e.g. a mirrored block insert) that don't
             land on bit-identical coordinates. Default None = no
             snapping (exact match required, as verified sufficient for
             the single hand-drawn gore file).
    """
    segs = _read_segments(dxf_path)
    if not segs:
        raise ValueError(f"No LINE/LWPOLYLINE entities found in {dxf_path}")
    if snap_tol:
        segs = _snap_points(segs, snap_tol)

    polys = list(polygonize([LineString(s) for s in segs]))
    if not polys:
        raise ValueError(
            f"{dxf_path}: lines don't close into any panel faces -- "
            "check for gaps or dangling segments (degenerate geometry).")

    poly_coords = [list(p.exterior.coords)[:-1] for p in polys]  # drop closing dup

    edge_to_panels = {}
    for panel_id, coords in enumerate(poly_coords):
        n = len(coords)
        for j in range(n):
            a, b = _pt(coords[j]), _pt(coords[(j + 1) % n])
            key = tuple(sorted([a, b]))
            edge_to_panels.setdefault(key, []).append(panel_id)

    creases = []
    for (a, b), panel_ids in edge_to_panels.items():
        if len(panel_ids) == 2:
            p_start = np.array([a[0], a[1], 0.0])
            p_end = np.array([b[0], b[1], 0.0])
            creases.append(dict(
                panel_a=panel_ids[0], panel_b=panel_ids[1],
                p_start=p_start, p_end=p_end,
            ))
        elif len(panel_ids) > 2:
            raise ValueError(
                f"Edge {a}-{b} borders {len(panel_ids)} panels -- "
                "overlapping/duplicate lines in the DXF?")

    if thickness is None:
        lengths = [np.linalg.norm(c['p_end'] - c['p_start']) for c in creases]
        thickness = thickness_fraction * float(np.mean(lengths))

    panels = [RigidPanel(i, vertices=np.array([[x, y, 0.0] for x, y in coords]),
                          thickness=thickness)
              for i, coords in enumerate(poly_coords)]

    return panels, creases, thickness


if __name__ == "__main__":
    import sys, os
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(__file__), 'Flasher_gore_reference_cleaned.dxf')
    panels, creases, t = load_gore_geometry(path)
    print(f"{path}: {len(panels)} panels, {len(creases)} interior creases, "
          f"thickness={t:.4f}")
    for p in panels:
        print(f"  panel {p.id}: {len(p.vertices)} verts, centroid={p.centroid.round(3)}")
    for c in creases:
        print(f"  crease panel {c['panel_a']} <-> panel {c['panel_b']}: "
              f"{c['p_start'][:2].round(3)} -> {c['p_end'][:2].round(3)}")
