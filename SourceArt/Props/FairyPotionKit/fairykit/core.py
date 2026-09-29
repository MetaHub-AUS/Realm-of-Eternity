"""Core procedural-geometry helpers for the Fairy Potion Kit.

Conventions
-----------
* Scene units: 1 Blender unit (BU) = 10 cm (scene unit scale 0.1), Z up.
* Every asset is built around the world origin, standing on Z = 0 and facing -Y
  (towards the camera). Layout transforms are applied afterwards.
* "Decal space": flat decorations are authored as 2D outlines in (x, y) where
  x = right and y = up. Relief meshes map a 2D point (x, y) with height h to 3D
  (x, -h, y), i.e. they lie in the XZ plane and bulge towards -Y. `wrap()` then
  projects such a mesh onto a curved surface, keeping the relief height along
  the surface normal.
"""

import math

import bmesh
import bpy
import numpy as np
from mathutils import Matrix, Vector
from mathutils.bvhtree import BVHTree
from mathutils.geometry import delaunay_2d_cdt

TAU = math.tau

# --------------------------------------------------------------------------
# Collections / objects
# --------------------------------------------------------------------------

_state = {"collection": None}


def set_collection(coll):
    _state["collection"] = coll


def collection():
    return _state["collection"] or bpy.context.scene.collection


def new_collection(name, parent=None):
    coll = bpy.data.collections.new(name)
    (parent or bpy.context.scene.collection).children.link(coll)
    return coll


def link(obj):
    collection().objects.link(obj)
    return obj


def empty(name, parent=None, size=0.2):
    obj = bpy.data.objects.new(name, None)
    obj.empty_display_type = "PLAIN_AXES"
    obj.empty_display_size = size
    link(obj)
    if parent is not None:
        obj.parent = parent
    return obj


def mesh_object(name, verts, faces, mat=None, smooth=True, parent=None):
    me = bpy.data.meshes.new(name)
    me.from_pydata([tuple(map(float, v)) for v in verts], [], [tuple(map(int, f)) for f in faces])
    me.validate(clean_customdata=False)
    me.update()
    obj = bpy.data.objects.new(name, me)
    link(obj)
    if mat is not None:
        me.materials.append(mat)
    set_smooth(obj, smooth)
    if parent is not None:
        obj.parent = parent
    return obj


def set_smooth(obj, smooth=True):
    me = obj.data
    if len(me.polygons):
        me.polygons.foreach_set("use_smooth", [bool(smooth)] * len(me.polygons))
    me.update()


def subsurf(obj, levels=2, render=None):
    mod = obj.modifiers.new("Subdivision", "SUBSURF")
    mod.levels = levels
    mod.render_levels = levels if render is None else render
    return mod


def depsgraph():
    bpy.context.view_layer.update()
    return bpy.context.evaluated_depsgraph_get()


def evaluated_mesh_object(src, name, mat=None):
    """Bake an object's evaluated geometry (curves, text, modifiers) into a new mesh object."""
    dg = depsgraph()
    me = bpy.data.meshes.new_from_object(src.evaluated_get(dg))
    me.name = name
    obj = bpy.data.objects.new(name, me)
    link(obj)
    if mat is not None:
        me.materials.clear()
        me.materials.append(mat)
    return obj


def remove(obj):
    data = obj.data
    bpy.data.objects.remove(obj, do_unlink=True)
    if data is not None and data.users == 0:
        if isinstance(data, bpy.types.Mesh):
            bpy.data.meshes.remove(data)
        elif isinstance(data, bpy.types.Curve):
            bpy.data.curves.remove(data)


def transform_mesh(obj, matrix):
    obj.data.transform(matrix)
    obj.data.update()
    return obj


def verts_np(obj):
    me = obj.data
    arr = np.empty(len(me.vertices) * 3, dtype=np.float64)
    me.vertices.foreach_get("co", arr)
    return arr.reshape(-1, 3)


def set_verts_np(obj, arr):
    obj.data.vertices.foreach_set("co", np.asarray(arr, dtype=np.float64).ravel())
    obj.data.update()


def join(objs, name=None):
    """Join mesh objects (all at identity transform relative to the first)."""
    objs = [o for o in objs if o is not None]
    if not objs:
        return None
    if len(objs) == 1:
        if name:
            objs[0].name = name
            objs[0].data.name = name
        return objs[0]
    target = objs[0]
    with bpy.context.temp_override(active_object=target, selected_editable_objects=objs,
                                   selected_objects=objs):
        bpy.ops.object.join()
    if name:
        target.name = name
        target.data.name = name
    return target


def apply_modifiers(obj):
    dg = depsgraph()
    me = bpy.data.meshes.new_from_object(obj.evaluated_get(dg))
    old = obj.data
    obj.modifiers.clear()
    obj.data = me
    if old.users == 0:
        bpy.data.meshes.remove(old)
    return obj


# --------------------------------------------------------------------------
# 2D curve utilities
# --------------------------------------------------------------------------

def catmull_rom(points, samples_per_seg=8, closed=False):
    """Centripetal-ish (uniform) Catmull-Rom interpolation through 2D/3D points."""
    P = np.asarray(points, dtype=np.float64)
    n = len(P)
    if n < 3:
        return P
    if closed:
        ext = np.vstack([P[-1:], P, P[:2]])
        segs = n
    else:
        ext = np.vstack([2 * P[0] - P[1], P, 2 * P[-1] - P[-2]])
        segs = n - 1
    out = []
    t = np.linspace(0.0, 1.0, samples_per_seg, endpoint=False)[:, None]
    for i in range(segs):
        p0, p1, p2, p3 = ext[i], ext[i + 1], ext[i + 2], ext[i + 3]
        a = 2 * p1
        b = p2 - p0
        c = 2 * p0 - 5 * p1 + 4 * p2 - p3
        d = -p0 + 3 * p1 - 3 * p2 + p3
        out.append(0.5 * (a + b * t + c * t * t + d * t * t * t))
    out = np.vstack(out)
    if not closed:
        out = np.vstack([out, P[-1:]])
    return out


def resample(points, spacing=None, count=None, closed=True):
    P = np.asarray(points, dtype=np.float64)
    Q = np.vstack([P, P[:1]]) if closed else P
    seg = Q[1:] - Q[:-1]
    L = np.linalg.norm(seg, axis=1)
    total = L.sum()
    if count is None:
        count = max(6, int(round(total / spacing)))
    cum = np.concatenate([[0.0], np.cumsum(L)])
    t = np.linspace(0.0, total, count, endpoint=not closed)
    idx = np.clip(np.searchsorted(cum, t, side="right") - 1, 0, len(seg) - 1)
    f = (t - cum[idx]) / np.maximum(L[idx], 1e-12)
    return Q[idx] + seg[idx] * f[:, None]


def polygon_area(P):
    P = np.asarray(P)
    x, y = P[:, 0], P[:, 1]
    return 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)


def ccw(P):
    P = np.asarray(P, dtype=np.float64)
    return P if polygon_area(P) > 0 else P[::-1].copy()


def offset_closed(P, dist):
    """Offset a smooth closed CCW polyline outward by `dist` (negative = inward)."""
    P = ccw(P)
    t = np.roll(P, -1, axis=0) - np.roll(P, 1, axis=0)
    t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-12)
    n = np.stack([t[:, 1], -t[:, 0]], axis=1)  # outward for CCW
    return P + n * dist


def points_in_polygon(pts, poly):
    pts = np.asarray(pts)
    poly = np.asarray(poly)
    x, y = pts[:, 0][:, None], pts[:, 1][:, None]
    x1, y1 = poly[:, 0][None, :], poly[:, 1][None, :]
    x2, y2 = np.roll(poly[:, 0], -1)[None, :], np.roll(poly[:, 1], -1)[None, :]
    cond = (y1 > y) != (y2 > y)
    with np.errstate(divide="ignore", invalid="ignore"):
        xint = (x2 - x1) * (y - y1) / (y2 - y1) + x1
    hit = cond & (x < xint)
    return (np.sum(hit, axis=1) % 2) == 1


def distance_to_loops(pts, loops):
    pts = np.asarray(pts, dtype=np.float64)
    best = np.full(len(pts), np.inf)
    for loop in loops:
        A = np.asarray(loop, dtype=np.float64)
        B = np.roll(A, -1, axis=0)
        AB = B - A
        denom = np.maximum(np.sum(AB * AB, axis=1), 1e-18)
        for start in range(0, len(pts), 2048):
            p = pts[start:start + 2048][:, None, :]
            t = np.clip(np.sum((p - A[None]) * AB[None], axis=2) / denom[None], 0.0, 1.0)
            proj = A[None] + t[..., None] * AB[None]
            d = np.min(np.linalg.norm(p - proj, axis=2), axis=1)
            best[start:start + 2048] = np.minimum(best[start:start + 2048], d)
    return best


def rot2(P, angle):
    c, s = math.cos(angle), math.sin(angle)
    P = np.asarray(P, dtype=np.float64)
    return P @ np.array([[c, s], [-s, c]])


# --------------------------------------------------------------------------
# Relief: inflate a 2D region (with optional holes) into a smooth embossed mesh
# --------------------------------------------------------------------------

def relief(name, outline, height, radius=None, holes=(), spacing=None, skirt=0.0,
           two_sided=False, back_height=None, rim=0.0, power=1.0, mat=None,
           max_verts=6000):
    """Create an embossed ("puffy") mesh from a closed 2D outline.

    height : peak relief height (towards -Y)
    radius : distance from the edge over which the rounded bevel rises; large
             radius -> dome/pillow, small radius -> flat plaque with round edge.
    rim    : vertical wall half-height at the outline (for chunky two-sided parts)
    skirt  : depth the open edge is pushed *into* the surface (hides seams)
    """
    outer = ccw(outline)
    hole_loops = [ccw(h)[::-1] for h in holes]
    mn, mx = outer.min(axis=0), outer.max(axis=0)
    size = float(np.max(mx - mn))
    if radius is None:
        radius = size * 0.25
    if spacing is None:
        spacing = max(size / 34.0, 0.0012)
    area = abs(polygon_area(outer)) - sum(abs(polygon_area(h)) for h in hole_loops)
    est = area / (0.866 * spacing * spacing)
    if est > max_verts:
        spacing *= math.sqrt(est / max_verts)
    loops = [resample(outer, spacing)] + [resample(h, spacing) for h in hole_loops]

    # hexagonal interior lattice
    xs = np.arange(mn[0], mx[0] + spacing, spacing)
    ys = np.arange(mn[1], mx[1] + spacing, spacing * 0.866)
    gx, gy = np.meshgrid(xs, ys)
    gx = gx + (np.arange(len(ys))[:, None] % 2) * spacing * 0.5
    grid = np.stack([gx.ravel(), gy.ravel()], axis=1)
    inside = points_in_polygon(grid, loops[0])
    for h in loops[1:]:
        inside &= ~points_in_polygon(grid, h)
    grid = grid[inside]
    if len(grid):
        grid = grid[distance_to_loops(grid, loops) > spacing * 0.45]

    pts2 = np.vstack(loops + ([grid] if len(grid) else []))
    edges = []
    base = 0
    for lp in loops:
        n = len(lp)
        edges += [(base + i, base + (i + 1) % n) for i in range(n)]
        base += n
    res = delaunay_2d_cdt([Vector(p) for p in pts2], edges, [], 0, spacing * 1e-4, False)
    coords = np.array([tuple(v) for v in res[0]], dtype=np.float64)
    tris = np.array(res[2], dtype=np.int64)
    cent = coords[tris].mean(axis=1)
    keep = points_in_polygon(cent, loops[0])
    for h in loops[1:]:
        keep &= ~points_in_polygon(cent, h)
    tris = tris[keep]
    # enforce CCW
    a, b, c = coords[tris[:, 0]], coords[tris[:, 1]], coords[tris[:, 2]]
    cross = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
    tris[cross < 0] = tris[cross < 0][:, ::-1]

    used = np.unique(tris)
    remap = -np.ones(len(coords), dtype=np.int64)
    remap[used] = np.arange(len(used))
    coords = coords[used]
    tris = remap[tris]

    d = distance_to_loops(coords, loops)
    t = np.clip(d / max(radius, 1e-9), 0.0, 1.0)
    prof = np.sqrt(np.maximum(0.0, 1.0 - (1.0 - t) ** 2)) ** power
    h_front = rim + (height - rim) * prof

    # boundary edges (used by exactly one triangle)
    e = np.vstack([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]])
    key = np.sort(e, axis=1)
    uniq, inv, counts = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    bmask = counts[inv.ravel()] == 1
    bedges = e[bmask]  # directed as in their (CCW) triangle
    is_boundary = np.zeros(len(coords), dtype=bool)
    is_boundary[bedges.ravel()] = True

    verts = [np.stack([coords[:, 0], -h_front, coords[:, 1]], axis=1)]
    faces = [tris]
    nv = len(coords)
    if two_sided:
        bh = height if back_height is None else back_height
        h_back = rim + (bh - rim) * prof
        if rim > 0:
            back_idx = np.arange(nv) + nv
            verts.append(np.stack([coords[:, 0], h_back, coords[:, 1]], axis=1))
            faces.append(back_idx[tris][:, ::-1])
            # wall between front and back rims
            a_, b_ = bedges[:, 0], bedges[:, 1]
            faces.append(np.stack([b_, a_, a_ + nv, b_ + nv], axis=1))
        else:
            interior = ~is_boundary
            back_idx = np.arange(nv)
            back_idx[interior] = nv + np.arange(interior.sum())
            verts.append(np.stack([coords[interior, 0], h_back[interior], coords[interior, 1]], axis=1))
            faces.append(back_idx[tris][:, ::-1])
    else:
        depth = skirt if skirt > 0 else 0.0
        bidx = np.where(is_boundary)[0]
        smap = -np.ones(nv, dtype=np.int64)
        smap[bidx] = nv + np.arange(len(bidx))
        verts.append(np.stack([coords[bidx, 0], np.full(len(bidx), depth), coords[bidx, 1]], axis=1))
        a_, b_ = bedges[:, 0], bedges[:, 1]
        faces.append(np.stack([b_, a_, smap[a_], smap[b_]], axis=1))

    V = np.vstack(verts)
    F = [list(f) for grp in faces for f in grp]
    return mesh_object(name, V, F, mat=mat)


# --------------------------------------------------------------------------
# Lathe / swept surfaces
# --------------------------------------------------------------------------

def superellipse(n_seg, exponent=2.0, phase=0.0):
    t = np.linspace(0.0, TAU, n_seg, endpoint=False) + phase
    c, s = np.cos(t), np.sin(t)
    e = 2.0 / exponent
    return np.sign(c) * np.abs(c) ** e, np.sign(s) * np.abs(s) ** e, t


def lathe(name, profile, segments=64, depth_ratio=1.0, exponent=2.0, smooth_samples=0,
          radial_mod=None, z_mod=None, mat=None, cap_bottom=True, cap_top=True,
          closed_profile=False, exponent_fn=None, depth_fn=None):
    """Surface of revolution from a (radius, z) profile listed bottom -> top.

    depth_ratio / depth_fn(z): front-back squash (Y scale) of the cross-section
    exponent / exponent_fn(z): superellipse exponent (2 = ellipse, 4 = rounded square)
    radial_mod(theta, z, r) -> factor ; z_mod(theta, z, r) -> dz  (for scallops)
    closed_profile: profile is a closed loop (e.g. bowl wall with thickness)
    """
    prof = np.asarray(profile, dtype=np.float64)
    if smooth_samples:
        prof = catmull_rom(prof, smooth_samples, closed=closed_profile)
    verts = []
    ring_idx = []
    for r, z in prof:
        if r <= 1e-6 and not closed_profile:
            ring_idx.append(("pole", len(verts)))
            verts.append((0.0, 0.0, z))
            continue
        ex = exponent_fn(z) if exponent_fn else exponent
        cx, cy, th = superellipse(segments, ex)
        dr = depth_fn(z) if depth_fn else depth_ratio
        rr = np.full(segments, r)
        zz = np.full(segments, z)
        if radial_mod is not None:
            rr = rr * np.array([radial_mod(a, z, r) for a in th])
        if z_mod is not None:
            zz = zz + np.array([z_mod(a, z, r) for a in th])
        start = len(verts)
        for k in range(segments):
            verts.append((rr[k] * cx[k], rr[k] * cy[k] * dr, zz[k]))
        ring_idx.append(("ring", start))
    faces = []
    count = len(ring_idx)
    pairs = [(i, i + 1) for i in range(count - 1)]
    if closed_profile:
        pairs.append((count - 1, 0))
    for i, j in pairs:
        ka, a = ring_idx[i]
        kb, b = ring_idx[j]
        if ka == "ring" and kb == "ring":
            for k in range(segments):
                k2 = (k + 1) % segments
                faces.append((a + k, a + k2, b + k2, b + k))
        elif ka == "pole" and kb == "ring":
            for k in range(segments):
                faces.append((a, b + (k + 1) % segments, b + k))
        elif ka == "ring" and kb == "pole":
            for k in range(segments):
                faces.append((a + k, a + (k + 1) % segments, b))
    if not closed_profile:
        if cap_bottom and ring_idx[0][0] == "ring":
            a = ring_idx[0][1]
            faces.append(tuple(a + k for k in reversed(range(segments))))
        if cap_top and ring_idx[-1][0] == "ring":
            a = ring_idx[-1][1]
            faces.append(tuple(a + k for k in range(segments)))
    return mesh_object(name, verts, faces, mat=mat)


def inflate(name, outline, depth, center=None, rings=12, power=0.6, back_depth=None, mat=None,
            count=96):
    """UV-sphere-like pillow built on a star-shaped 2D outline (x, z plane), bulging +/-Y."""
    P = resample(ccw(outline), count=count)
    c = np.asarray(center if center is not None else P.mean(axis=0))
    bd = depth if back_depth is None else back_depth
    lat = np.linspace(-math.pi / 2, math.pi / 2, 2 * rings + 1)[1:-1]
    verts = []
    for phi in lat:
        s = math.cos(phi) ** power
        y = (bd if phi < 0 else depth) * math.sin(phi)
        Q = c + (P - c) * s
        for q in Q:
            verts.append((q[0], -y, q[1]))
    n = len(P)
    faces = []
    for i in range(len(lat) - 1):
        for k in range(n):
            a = i * n + k
            b = i * n + (k + 1) % n
            faces.append((a, b, b + n, a + n))
    back_pole = len(verts)
    verts.append((c[0], bd, c[1]))
    front_pole = len(verts)
    verts.append((c[0], -depth, c[1]))
    for k in range(n):
        faces.append((back_pole, (k + 1) % n, k))
        last = (len(lat) - 1) * n
        faces.append((last + k, last + (k + 1) % n, front_pole))
    obj = mesh_object(name, verts, faces, mat=mat)
    return obj


def _frames(P, closed=False):
    n = len(P)
    if closed:
        T = np.roll(P, -1, axis=0) - np.roll(P, 1, axis=0)
    else:
        T = np.gradient(P, axis=0)
    T /= np.maximum(np.linalg.norm(T, axis=1, keepdims=True), 1e-12)
    ref = np.array([0.0, 0.0, 1.0]) if abs(T[0][2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    N = np.zeros_like(P)
    N[0] = np.cross(T[0], ref)
    N[0] /= np.linalg.norm(N[0])
    for i in range(1, n):
        v = np.cross(T[i - 1], T[i])
        s = np.linalg.norm(v)
        if s < 1e-10:
            N[i] = N[i - 1]
        else:
            c = np.clip(np.dot(T[i - 1], T[i]), -1, 1)
            ang = math.atan2(s, c)
            R = Matrix.Rotation(ang, 3, Vector(v / s))
            N[i] = np.array(R @ Vector(N[i - 1]))
        N[i] -= T[i] * np.dot(N[i], T[i])
        N[i] /= max(np.linalg.norm(N[i]), 1e-12)
    B = np.cross(T, N)
    return T, N, B


def tube(name, points, radius, sides=10, closed=False, radii=None, caps="round", mat=None,
         flatten=1.0, up=None):
    """Sweep a circular (or flattened elliptical) cross-section along a 3D polyline."""
    P = np.asarray(points, dtype=np.float64)
    n = len(P)
    R = np.full(n, float(radius)) if radii is None else np.asarray(radii, dtype=np.float64)
    T, N, B = _frames(P, closed)
    if up is not None:
        # orient the flattened axis using a per-point "up" (surface normal) vector
        U = np.asarray(up, dtype=np.float64)
        if U.ndim == 1:
            U = np.tile(U, (n, 1))
        B = U - T * np.sum(U * T, axis=1, keepdims=True)
        B /= np.maximum(np.linalg.norm(B, axis=1, keepdims=True), 1e-12)
        N = np.cross(B, T)
    ang = np.linspace(0, TAU, sides, endpoint=False)
    verts = []
    for i in range(n):
        for a in ang:
            verts.append(P[i] + R[i] * (math.cos(a) * N[i] + flatten * math.sin(a) * B[i]))
    faces = []
    last = n if closed else n - 1
    for i in range(last):
        j = (i + 1) % n
        for k in range(sides):
            k2 = (k + 1) % sides
            faces.append((i * sides + k, i * sides + k2, j * sides + k2, j * sides + k))
    if not closed:
        for end, sign in ((0, -1.0), (n - 1, 1.0)):
            ring0 = end * sides
            if caps == "round":
                steps = 3
                prev = list(range(ring0, ring0 + sides))
                for st in range(1, steps + 1):
                    phi = st / (steps + 1) * math.pi / 2
                    base = len(verts)
                    for a in ang:
                        off = R[end] * math.cos(phi) * (math.cos(a) * N[end] + flatten * math.sin(a) * B[end])
                        verts.append(P[end] + off + T[end] * sign * R[end] * math.sin(phi))
                    cur = list(range(base, base + sides))
                    for k in range(sides):
                        k2 = (k + 1) % sides
                        f = (prev[k], prev[k2], cur[k2], cur[k])
                        faces.append(f if sign > 0 else f[::-1])
                    prev = cur
                tip = len(verts)
                verts.append(P[end] + T[end] * sign * R[end])
                for k in range(sides):
                    f = (prev[k], prev[(k + 1) % sides], tip)
                    faces.append(f if sign > 0 else f[::-1])
            elif caps == "flat":
                f = tuple(range(ring0, ring0 + sides))
                faces.append(f if sign > 0 else f[::-1])
    return mesh_object(name, verts, faces, mat=mat)


def uv_sphere(name, radius=1.0, segments=24, rings=12, scale=(1, 1, 1), location=(0, 0, 0), mat=None):
    verts = [(0, 0, -radius)]
    for i in range(1, rings):
        phi = -math.pi / 2 + math.pi * i / rings
        for k in range(segments):
            th = TAU * k / segments
            verts.append((radius * math.cos(phi) * math.cos(th), radius * math.cos(phi) * math.sin(th),
                          radius * math.sin(phi)))
    verts.append((0, 0, radius))
    faces = []
    for k in range(segments):
        faces.append((0, 1 + (k + 1) % segments, 1 + k))
    for i in range(rings - 2):
        a = 1 + i * segments
        b = a + segments
        for k in range(segments):
            k2 = (k + 1) % segments
            faces.append((a + k, a + k2, b + k2, b + k))
    top = len(verts) - 1
    a = 1 + (rings - 2) * segments
    for k in range(segments):
        faces.append((a + k, a + (k + 1) % segments, top))
    V = np.array(verts) * np.array(scale) + np.array(location)
    return mesh_object(name, V, faces, mat=mat)


def torus(name, major, minor, seg=32, ring=10, mat=None):
    verts, faces = [], []
    for i in range(seg):
        u = TAU * i / seg
        for j in range(ring):
            v = TAU * j / ring
            r = major + minor * math.cos(v)
            verts.append((r * math.cos(u), r * math.sin(u), minor * math.sin(v)))
    for i in range(seg):
        for j in range(ring):
            a = i * ring + j
            b = ((i + 1) % seg) * ring + j
            c = ((i + 1) % seg) * ring + (j + 1) % ring
            d = i * ring + (j + 1) % ring
            faces.append((a, b, c, d))
    return mesh_object(name, verts, faces, mat=mat)


# --------------------------------------------------------------------------
# Surface projection
# --------------------------------------------------------------------------

class Surface:
    """World-space BVH over the evaluated geometry of one or more objects."""

    def __init__(self, objs):
        dg = depsgraph()
        verts, polys = [], []
        for obj in objs:
            ev = obj.evaluated_get(dg)
            me = ev.to_mesh()
            mw = obj.matrix_world
            base = len(verts)
            verts.extend(mw @ v.co for v in me.vertices)
            polys.extend([base + i for i in p.vertices] for p in me.polygons)
            ev.to_mesh_clear()
        self.bvh = BVHTree.FromPolygons(verts, polys)

    def hit(self, origin, direction):
        loc, nor, _, _ = self.bvh.ray_cast(origin, direction)
        if loc is None:
            loc, nor, _, _ = self.bvh.find_nearest(origin + direction * 10.0)
        if nor.dot(direction) > 0:
            nor = -nor
        return loc, nor

    def radial(self, theta, z, axis=(0.0, 0.0), inward=True):
        """Surface point hit by a horizontal ray aimed at the vertical axis."""
        d = Vector((math.sin(theta), -math.cos(theta), 0.0))  # theta = 0 -> front (-Y)
        origin = Vector((axis[0], axis[1], z)) + d * 10.0
        return self.hit(origin, -d)


def frame_matrix(theta=0.0, axis=(0.0, 0.0)):
    return Matrix.Translation((axis[0], axis[1], 0.0)) @ Matrix.Rotation(theta, 4, "Z")


def wrap(obj, surf, u=0.0, v=0.0, theta=0.0, lift=0.0, rot=0.0, scale=1.0, axis=(0.0, 0.0),
         tilt=0.0):
    """Project a decal-space mesh onto `surf`.

    The decal's (x, z) plane is placed at horizontal offset `u` and height `v`,
    rotated `theta` around the vertical axis (0 = front). Rays travel along the
    rotated +Y direction; relief height (-y) is re-applied along the surface normal.
    `tilt` pitches the projection direction up/down (for shoulders/domes).
    """
    V = verts_np(obj)
    x, depth, z = V[:, 0] * scale, -V[:, 1] * scale, V[:, 2] * scale
    if rot:
        c, s = math.cos(rot), math.sin(rot)
        x, z = x * c - z * s, x * s + z * c
    O, X, Z, D = _decal_frame(u, v, theta, axis, tilt)
    out = np.empty_like(V)
    for i in range(len(V)):
        base = O + X * x[i] + Z * z[i]
        loc, nor = surf.hit(base - D * 10.0, D)
        p = loc + nor * (lift + depth[i])
        out[i] = (p.x, p.y, p.z)
    set_verts_np(obj, out)
    return obj


def _decal_frame(u, v, theta, axis, tilt):
    """Origin, right, up and projection direction of a decal frame. tilt > 0 aims the
    projection downwards (decal plane leans back) for domes and shoulders."""
    M = frame_matrix(theta, axis)
    M3 = M.to_3x3()
    Rt = Matrix.Rotation(-tilt, 3, "X")
    X = (M3 @ Vector((1.0, 0.0, 0.0))).normalized()
    Z = (M3 @ (Rt @ Vector((0.0, 0.0, 1.0)))).normalized()
    D = (M3 @ (Rt @ Vector((0.0, 1.0, 0.0)))).normalized()
    O = M @ Vector((u, 0.0, v))
    return O, X, Z, D


def surface_path(surf, pts2d, lift=0.0, theta=0.0, axis=(0.0, 0.0), tilt=0.0):
    """Project a 2D decal-space polyline onto the surface; returns (points, normals)."""
    O, X, Z, D = _decal_frame(0.0, 0.0, theta, axis, tilt)
    P, N = [], []
    for x, z in pts2d:
        base = O + X * x + Z * z
        loc, nor = surf.hit(base - D * 10.0, D)
        P.append(tuple(loc + nor * lift))
        N.append(tuple(nor))
    return np.array(P), np.array(N)


def surface_path_radial(surf, theta_z, lift=0.0, axis=(0.0, 0.0)):
    """Project (theta, z) samples radially onto the surface (for wrap-around vines)."""
    P, N = [], []
    for th, z in theta_z:
        loc, nor = surf.radial(th, z, axis)
        P.append(tuple(loc + nor * lift))
        N.append(tuple(nor))
    return np.array(P), np.array(N)


# --------------------------------------------------------------------------
# Faceted gems (flat shaded)
# --------------------------------------------------------------------------

def faceted_outline_gem(name, outline, crown=0.3, pavilion=0.5, table=0.55, girdle=0.03,
                        pavilion_tip=0.15, mat=None, facet_points=None, center=None):
    """Faceted stone from a 2D outline lying in XY, crown towards +Z.

    The crown rises from the girdle to an inset table; the pavilion falls to a
    small culet outline. Alternating girdle/table vertices produce kite facets.
    """
    P = resample(ccw(outline), count=facet_points or 24)
    c = np.asarray(center if center is not None else P.mean(axis=0))
    n = len(P)
    mid = (P + np.roll(P, -1, axis=0)) * 0.5
    gird = np.empty((2 * n, 2))
    gird[0::2] = P
    gird[1::2] = mid
    size = float(np.max(P.max(axis=0) - P.min(axis=0)))
    verts, faces = [], []
    # girdle top ring (2n) and bottom ring (2n)
    for q in gird:
        verts.append((q[0], q[1], girdle * size * 0.5))
    for q in gird:
        verts.append((q[0], q[1], -girdle * size * 0.5))
    tab = c + (mid - c) * table
    t0 = len(verts)
    for q in tab:
        verts.append((q[0], q[1], crown * size))
    culet = c + (mid - c) * pavilion_tip
    c0 = len(verts)
    for q in culet:
        verts.append((q[0], q[1], -pavilion * size))
    # crown facets: each table vertex (at mid of edge k) connects to girdle mid (2k+1)
    # and girdle corners 2k, 2k+2
    for k in range(n):
        gk, gm, gn = 2 * k, 2 * k + 1, (2 * k + 2) % (2 * n)
        tk, tn = t0 + k, t0 + (k + 1) % n
        faces.append((gk, gm, tk))
        faces.append((gm, gn, tk))
        faces.append((gn, tn, tk))
    faces.append(tuple(t0 + k for k in range(n)))
    # girdle band
    for k in range(2 * n):
        k2 = (k + 1) % (2 * n)
        faces.append((2 * n + k, 2 * n + k2, k2, k))
    # pavilion facets
    for k in range(n):
        gk, gm, gn = 2 * n + 2 * k, 2 * n + 2 * k + 1, 2 * n + (2 * k + 2) % (2 * n)
        ck, cn = c0 + k, c0 + (k + 1) % n
        faces.append((gm, gk, ck))
        faces.append((gn, gm, ck))
        faces.append((cn, gn, ck))
    faces.append(tuple(c0 + k for k in reversed(range(n))))
    obj = mesh_object(name, verts, faces, mat=mat, smooth=False)
    return obj


def faceted_lathe(name, profile, segments=8, depth_ratio=1.0, mat=None, twist=True):
    """Crystal-like solid of revolution with flat facets. Alternate rings are rotated
    half a step so adjacent bands form triangular / kite facets. profile: (r, z) bottom->top."""
    verts, idx = [], []
    for i, (r, z) in enumerate(profile):
        if r <= 1e-6:
            idx.append(("pole", len(verts)))
            verts.append((0.0, 0.0, z))
            continue
        off = (math.pi / segments) * (i % 2) if twist else 0.0
        start = len(verts)
        for k in range(segments):
            a = TAU * k / segments + off
            verts.append((r * math.sin(a), -r * math.cos(a) * depth_ratio, z))
        idx.append(("ring", start))
    faces = []
    for i in range(len(idx) - 1):
        ka, a = idx[i]
        kb, b = idx[i + 1]
        if ka == "ring" and kb == "ring":
            shifted = twist and (i % 2 == 0)
            for k in range(segments):
                k2 = (k + 1) % segments
                if shifted:  # upper ring offset by +half step
                    faces.append((a + k, a + k2, b + k))
                    faces.append((a + k2, b + k2, b + k))
                else:  # lower ring offset
                    faces.append((a + k, b + k2, b + k))
                    faces.append((a + k, a + k2, b + k2))
        elif ka == "pole":
            for k in range(segments):
                faces.append((a, b + (k + 1) % segments, b + k))
        elif kb == "pole":
            for k in range(segments):
                faces.append((a + k, a + (k + 1) % segments, b))
    if idx[0][0] == "ring":
        faces.append(tuple(idx[0][1] + k for k in reversed(range(segments))))
    if idx[-1][0] == "ring":
        faces.append(tuple(idx[-1][1] + k for k in range(segments)))
    return mesh_object(name, verts, faces, mat=mat, smooth=False)


def ridge_star(name, r_outer, r_inner, height, points=5, table=0.0, back=0.0, wall=0.0, mat=None,
               smooth=False, rotation=0.0):
    """Classic 3D faceted star: ridges from the centre to each tip (XY plane, +Z up).
    `wall` adds a straight side band of that thickness around the outline."""
    ring = []
    for i in range(points * 2):
        a = math.pi / 2 + rotation + math.pi * i / points
        r = r_outer if i % 2 == 0 else r_inner
        ring.append((r * math.cos(a), r * math.sin(a)))
    m = len(ring)
    hw = wall * 0.5
    verts = [(q[0], q[1], hw) for q in ring]
    faces = []
    if wall > 0:
        verts += [(q[0], q[1], -hw) for q in ring]
        for k in range(m):
            k2 = (k + 1) % m
            faces.append((m + k, m + k2, k2, k))
    back_ring = m if wall > 0 else 0
    if table > 0:
        t0 = len(verts)
        verts += [(q[0] * table, q[1] * table, hw + height) for q in ring]
        for k in range(m):
            k2 = (k + 1) % m
            faces.append((k, k2, t0 + k2, t0 + k))
        faces.append(tuple(t0 + k for k in range(m)))
    else:
        top = len(verts)
        verts.append((0.0, 0.0, hw + height))
        for k in range(m):
            faces.append((k, (k + 1) % m, top))
    if back > 0:
        bot = len(verts)
        verts.append((0.0, 0.0, -hw - back))
        for k in range(m):
            faces.append((back_ring + (k + 1) % m, back_ring + k, bot))
    else:
        faces.append(tuple(back_ring + k for k in reversed(range(m))))
    return mesh_object(name, verts, faces, mat=mat, smooth=smooth)


def face_front(obj):
    """Rotate a mesh authored in XY (+Z up) so +Z faces the camera (-Y) and +Y becomes up."""
    return transform_mesh(obj, Matrix.Rotation(math.pi / 2, 4, "X"))


def nugget_gem(name, radius, seed, points=18, squash=(1.0, 1.0, 0.8), mat=None):
    rng = np.random.default_rng(seed)
    pts = rng.normal(size=(points, 3))
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    pts *= radius * (0.85 + 0.15 * rng.random((points, 1)))
    pts *= np.array(squash)
    bm = bmesh.new()
    for p in pts:
        bm.verts.new(p)
    bmesh.ops.convex_hull(bm, input=bm.verts)
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    obj = bpy.data.objects.new(name, me)
    link(obj)
    if mat is not None:
        me.materials.append(mat)
    set_smooth(obj, False)
    return obj


# --------------------------------------------------------------------------
# Text
# --------------------------------------------------------------------------

_fonts = {}


def load_font(path):
    if path not in _fonts:
        _fonts[path] = bpy.data.fonts.load(path, check_existing=True)
    return _fonts[path]


def text_relief(name, text, font_path, size, depth, bevel, line_spacing=0.95, spacing=1.0,
                mat=None, shear=0.0):
    """3D text in decal space (XZ plane, extruding towards -Y, back face at y = 0)."""
    cu = bpy.data.curves.new(name + "_curve", "FONT")
    cu.body = text
    cu.font = load_font(font_path)
    cu.size = size
    cu.align_x = "CENTER"
    cu.align_y = "CENTER"
    cu.space_line = line_spacing
    cu.space_character = spacing
    cu.shear = shear
    cu.extrude = max(depth * 0.5 - bevel, 0.0)
    cu.bevel_depth = bevel
    cu.offset = -bevel  # bevel grows strokes outwards; pull back so glyphs keep the photo's medium weight
    cu.bevel_resolution = 2
    cu.resolution_u = 6
    tmp = bpy.data.objects.new(name + "_tmp", cu)
    link(tmp)
    obj = evaluated_mesh_object(tmp, name, mat=mat)
    remove(tmp)
    V = verts_np(obj)
    zmin = V[:, 2].min()
    # (x, y, z) text -> decal (x, -(z - zmin), y)
    out = np.stack([V[:, 0], -(V[:, 2] - zmin), V[:, 1]], axis=1)
    set_verts_np(obj, out)
    set_smooth(obj, True)
    obj.data.update()
    return obj
