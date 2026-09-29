"""Reusable decorations: embossed decals (flowers, leaves, vines, labels, butterflies)
and 3D hardware (collars, charms, toppers). Units are reference-image pixels
converted with `px()` so every dimension traces back to the source photo."""

import math

import numpy as np
from mathutils import Matrix, Vector

from . import core as C
from . import materials as M
from . import shapes2d as S

PX = 1.0 / 300.0  # BU per reference pixel (1 BU = 10 cm)


def px(v):
    return np.asarray(v, dtype=np.float64) * PX if not np.isscalar(v) else v * PX


FONT_LABEL = None
FONT_TITLE = None


def set_fonts(label, title):
    global FONT_LABEL, FONT_TITLE
    FONT_LABEL, FONT_TITLE = label, title


# --------------------------------------------------------------------------
# Decal groups
# --------------------------------------------------------------------------

class Decal:
    """A set of decal-space meshes sharing one placement on a surface."""

    def __init__(self):
        self.items = []  # (obj, extra_lift)

    def add(self, obj, lift=0.0, offset=(0.0, 0.0), rot=0.0, scale=1.0):
        if rot or scale != 1.0 or offset != (0.0, 0.0):
            V = C.verts_np(obj)
            x, z = V[:, 0] * scale, V[:, 2] * scale
            c, s = math.cos(rot), math.sin(rot)
            x, z = x * c - z * s, x * s + z * c
            V[:, 0], V[:, 2] = x + offset[0], z + offset[1]
            V[:, 1] *= scale
            C.set_verts_np(obj, V)
        self.items.append((obj, lift))
        return obj

    def extend(self, other, lift=0.0, offset=(0.0, 0.0), rot=0.0, scale=1.0):
        for obj, l in other.items:
            self.add(obj, l + lift, offset, rot, scale)
        return self

    def place(self, surf, u=0.0, v=0.0, theta=0.0, lift=0.0, rot=0.0, scale=1.0, parent=None,
              axis=(0.0, 0.0), tilt=0.0):
        out = []
        for obj, l in self.items:
            C.wrap(obj, surf, u=u, v=v, theta=theta, lift=lift + l, rot=rot, scale=scale, axis=axis,
                   tilt=tilt)
            if parent is not None:
                obj.parent = parent
            out.append(obj)
        return out

    def flat(self, matrix, parent=None):
        """Place on a planar surface: decal space transformed by `matrix` (no projection)."""
        for obj, l in self.items:
            V = C.verts_np(obj)
            V[:, 1] -= l
            C.set_verts_np(obj, V)
            C.transform_mesh(obj, matrix)
            if parent is not None:
                obj.parent = parent
        return [o for o, _ in self.items]


def emboss(outline, height, mat, radius=None, name="Emboss", **kw):
    return C.relief(name, outline, height, radius=radius, mat=mat, skirt=kw.pop("skirt", 0.004), **kw)


# --------------------------------------------------------------------------
# Botanical decals
# --------------------------------------------------------------------------

def flower(size, petal_mat, center_mat, petals=5, inner_mat=None, height=None, name="Flower",
           style="round", rotation=0.0):
    """Embossed flower of diameter `size` (BU). style: 'round' (pink blossom) or 'daisy'."""
    d = Decal()
    h = height if height is not None else size * 0.16
    r = size / 2
    if style == "daisy":
        for i in range(petals):
            a = rotation + i * math.tau / petals
            p = S.petal(r * 1.0, r * 0.62)
            p = C.rot2(p, a - math.pi / 2)
            d.add(C.relief(f"{name}_Petal", p, h, radius=h * 1.8, mat=petal_mat, skirt=0.003))
        cen = S.circle(r * 0.36, 32)
        d.add(C.relief(f"{name}_Center", cen, h * 0.9, radius=r * 0.36, mat=center_mat, skirt=0.003), lift=h * 0.55)
    else:
        outline = S.flower(petals, r, r * 0.38, sharp=0.5, rotation=rotation + math.pi / 2)
        d.add(C.relief(f"{name}_Petals", outline, h, radius=r * 0.35, mat=petal_mat, skirt=0.003))
        if inner_mat is not None:
            inner = S.flower(petals, r * 0.55, r * 0.3, sharp=0.6, rotation=rotation + math.pi / 2)
            d.add(C.relief(f"{name}_Inner", inner, h * 0.25, radius=r * 0.2, mat=inner_mat, skirt=0.002),
                  lift=h * 0.8)
        cen = S.circle(r * 0.27, 28)
        d.add(C.relief(f"{name}_Center", cen, h * 0.7, radius=r * 0.27, mat=center_mat, skirt=0.002), lift=h * 0.85)
    return d


def leaf(length, width, mat, bend=0.0, height=None, name="Leaf", vein_mat=None):
    d = Decal()
    h = height if height is not None else width * 0.22
    d.add(C.relief(name, S.leaf(length, width, bend=bend), h, radius=width * 0.45, mat=mat, skirt=0.003))
    return d


def sprig(stem_pts, leaf_len, leaf_w, stem_r, mat_stem, mat_leaf, every=2, start=1, name="Sprig",
          flip=True, leaf_angle=0.9):
    """Stem (flat embossed tube in decal space) with alternating leaves along it."""
    d = Decal()
    P = C.catmull_rom(np.asarray(stem_pts), 8)
    d.add(stem_relief(P, stem_r, mat_stem, name=f"{name}_Stem"))
    n = len(stem_pts)
    side = 1
    for i in range(start, n, every):
        j = min(i * 8, len(P) - 2)
        tdir = P[j + 1] - P[j]
        a = math.atan2(tdir[1], tdir[0])
        la = a + side * leaf_angle - math.pi / 2
        lf = S.leaf(leaf_len, leaf_w, bend=0.12 * side)
        lf = C.rot2(lf, la) + P[j]
        d.add(C.relief(f"{name}_Leaf", lf, leaf_w * 0.22, radius=leaf_w * 0.45, mat=mat_leaf, skirt=0.003))
        if flip:
            side = -side
    return d


def stem_relief(P, r, mat, name="Stem"):
    """Flat stroke outline around an open polyline (for stems / filigree in decal space)."""
    P = np.asarray(P, dtype=np.float64)
    T = np.gradient(P, axis=0)
    T /= np.maximum(np.linalg.norm(T, axis=1, keepdims=True), 1e-12)
    N = np.stack([-T[:, 1], T[:, 0]], axis=1)
    left = P + N * r
    right = P - N * r
    phis_end = np.linspace(-math.pi / 2, math.pi / 2, 9)[1:-1]
    phis_start = np.linspace(math.pi / 2, 3 * math.pi / 2, 9)[1:-1]
    cap_end = [P[-1] + r * (math.cos(f) * T[-1] + math.sin(f) * N[-1]) for f in phis_end]
    cap_start = [P[0] + r * (math.cos(f) * T[0] + math.sin(f) * N[0]) for f in phis_start]
    outline = np.vstack([right, np.array(cap_end), left[::-1], np.array(cap_start)])
    return C.relief(name, outline, r * 0.9, radius=r, mat=mat, skirt=0.002, spacing=r * 0.5)


def curl_path(start, heading, length, curl_r, turns=1.1, handed=1, n=50):
    return S.scroll(start, heading, length, curl_r, turns=turns, handed=handed, n=n)


def surface_vine(surf, pts2d, radius, mat, parent=None, lift_frac=0.45, name="Vine", theta=0.0,
                 radial=False, taper=True, flatten=0.7):
    """Tube following the surface. pts2d are decal (x, z) or (theta, z) if radial."""
    pts2d = np.asarray(pts2d)
    if radial:
        P, N = C.surface_path_radial(surf, pts2d, lift=radius * lift_frac)
    else:
        P, N = C.surface_path(surf, pts2d, lift=radius * lift_frac, theta=theta)
    radii = None
    if taper:
        s = np.linspace(0, 1, len(P))
        radii = radius * (1.0 - 0.55 * s ** 1.5)
    obj = C.tube(name, P, radius, sides=8, radii=radii, mat=mat, flatten=flatten, up=N)
    if parent is not None:
        obj.parent = parent
    return obj


def butterfly(size, wing_mat, inner_mat, body_mat, height=None, name="Butterfly", open_=1.0):
    """Front-facing embossed butterfly, `size` = wingspan (BU)."""
    d = Decal()
    h = height if height is not None else size * 0.09
    s = size / 2
    for side in (1, -1):
        up = S.wing_lobe(s * 0.98, s * 0.95, angle=math.radians(38) * open_)
        lo = S.wing_lobe(s * 0.72, s * 0.62, angle=-math.radians(42))
        for lobe, nm in ((up, "Up"), (lo, "Low")):
            L = lobe.copy()
            L[:, 0] *= side
            d.add(C.relief(f"{name}_Wing{nm}", C.ccw(L), h, radius=h * 1.4, mat=wing_mat, skirt=0.003))
            if inner_mat is not None:
                inner = C.offset_closed(C.ccw(L), -s * 0.1)
                cen = inner.mean(axis=0)
                inner = cen + (inner - cen) * 0.8
                d.add(C.relief(f"{name}_Inner{nm}", inner, h * 0.25, radius=h * 0.6, mat=inner_mat, skirt=0.002),
                      lift=h * 0.7)
    body = S.ellipse(s * 0.1, s * 0.55, 24, center=(0, -s * 0.05))
    d.add(C.relief(f"{name}_Body", body, h * 1.2, radius=s * 0.1, mat=body_mat, skirt=0.003), lift=h * 0.3)
    return d


def sparkle_star(size, mat, height=None, name="Star", rounding=0.25):
    h = height if height is not None else size * 0.14
    d = Decal()
    d.add(C.relief(name, S.star(size / 2, size / 2 * 0.5, rounding=rounding), h, radius=size * 0.2, mat=mat,
                   skirt=0.003))
    return d


def heart_decal(size, mat, height=None, name="Heart"):
    h = height if height is not None else size * 0.2
    d = Decal()
    d.add(C.relief(name, S.heart(size, size * 0.88), h, radius=size * 0.35, mat=mat, skirt=0.003))
    return d


def dots(surf, pts2d, r, mat, parent=None, theta=0.0, name="Dot", lift_frac=0.25, squash=0.6):
    out = []
    P, N = C.surface_path(surf, pts2d, lift=r * lift_frac, theta=theta)
    for p, n in zip(P, N):
        o = C.uv_sphere(name, r, 12, 7, scale=(1, 1, squash), mat=mat)
        q = Vector((0, 0, 1)).rotation_difference(Vector(n))
        C.transform_mesh(o, Matrix.Translation(Vector(p)) @ q.to_matrix().to_4x4())
        if parent is not None:
            o.parent = parent
        out.append(o)
    return out


# --------------------------------------------------------------------------
# Labels
# --------------------------------------------------------------------------

def label(surf, parent, text, text_hex, w, h, u, v, theta=0.0, font_size=None, shape=None,
          cartouche_kw=None, rim_r=None, plaque_h=None, margin=None, text_offset=(0.0, 0.0),
          name="Label", line_spacing=0.92, text_depth=None, tilt=0.0):
    """Cream plaque + rounded gold rim + raised lettering, projected onto `surf`.
    w, h, u, v in BU (u, v = label centre in decal frame)."""
    gold = M.gold()
    cream = M.cream()
    ph = plaque_h if plaque_h is not None else h * 0.03
    rim_r = rim_r if rim_r is not None else min(w, h) * 0.03
    margin = margin if margin is not None else rim_r * 1.6
    outline = shape if shape is not None else S.cartouche(w, h, **(cartouche_kw or {}))
    outer = C.offset_closed(outline, margin)
    d = Decal()
    d.add(C.relief(f"{name}_Plaque", outer, ph, radius=ph * 2.0, mat=cream, skirt=0.006,
                   spacing=min(w, h) / 45))
    d.place(surf, u=u, v=v, theta=theta, parent=parent, tilt=tilt)
    rim_pts = C.resample(outline, count=240) + np.array([u, v])
    P, N = C.surface_path(surf, rim_pts, lift=ph * 0.85 + rim_r * 0.35, theta=theta, tilt=tilt)
    rim = C.tube(f"{name}_Rim", P, rim_r, sides=8, closed=True, mat=gold, flatten=0.85, up=N)
    rim.parent = parent
    objs = []
    if text:
        # satin, not glossy: bevel highlights on small glyphs smear the letter shapes
        mat = M.plastic(f"M_FairyKit_Text_{text_hex.strip('#')}", text_hex, rough=0.5, coat=0.0,
                        variation=0.0, bump=0.0)
        fs = font_size if font_size is not None else h * 0.26
        td = text_depth if text_depth is not None else fs * 0.1
        t = C.text_relief(f"{name}_Text", text, FONT_LABEL, fs, td, td * 0.3, line_spacing=line_spacing,
                          mat=mat)
        dt = Decal()
        dt.add(t, offset=text_offset)
        objs += dt.place(surf, u=u, v=v, theta=theta, lift=ph * 0.9, parent=parent, tilt=tilt)
    return objs


# --------------------------------------------------------------------------
# 3D hardware
# --------------------------------------------------------------------------

def collar(name, r, height, z0, mat, lip=None, beads=0, bead_r=None, bead_z=None, depth_ratio=1.0,
           flare=0.0, parent=None, segments=64, bead_mat=None, taper=0.0):
    """Rounded band ring (bottle neck collar)."""
    lip = lip if lip is not None else height * 0.18
    rb = r + flare
    prof = [(0.0, z0), (rb - lip * 0.6, z0), (rb, z0 + lip * 0.5), (rb + lip * 0.15, z0 + lip),
            (r + lip * 0.1 - taper * 0.5, z0 + height * 0.5), (r + lip * 0.2 - taper, z0 + height - lip),
            (r - taper, z0 + height - lip * 0.2), (r - lip - taper, z0 + height), (0.0, z0 + height)]
    obj = C.lathe(name, prof, segments=segments, depth_ratio=depth_ratio, mat=mat, smooth_samples=4)
    C.subsurf(obj, 1)
    if parent is not None:
        obj.parent = parent
    out = [obj]
    if beads:
        bead_r = bead_r if bead_r is not None else height * 0.12
        bz = z0 + height * 0.5 if bead_z is None else bead_z
        rr = r + lip * 0.2 - taper * 0.5 + bead_r * 0.25
        for i in range(beads):
            a = math.tau * i / beads
            o = C.uv_sphere(f"{name}_Bead", bead_r, 12, 7, mat=bead_mat or mat,
                            location=(rr * math.sin(a), -rr * math.cos(a) * depth_ratio, bz))
            if parent is not None:
                o.parent = parent
            out.append(o)
    return out


def jump_ring(name, r, thick, location, rot_z=0.0, rot_x=0.0, mat=None, parent=None):
    o = C.torus(name, r, thick, 20, 8, mat=mat)
    C.transform_mesh(o, Matrix.Translation(location) @ Matrix.Rotation(rot_z, 4, "Z") @ Matrix.Rotation(
        math.pi / 2 + rot_x, 4, "X"))
    if parent is not None:
        o.parent = parent
    return o


def two_sided(name, outline, thickness, mat, radius=None, holes=(), rim=0.0, spacing=None):
    """Free-standing puffy plate in the XZ plane (thickness along Y)."""
    return C.relief(name, outline, thickness / 2, radius=radius if radius else thickness / 2, mat=mat,
                    holes=holes, two_sided=True, rim=rim, spacing=spacing)


def heart_charm(name, size, gem_mat, location, parent=None, tilt=0.0, ring_r=None):
    """Gold-framed puffy heart hanging from a jump ring. location = top of the ring."""
    gold = M.gold()
    out = []
    frame_o = S.heart(size, size * 0.9)
    frame_i = S.heart(size * 0.74, size * 0.64)
    frame_i = frame_i + np.array([0.0, -size * 0.03])
    fr = two_sided(f"{name}_Frame", frame_o, size * 0.16, gold, radius=size * 0.07, holes=[frame_i])
    gem = C.relief(f"{name}_Gem", S.heart(size * 0.8, size * 0.7) + np.array([0.0, -size * 0.03]), size * 0.11,
                   radius=size * 0.3, mat=gem_mat, two_sided=True)
    ring_r = ring_r if ring_r is not None else size * 0.13
    bail = C.torus(f"{name}_Bail", ring_r, ring_r * 0.32, 20, 8, mat=gold)
    C.transform_mesh(bail, Matrix.Translation((0, 0, size * 0.45 + ring_r * 0.8)) @ Matrix.Rotation(
        math.pi / 2, 4, "Y"))
    grp = [fr, gem, bail]
    top = size * 0.45 + ring_r * 1.8
    mat4 = Matrix.Translation(location) @ Matrix.Rotation(tilt, 4, "Z") @ Matrix.Translation((0, 0, -top))
    for o in grp:
        C.transform_mesh(o, mat4)
        if parent is not None:
            o.parent = parent
        out.append(o)
    return out


def star_gold(name, size, thickness, mat, rounding=0.3):
    outline = S.star(size / 2, size / 2 * 0.5, rounding=rounding)
    return two_sided(name, outline, thickness, mat, radius=thickness * 0.5)


def place(objs, matrix, parent=None):
    for o in objs:
        C.transform_mesh(o, matrix)
        if parent is not None:
            o.parent = parent
    return objs
