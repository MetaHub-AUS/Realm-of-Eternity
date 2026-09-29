"""The nine potion bottles/jars.

Every dimension is measured on the reference photo in pixels: profiles are
(radius, height-above-base) pairs, decals are (u, v) = (offset from the bottle's
centre line, height above its base). `px()` converts to Blender units.
"""

import math

import numpy as np
from mathutils import Matrix, Vector

from . import core as C
from . import decor as D
from . import materials as M
from . import shapes2d as S
from .decor import px


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _profile(pts):
    return [(px(r), px(h)) for r, h in pts]


def _body(name, pts, mat, depth_ratio=1.0, exponent=2.0, segments=72, parent=None, samples=6, **kw):
    obj = C.lathe(name, _profile(pts), segments=segments, depth_ratio=depth_ratio, exponent=exponent,
                  smooth_samples=samples, mat=mat, **kw)
    C.subsurf(obj, 2)
    obj.parent = parent
    return obj


def _glossy(name, hexcolor, sparkle=0.25):
    return M.plastic(f"M_FairyKit_{name}", hexcolor, rough=0.3, coat=0.4, sparkle=sparkle)


def palette():
    return dict(
        gold=M.gold(),
        cream=M.cream(),
        white=M.plastic("M_FairyKit_PetalWhite", "#FFF7F2", rough=0.35, coat=0.5, variation=0.02),
        yellow=M.plastic("M_FairyKit_FlowerCenter", "#FFC21F", rough=0.3, coat=0.6, variation=0.03),
        leaf=M.plastic("M_FairyKit_Leaf", "#5DB236", rough=0.32, coat=0.6, variation=0.05),
        leaf_dark=M.plastic("M_FairyKit_LeafDark", "#3F9C38", rough=0.32, coat=0.6, variation=0.05),
        pink=M.plastic("M_FairyKit_BlossomPink", "#FF78B5", rough=0.3, coat=0.6, variation=0.03),
        pink_light=M.plastic("M_FairyKit_BlossomPinkLight", "#FFC4DE", rough=0.3, coat=0.6, variation=0.02),
        peach=M.plastic("M_FairyKit_BlossomPeach", "#FFB08A", rough=0.3, coat=0.6, variation=0.02),
        blue=M.plastic("M_FairyKit_BlossomBlue", "#38C6EA", rough=0.3, coat=0.6, variation=0.03),
        teal=M.plastic("M_FairyKit_BlossomTeal", "#2FD0C2", rough=0.3, coat=0.6, variation=0.03),
        purple=M.plastic("M_FairyKit_ButterflyPurple", "#A874EC", rough=0.3, coat=0.6, variation=0.03),
        lilac=M.plastic("M_FairyKit_ButterflyLilac", "#E6BDF8", rough=0.3, coat=0.6, variation=0.02),
        magenta=M.plastic("M_FairyKit_HeartPink", "#F5479A", rough=0.25, coat=0.8, variation=0.02),
    )


class Painter:
    """Places embossed decorations on a bottle surface using reference-pixel coordinates."""

    def __init__(self, surf, root, pal, name, dv=0.0):
        """dv: vertical offset (px) applied to every decal and the label; measured by
        overlaying test renders on the reference photo."""
        self.surf, self.root, self.P, self.name, self.dv = surf, root, pal, name, dv

    def _place(self, dec, u, v, lift=0.0, rot=0.0, theta=0.0, tilt=0.0):
        return dec.place(self.surf, u=px(u), v=px(v + self.dv), lift=px(lift), rot=rot, theta=theta,
                         parent=self.root, tilt=tilt)

    def label(self, text, text_hex, w, h, u, v, font_size, **kw):
        """Cream plaque + gold rim + lettering; w, h, u, v and font_size in px."""
        return D.label(self.surf, self.root, text, text_hex, px(w), px(h), px(u), px(v + self.dv),
                       font_size=px(font_size), name=f"{self.name}_Label", **kw)

    def flower(self, u, v, size, petal, center=None, inner=None, petals=5, style="round", lift=0.0, rot=0.0,
               theta=0.0, h=None, tilt=0.0):
        center = center or self.P["yellow"]
        size = size * 1.2
        dec = D.flower(px(size), petal, center, petals=petals, inner_mat=inner, style=style,
                       height=px(h) if h else None, name=f"{self.name}_Flower", rotation=rot)
        return self._place(dec, u, v, lift, theta=theta, tilt=tilt)

    def daisy(self, u, v, size, lift=0.0, rot=0.0, petals=6, theta=0.0, tilt=0.0):
        dec = D.flower(px(size * 1.15), self.P["white"], self.P["yellow"], petals=petals, style="daisy",
                       name=f"{self.name}_Daisy", rotation=rot)
        return self._place(dec, u, v, lift, theta=theta, tilt=tilt)

    def star(self, u, v, size, mat=None, lift=0.0, rot=0.0, theta=0.0):
        dec = D.sparkle_star(px(size), mat or self.P["gold"], name=f"{self.name}_Star")
        return self._place(dec, u, v, lift, rot, theta)

    def heart(self, u, v, size, mat=None, lift=0.0, rot=0.0):
        dec = D.heart_decal(px(size), mat or self.P["magenta"], name=f"{self.name}_Heart")
        return self._place(dec, u, v, lift, rot)

    def butterfly(self, u, v, size, wing, inner, body=None, lift=0.0, rot=0.0):
        dec = D.butterfly(px(size), wing, inner, body or self.P["gold"], name=f"{self.name}_Butterfly")
        return self._place(dec, u, v, lift, rot)

    def leaf(self, u, v, length, width, rot, mat=None, lift=0.0, bend=0.1, theta=0.0, tilt=0.0):
        dec = D.leaf(px(length), px(width), mat or self.P["leaf"], bend=bend, name=f"{self.name}_Leaf")
        return self._place(dec, u, v, lift, rot, theta, tilt)

    def vine(self, pts, r=4.8, every=3, leaf_len=30, leaf_w=15, mat=None, leaf_mat=None, side=1, theta=0.0,
             leaves=True, lift=0.0):
        """Embossed stem through `pts` (px) with leaves alternating every `every` control points."""
        pts = np.asarray(pts, dtype=np.float64)
        r, leaf_len, leaf_w = r * 1.25, leaf_len * 1.15, leaf_w * 1.15  # the photo's vines are chunky
        dense = C.catmull_rom(pts, 5) if len(pts) > 2 else pts
        D.surface_vine(self.surf, px(dense + np.array([0.0, self.dv])), px(r), mat or self.P["leaf"],
                       parent=self.root, name=f"{self.name}_Vine", theta=theta,
                       lift_frac=0.45 + lift / max(r, 1e-6))
        if not leaves:
            return
        for i in range(max(1, every // 2) * 5, len(dense) - 4, every * 5):
            t = dense[i + 1] - dense[i]
            a = math.atan2(t[1], t[0]) + side * 1.0 - math.pi / 2
            self.leaf(dense[i][0], dense[i][1], leaf_len, leaf_w, a, mat=leaf_mat, lift=r * 0.3 + lift,
                      bend=0.12 * side, theta=theta)
            side = -side

    def scroll(self, start, heading_deg, length, curl, handed=1, r=4.4, mat=None, turns=1.15, leaves=0,
               theta=0.0, lift=0.0):
        path = S.scroll(start, math.radians(heading_deg), length, curl, turns=turns, handed=handed, n=46)
        D.surface_vine(self.surf, px(path + np.array([0.0, self.dv])), px(r), mat or self.P["leaf"],
                       parent=self.root, name=f"{self.name}_Scroll", theta=theta,
                       lift_frac=0.45 + lift / max(r, 1e-6))
        for k in range(leaves):
            i = int(len(path) * (0.15 + 0.3 * k))
            t = path[i + 1] - path[i]
            side = 1 if k % 2 == 0 else -1
            a = math.atan2(t[1], t[0]) + side * 1.0 - math.pi / 2
            self.leaf(path[i][0], path[i][1], 24, 12, a, lift=r * 0.3 + lift, theta=theta)
        return path

    def dots(self, pts, r, mat=None, theta=0.0):
        if len(pts):
            P = np.asarray(pts, dtype=np.float64) + np.array([0.0, self.dv])
            D.dots(self.surf, px(P), px(r), mat or self.P["gold"], parent=self.root, theta=theta,
                   name=f"{self.name}_Dot")

    def sparkles(self, n, box, exclude, r, mat, seed=0):
        rng = np.random.default_rng(seed)
        pts = []
        while len(pts) < n:
            p = (rng.uniform(box[0], box[1]), rng.uniform(box[2], box[3]))
            if any(e[0] < p[0] < e[1] and e[2] < p[1] < e[3] for e in exclude):
                continue
            pts.append(p)
        self.dots(pts, r, mat)


def _place_objs(objs, matrix, parent):
    for o in objs:
        C.transform_mesh(o, matrix)
        o.parent = parent
    return objs


def _loc(u, depth, h):
    return Vector((px(u), px(depth), px(h)))


def _hang(surf, u, h_top, h_center, clearance):
    """Top point for a charm hanging in front of the body: its centre sits `clearance`
    px in front of the surface at (u, h_center)."""
    loc, _ = surf.hit(Vector((px(u), -10.0, px(h_center))), Vector((0.0, 1.0, 0.0)))
    return Vector((px(u), loc.y - px(clearance), px(h_top)))


# ---------------------------------------------------------------------------
# Stoppers, toppers and charms
# ---------------------------------------------------------------------------

def crown(name, r, h_band, h_tip, z0, parent, points=5):
    gold = M.gold()
    cols = points * 4
    verts, faces = [], []
    levels = 5
    for k in range(cols):
        a = math.tau * k / cols
        phase = k % 4
        ztop = h_tip if phase == 0 else (h_band if phase == 2 else (h_band + h_tip) / 2)
        for j in range(levels):
            f = j / (levels - 1)
            rr = r * (1.0 + 0.16 * f)
            verts.append((rr * math.sin(a), -rr * math.cos(a), z0 + ztop * f))
    for k in range(cols):
        k2 = (k + 1) % cols
        for j in range(levels - 1):
            faces.append((k * levels + j, k2 * levels + j, k2 * levels + j + 1, k * levels + j + 1))
    band = C.mesh_object(f"{name}_Crown", verts, faces, mat=gold, smooth=False)
    sol = band.modifiers.new("Thickness", "SOLIDIFY")
    sol.thickness = r * 0.16
    sol.offset = -1
    band.parent = parent
    out = [band]
    for k in range(points):
        a = math.tau * k * 4 / cols
        rr = r * 1.16 - r * 0.08
        out.append(C.uv_sphere(f"{name}_CrownBall", r * 0.24, 16, 8, mat=gold,
                               location=(rr * math.sin(a), -rr * math.cos(a), z0 + h_tip + r * 0.14)))
    ring = C.torus(f"{name}_CrownRim", r * 1.02, r * 0.13, 48, 8, mat=gold)
    C.transform_mesh(ring, Matrix.Translation((0, 0, z0 + r * 0.06)))
    out.append(ring)
    # solid faceted cap filling the crown (the toy crown is one moulded piece, not a hollow band)
    dome = C.faceted_lathe(f"{name}_CrownDome", [(r * 1.0, z0 + h_band * 0.2), (r * 1.08, z0 + h_band * 1.0),
                                                 (r * 0.8, z0 + h_band * 1.6), (r * 0.42, z0 + h_band * 1.95),
                                                 (0, z0 + h_band * 2.05)], segments=10, mat=gold)
    out.append(dome)
    for o in out:
        o.parent = parent
    return out


def star_frame_topper(name, r_out, z_base, gem_mat, parent, thickness=None):
    gold = M.gold()
    th = thickness or r_out * 0.32
    outer = S.star(r_out, r_out * 0.54, rounding=0.35)
    hole = S.star(r_out * 0.64, r_out * 0.64 * 0.54, rounding=0.25)
    frame = D.two_sided(f"{name}_StarFrame", outer, th, gold, radius=th * 0.45, holes=[hole],
                        spacing=r_out / 40)
    gem = C.ridge_star(f"{name}_StarGem", r_out * 0.7, r_out * 0.7 * 0.52, th * 0.35, table=0.35, back=th * 0.3,
                       wall=th * 0.4, mat=gem_mat)
    C.face_front(gem)
    cz = z_base + r_out * 0.84
    for o in (frame, gem):
        C.transform_mesh(o, Matrix.Translation((0, 0, cz)))
        o.parent = parent
    stem = C.lathe(f"{name}_StarStem", [(0, z_base - r_out * 0.1), (r_out * 0.34, z_base - r_out * 0.1),
                                        (r_out * 0.3, z_base + r_out * 0.18), (r_out * 0.2, z_base + r_out * 0.42),
                                        (0, z_base + r_out * 0.42)], segments=24, mat=gold, smooth_samples=3)
    stem.parent = parent
    return [frame, gem, stem]


def moon_topper(name, r, z_base, parent, rot=0.45):
    gold = M.gold()
    moon = D.two_sided(f"{name}_Moon", S.crescent(r, 0.56), r * 0.5, gold, radius=r * 0.25)
    C.transform_mesh(moon, Matrix.Translation((0, 0, z_base + r * 1.02)) @ Matrix.Rotation(-rot, 4, "Y"))
    k = r / px(52)  # knob proportions measured at r = 52 px
    knob = C.lathe(f"{name}_MoonKnob", [(0, z_base - px(4)), (px(34) * k, z_base - px(4)),
                                        (px(38) * k, z_base + px(8) * k), (px(32) * k, z_base + px(22) * k),
                                        (px(18) * k, z_base + px(32) * k), (px(11) * k, z_base + px(40) * k),
                                        (0, z_base + px(42) * k)],
                   segments=32, mat=gold, smooth_samples=4)
    C.subsurf(knob, 1)
    for o in (moon, knob):
        o.parent = parent
    return [moon, knob]


def gold_star_topper(name, r_out, z_base, parent):
    gold = M.gold()
    # puffy moulded star (rounded tips, pillowed faces) standing upright on the collar
    st = D.two_sided(f"{name}_GoldStar", S.star(r_out, r_out * 0.56, rounding=0.45), r_out * 0.62, gold,
                     radius=r_out * 0.34, spacing=r_out / 30)
    C.transform_mesh(st, Matrix.Translation((0, 0, z_base + r_out * 0.66)))
    stem = C.lathe(f"{name}_StarStem", [(0, z_base - px(3)), (r_out * 0.35, z_base - px(3)),
                                        (r_out * 0.3, z_base + r_out * 0.25), (0, z_base + r_out * 0.3)],
                   segments=24, mat=gold, smooth_samples=3)
    for o in (st, stem):
        o.parent = parent
    return [st, stem]


def flower_topper(name, r, z_base, parent):
    gold = M.gold()
    out = []
    for k in range(5):
        a = math.tau * k / 5 + math.pi / 2
        petal = C.uv_sphere(f"{name}_GoldPetal", 1.0, 20, 10, mat=gold, scale=(r * 0.6, r * 0.46, r * 0.3))
        M4 = (Matrix.Translation((math.cos(a) * r * 0.66, math.sin(a) * r * 0.66, z_base + r * 0.3))
              @ Matrix.Rotation(a, 4, "Z") @ Matrix.Rotation(-0.08, 4, "Y"))
        C.transform_mesh(petal, M4)
        out.append(petal)
    out.append(C.uv_sphere(f"{name}_GoldBud", r * 0.38, 20, 10, mat=gold, location=(0, 0, z_base + r * 0.52)))
    base = C.lathe(f"{name}_GoldBase", [(0, z_base - px(2)), (r * 0.7, z_base - px(2)),
                                        (r * 0.55, z_base + r * 0.35), (0, z_base + r * 0.4)],
                   segments=32, mat=gold, smooth_samples=3)
    out.append(base)
    for o in out:
        o.parent = parent
    return out


def moon_charm(name, size, top, parent, rot=0.35):
    gold = M.gold()
    ring = D.jump_ring(f"{name}_Bail", size * 0.1, size * 0.03, top - Vector((0, 0, size * 0.1)), rot_z=0.0,
                       mat=gold, parent=parent)
    moon = D.two_sided(f"{name}_Moon", S.crescent(size / 2, 0.45), size * 0.2, gold, radius=size * 0.1)
    C.transform_mesh(moon, Matrix.Translation(top - Vector((0, 0, size * 0.62))) @ Matrix.Rotation(-rot, 4, "Y"))
    moon.parent = parent
    return [ring, moon]


def star_charm(name, size, top, parent, links=3, tilt=0.0):
    gold = M.gold()
    out = []
    link_r = size * 0.075
    z = top.z
    for i in range(links):
        o = D.jump_ring(f"{name}_Link", link_r, link_r * 0.28, Vector((top.x, top.y, z - link_r)),
                        rot_z=(math.pi / 2) * (i % 2), mat=gold, parent=parent)
        out.append(o)
        z -= link_r * 1.5
    st = C.ridge_star(f"{name}_Star", size / 2, size / 2 * 0.5, size * 0.12, back=size * 0.08, wall=size * 0.1,
                      mat=gold)
    C.face_front(st)
    C.transform_mesh(st, Matrix.Translation((top.x, top.y, z - size * 0.45)) @ Matrix.Rotation(tilt, 4, "Y"))
    st.parent = parent
    out.append(st)
    return out


def butterfly_charm(name, size, center, parent, wing_mat, inner_mat, yaw=0.0, pitch=0.0):
    gold = M.gold()
    s = size / 2
    out = []
    for side in (1, -1):
        up = S.wing_lobe(s * 0.98, s * 0.95, angle=math.radians(38))
        lo = S.wing_lobe(s * 0.72, s * 0.62, angle=-math.radians(42))
        for lobe, nm in ((up, "Up"), (lo, "Low")):
            L = lobe.copy()
            L[:, 0] *= side
            L = C.ccw(L)
            w = D.two_sided(f"{name}_Wing{nm}", L, size * 0.07, wing_mat, radius=size * 0.035)
            out.append(w)
            inner = C.offset_closed(L, -s * 0.12)
            cen = inner.mean(axis=0)
            inner = cen + (inner - cen) * 0.8
            ins = C.relief(f"{name}_WingInner{nm}", inner, size * 0.012, radius=size * 0.02, mat=inner_mat)
            C.transform_mesh(ins, Matrix.Translation((0, -size * 0.034, 0)))
            out.append(ins)
    body = C.uv_sphere(f"{name}_Body", 1.0, 16, 10, mat=gold, scale=(s * 0.12, s * 0.12, s * 0.5))
    C.transform_mesh(body, Matrix.Translation((0, -size * 0.04, -s * 0.05)))
    out.append(body)
    head = C.uv_sphere(f"{name}_Head", s * 0.11, 12, 8, mat=gold, location=(0, -size * 0.04, s * 0.5))
    out.append(head)
    M4 = Matrix.Translation(center) @ Matrix.Rotation(yaw, 4, "Z") @ Matrix.Rotation(pitch, 4, "X")
    return _place_objs(out, M4, parent)


def fairy_wing_side(name, size, anchor, parent, yaw=-0.5, roll=0.25):
    """The iridescent wing on Stardust Elixir's neck (upper + lower lobe, purple rim, gold veins)."""
    irid = M.iridescent("M_FairyKit_WingIridescentPink", stops=["#F27CC8", "#E58BE0", "#B98AF0", "#F08ED6"])
    rim_mat = M.plastic("M_FairyKit_WingRimPurple", "#B784EE", rough=0.3, coat=0.7, variation=0.02)
    gold = M.gold()
    out = []
    lobes = [S.round_lobe(size, size * 0.6, angle=math.radians(52)),
             S.round_lobe(size * 0.58, size * 0.38, angle=math.radians(-12))]
    for i, L in enumerate(lobes):
        out.append(D.two_sided(f"{name}_WingMembrane{i}", L, size * 0.05, irid, radius=size * 0.02))
        pts = C.resample(L, count=120)
        P3 = np.stack([pts[:, 0], np.zeros(len(pts)), pts[:, 1]], axis=1)
        out.append(C.tube(f"{name}_WingRim{i}", P3, size * 0.035, sides=8, closed=True, mat=rim_mat))
        # veins
        tip = pts[np.argmax(np.linalg.norm(pts, axis=1))]
        for f in (-0.18, 0.0, 0.2):
            ang = math.atan2(tip[1], tip[0]) + f
            ln = np.linalg.norm(tip) * (0.8 - abs(f))
            vpts = np.array([[math.cos(ang) * ln * t + 0.05 * size * math.sin(t * 3), -size * 0.03,
                              math.sin(ang) * ln * t] for t in np.linspace(0.08, 1.0, 12)])
            out.append(C.tube(f"{name}_WingVein", vpts, size * 0.012, sides=6, mat=gold))
    M4 = Matrix.Translation(anchor) @ Matrix.Rotation(yaw, 4, "Z") @ Matrix.Rotation(roll, 4, "Y")
    return _place_objs(out, M4, parent)


# ---------------------------------------------------------------------------
# 1. Petal Glow
# ---------------------------------------------------------------------------

def petal_glow():
    P = palette()
    name = "SM_Prop_FairyPotion_PetalGlow"
    root = C.empty(name)
    body_mat = _glossy("PetalGlowBody", "#EC4F92", sparkle=0.25)
    prof = [(0, 0), (86, 0), (94, 5), (97, 18), (98, 45), (101, 75), (106, 110), (113, 145), (119, 180),
            (123, 215), (124, 245), (120, 265), (111, 283), (99, 298), (84, 312), (68, 324), (58, 336),
            (54, 350), (54, 368), (0, 368)]
    body = _body(f"{name}_Body", prof, body_mat, depth_ratio=0.62, parent=root)
    # base foot ring
    foot = C.torus(f"{name}_Foot", px(92), px(5), 64, 8, mat=body_mat)
    C.transform_mesh(foot, Matrix.Translation((0, 0, px(5))) @ Matrix.Diagonal((1, 0.62, 1, 1)))
    foot.parent = root
    surf = C.Surface([body])
    gold = P["gold"]
    col = D.collar(f"{name}_Collar", px(61), px(54), px(356), gold, lip=px(9), depth_ratio=0.95, flare=px(4),
                   parent=root)
    csurf = C.Surface(col[:1])
    for i in range(12):
        th = math.tau * i / 12
        if i % 2 == 0:
            o = C.torus(f"{name}_CollarRing", px(7), px(2.2), 16, 6, mat=gold)
        else:
            o = C.ridge_star(f"{name}_CollarDiamond", px(9), px(6), px(3), points=2, mat=gold)
        loc, nor = csurf.radial(th, px(382))
        q = Vector((0, 0, 1)).rotation_difference(nor)
        C.transform_mesh(o, Matrix.Translation(loc + nor * px(0.8)) @ q.to_matrix().to_4x4())
        o.parent = root
    gem_mat = M.glass("M_FairyKit_GemPink", "#FF5FB8", ior=1.9, glow=0.08, transmission=0.7)
    gem = C.faceted_lathe(f"{name}_Stopper", _profile([(40, 404), (60, 426), (66, 446), (54, 486), (0, 540)]),
                          segments=8, mat=gem_mat)
    gem.parent = root
    charm_mat = M.plastic("M_FairyKit_CharmHeartPink", "#FF4FA3", rough=0.18, coat=1.0, variation=0.0)
    pt = Painter(surf, root, P, name, dv=32)
    D.heart_charm(f"{name}_Charm", px(70), charm_mat, _hang(surf, 74, 352, 300, 12), parent=root, tilt=0.1)
    pt.label("Petal\nGlow", "#C8106E", 140, 172, 0, 145, 57, cartouche_kw=dict(peak=0.05, notch=0.07, side_notch=0.03))
    LIFT = 6
    pt.heart(5, 213, 30, lift=LIFT)
    pt.heart(5, 83, 30, lift=LIFT)
    pt.heart(4, 276, 42)
    pt.flower(-52, 284, 26, P["magenta"], P["magenta"], h=3)
    pt.flower(58, 247, 38, P["white"], P["yellow"], inner=None, lift=2)
    pt.flower(-66, 96, 40, P["white"], P["yellow"], lift=LIFT)
    pt.flower(68, 86, 36, P["white"], P["yellow"], lift=LIFT)
    pt.vine([(-84, 40), (-90, 90), (-95, 140), (-96, 185), (-88, 222)], every=4)
    pt.scroll((-88, 222), 60, 10, 11, handed=1, leaves=0)
    pt.scroll((-92, 130), 150, 14, 9, handed=-1)
    pt.scroll((-94, 175), 20, 12, 8, handed=1)
    pt.vine([(86, 40), (92, 90), (95, 140), (92, 190)], every=4, side=-1)
    pt.scroll((92, 190), 110, 10, 10, handed=-1)
    pt.scroll((94, 120), 30, 14, 9, handed=1)
    pt.vine([(-86, 26), (-50, 18), (-15, 22), (20, 17), (55, 22), (88, 28)], r=2.8, every=2, leaf_len=14, leaf_w=7)
    pt.sparkles(34, (-104, 104, 12, 330), [(-86, 86, 40, 250), (-40, 40, 250, 300)], 2.2, P["pink_light"], seed=3)
    return root


# ---------------------------------------------------------------------------
# 2. Moonlight Mist
# ---------------------------------------------------------------------------

def moonlight_mist():
    P = palette()
    name = "SM_Prop_FairyPotion_MoonlightMist"
    root = C.empty(name)
    body_mat = _glossy("MoonlightMistBody", "#8C62D6", sparkle=0.2)
    swirl = M.plastic("M_FairyKit_MoonlightSwirl", "#B394EA", rough=0.3, coat=0.7, variation=0.02)
    prof = [(0, 0), (66, 0), (86, 6), (100, 18), (112, 38), (123, 62), (131, 92), (134, 122), (133, 150),
            (128, 178), (124, 205), (122, 228), (117, 247), (106, 262), (92, 275), (76, 287), (60, 298), (48, 310),
            (43, 322), (42, 338), (0, 338)]
    body = _body(f"{name}_Body", prof, body_mat, depth_ratio=0.66, parent=root)
    lip = C.torus(f"{name}_Lip", px(46), px(11), 64, 12, mat=body_mat)
    C.transform_mesh(lip, Matrix.Translation((0, 0, px(346))) @ Matrix.Diagonal((1, 0.9, 1.05, 1)))
    lip.parent = root
    for i in range(12):
        a = math.tau * i / 12
        C.uv_sphere(f"{name}_LipBead", px(5.5), 12, 7, mat=P["gold"],
                    location=(px(56) * math.sin(a), -px(56) * 0.9 * math.cos(a), px(346))).parent = root
    crown(f"{name}", px(56), px(34), px(62), px(354), root)
    surf = C.Surface([body])
    moon_charm(f"{name}_Charm", px(66), _hang(surf, 26, 334, 290, 12), root, rot=-0.25)
    pt = Painter(surf, root, P, name, dv=4)
    pt.label("Moonlight\nMist", "#4B1E98", 172, 141, 8, 140, 44,
             cartouche_kw=dict(peak=0.07, notch=0.08, side_notch=0.035))
    LIFT = 5.5
    pt.star(10, 196, 28, lift=LIFT)
    pt.star(10, 82, 24, lift=LIFT)
    pt.dots([(-70, 180), (86, 180), (-70, 98), (86, 98), (-40, 196), (60, 196)], 3.2)
    pt.flower(-92, 98, 48, P["pink"], P["yellow"], inner=P["pink_light"], lift=3)
    pt.flower(98, 104, 34, P["blue"], P["yellow"], lift=3)
    pt.flower(95, 236, 32, P["pink_light"], P["yellow"])
    pt.flower(-50, 52, 22, P["peach"], P["yellow"])
    pt.flower(66, 48, 22, P["peach"], P["yellow"])
    pt.star(-44, 262, 22, lift=0)
    pt.star(-104, 150, 14)
    pt.star(112, 140, 12)
    pt.vine([(-70, 40), (-100, 70), (-114, 120), (-112, 170), (-100, 215), (-80, 245), (-50, 262)], every=3)
    pt.scroll((-50, 262), 20, 20, 12, handed=1, r=3.0)
    pt.vine([(100, 130), (110, 170), (104, 210)], every=2, side=-1)
    pt.scroll((40, 250), 20, 30, 12, handed=1, r=4.0, mat=swirl)
    pt.scroll((-20, 30), 190, 30, 11, handed=-1, r=4.0, mat=swirl)
    pt.scroll((20, 30), -10, 30, 11, handed=1, r=4.0, mat=swirl)
    pt.scroll((100, 270), 150, 18, 9, handed=-1, r=3.5, mat=swirl)
    pt.sparkles(20, (-125, 125, 20, 300), [(-100, 110, 50, 230)], 2.2, P["lilac"], seed=5)
    return root


# ---------------------------------------------------------------------------
# 3. Fairy Wings
# ---------------------------------------------------------------------------

def fairy_wings():
    P = palette()
    name = "SM_Prop_FairyPotion_FairyWings"
    root = C.empty(name)
    body_mat = _glossy("FairyWingsBody", "#1CC0BE", sparkle=0.25)
    prof = [(0, 0), (90, 0), (97, 6), (101, 20), (103, 60), (104, 120), (105, 180), (106, 225), (104, 248),
            (97, 268), (88, 283), (78, 296), (68, 309), (60, 323), (54, 338), (51, 355), (50, 380), (50, 413),
            (0, 413)]
    body = _body(f"{name}_Body", prof, body_mat, depth_ratio=0.74, parent=root)
    rim = C.lathe(f"{name}_Rim", _profile([(0, 408), (54, 408), (58, 414), (58, 432), (53, 438), (0, 438)]),
                  segments=48, mat=body_mat, smooth_samples=3)
    C.subsurf(rim, 1)
    rim.parent = root
    for i in range(10):
        a = math.tau * i / 10
        C.uv_sphere(f"{name}_RimBead", px(5.5), 12, 7, mat=P["gold"],
                    location=(px(59) * math.sin(a), -px(59) * math.cos(a), px(423))).parent = root
    D.collar(f"{name}_Collar", px(48), px(26), px(436), P["gold"], lip=px(6), parent=root)
    gem_mat = M.glass("M_FairyKit_GemTeal", "#5FF0DD", ior=1.6)
    star_frame_topper(f"{name}", px(74), px(460), gem_mat, root)
    surf = C.Surface([body])
    bc = _hang(surf, 46, 350, 350, 14)
    butterfly_charm(f"{name}_Charm", px(130), bc, root, P["purple"], P["pink"], yaw=0.05, pitch=0.3)
    for i, h in enumerate((436, 424, 412, 400, 388)):
        D.jump_ring(f"{name}_CharmLink", px(4.5), px(1.5), Vector((px(46), bc.y + px(10) + px(3) * (4 - i), px(h))),
                    rot_z=1.57 * (i % 2), mat=P["gold"], parent=root)
    pt = Painter(surf, root, P, name, dv=30)
    pt.label("Fairy\nWings", "#147C8C", 136, 149, 8, 140, 50, cartouche_kw=dict(peak=0.06, notch=0.08, side_notch=0.03))
    LIFT = 5.5
    pt.butterfly(12, 238, 46, P["pink"], P["pink_light"], lift=LIFT)
    pt.heart(10, 90, 28, lift=LIFT)
    pt.flower(-50, 96, 36, P["purple"], P["yellow"], inner=P["lilac"], lift=LIFT)
    pt.flower(-60, 62, 20, P["pink"], P["yellow"])
    pt.flower(66, 108, 30, P["teal"], P["yellow"], lift=LIFT)
    pt.flower(46, 50, 32, P["white"], P["yellow"])
    pt.vine([(-60, 150), (-72, 200), (-66, 240), (-50, 270)], every=2)
    pt.scroll((-50, 270), 70, 8, 8, handed=-1)
    pt.vine([(76, 150), (84, 200), (78, 245), (62, 272)], every=2, side=-1)
    pt.vine([(-90, 30), (-80, 60), (-86, 110)], every=2, r=2.8)
    pt.sparkles(24, (-105, 105, 15, 320), [(-80, 90, 55, 225)], 2.2, P["lilac"], seed=7)
    return root


# ---------------------------------------------------------------------------
# 4. Forest Breath
# ---------------------------------------------------------------------------

def forest_breath():
    P = palette()
    name = "SM_Prop_FairyPotion_ForestBreath"
    root = C.empty(name)
    body_mat = _glossy("ForestBreathBody", "#7FC034", sparkle=0.15)
    prof = [(0, 0), (52, 0), (78, 12), (106, 35), (124, 70), (134, 105), (137, 140), (135, 175), (128, 205),
            (116, 232), (100, 255), (84, 274), (70, 290), (62, 302), (60, 322), (0, 322)]
    body = _body(f"{name}_Body", prof, body_mat, depth_ratio=0.64, parent=root)
    neck = C.lathe(f"{name}_Neck", _profile([(0, 300), (62, 300), (66, 306), (68, 318), (63, 324), (66, 330),
                                             (70, 338), (69, 352), (62, 358), (54, 358), (0, 358)]),
                   segments=64, mat=body_mat, smooth_samples=3)
    C.subsurf(neck, 1)
    neck.parent = root
    ck = C.lathe(f"{name}_Cork", _profile([(0, 330), (52, 330), (55, 358), (58, 385), (60, 405), (58, 411),
                                           (50, 414), (0, 415)]), segments=48, mat=M.cork(), smooth_samples=4)
    C.subsurf(ck, 1)
    ck.parent = root
    surf = C.Surface([body])
    pt = Painter(surf, root, P, name, dv=26)
    # dense leafy wreath with big daisies around the shoulder
    for i, th in enumerate(np.linspace(-2.9, 2.9, 26)):
        pt.leaf(0, 222 + 10 * math.sin(i * 1.7), 58, 30,
                math.pi / 2 + 0.5 * math.sin(i * 2.3) + (0.5 if i % 2 else -0.5),
                mat=P["leaf_dark"] if i % 3 else P["leaf"], theta=th, tilt=0.55, bend=0.15)
    for i, th in enumerate(np.linspace(-2.8, 2.8, 20)):
        pt.leaf(0, 206, 46, 24, -math.pi / 2 + (0.7 if i % 2 else -0.7), mat=P["leaf_dark"], theta=th, tilt=0.4,
                lift=2)
    for th in (-0.55, 0.52, -1.9, 1.9, math.pi):
        pt.daisy(0, 230, 76, lift=7, theta=th, tilt=0.5, rot=0.3)
    pt.label("Forest\nBreath", "#1F7A2C", 147, 147, 0, 112, 50, cartouche_kw=dict(peak=0.06, notch=0.07, side_notch=0.03))
    LIFT = 5.5
    pt.butterfly(4, 198, 46, P["purple"], P["lilac"], lift=LIFT)
    pt.star(4, 44, 22, lift=LIFT)
    pt.daisy(-78, 96, 42, lift=LIFT)
    pt.daisy(82, 74, 38, lift=LIFT)
    for sgn in (-1, 1):  # laurel garlands: paired leaves along a curving stem either side of the label
        stem = [(sgn * 50, 20), (sgn * 95, 45), (sgn * 118, 100), (sgn * 118, 160), (sgn * 100, 205), (sgn * 80, 230)]
        pt.vine(stem, r=4.2, every=1, leaf_len=30, leaf_w=15, mat=P["leaf_dark"], leaf_mat=P["leaf_dark"],
                side=sgn)
        pt.vine(stem, r=4.2, every=1, leaf_len=26, leaf_w=13, mat=P["leaf_dark"], leaf_mat=P["leaf"],
                side=-sgn)
    pt.dots([(-60, 205), (-40, 212), (60, 205), (40, 212), (-86, 150), (90, 150), (-60, 30), (66, 28)], 3.0)
    pt.sparkles(16, (-120, 120, 10, 240), [(-95, 95, 25, 200)], 2.4, P["yellow"], seed=11)
    return root


# ---------------------------------------------------------------------------
# 5. Stardust Elixir
# ---------------------------------------------------------------------------

def stardust_elixir():
    P = palette()
    name = "SM_Prop_FairyPotion_StardustElixir"
    root = C.empty(name)
    body_mat = _glossy("StardustElixirBody", "#F7B80E", sparkle=0.3)
    swirl = M.plastic("M_FairyKit_StardustSwirl", "#FFD84A", rough=0.3, coat=0.7, variation=0.02)
    prof = [(0, 0), (100, 0), (112, 5), (120, 15), (125, 35), (128, 70), (129, 130), (127, 180), (122, 210),
            (114, 230), (102, 248), (88, 264), (72, 280), (58, 295), (50, 310), (48, 322), (0, 322)]
    body = _body(f"{name}_Body", prof, body_mat, depth_ratio=0.66, parent=root)
    purple = M.plastic("M_FairyKit_StardustCollar", "#A565D8", rough=0.28, coat=0.7, variation=0.02)
    band = C.lathe(f"{name}_Collar", _profile([(0, 314), (52, 314), (58, 318), (60, 330), (60, 352), (57, 362),
                                               (52, 366), (0, 366)]), segments=64, mat=purple, smooth_samples=3)
    C.subsurf(band, 1)
    band.parent = root
    for i in range(10):
        a = math.tau * i / 10
        C.uv_sphere(f"{name}_CollarBead", px(5), 12, 7, mat=P["gold"],
                    location=(px(61) * math.sin(a), -px(61) * math.cos(a), px(341))).parent = root
        if i % 2 == 0:
            C.uv_sphere(f"{name}_CollarGem", px(3.5), 10, 6, mat=P["gold"],
                        location=(px(61) * math.sin(a + 0.31), -px(61) * math.cos(a + 0.31), px(341))).parent = root
    moon_topper(f"{name}", px(60), px(366), root, rot=0.5)
    surf = C.Surface([body])
    star_charm(f"{name}_Charm", px(70), _hang(surf, 40, 334, 262, 10), root, links=3)
    fairy_wing_side(f"{name}_Wing", px(150), _loc(54, 4, 326), root, yaw=-0.3, roll=-0.12)
    pt = Painter(surf, root, P, name, dv=6)
    pt.label("Stardust\nElixir", "#C0136E", 167, 164, 10, 140, 48,
             cartouche_kw=dict(peak=0.06, notch=0.08, side_notch=0.03))
    LIFT = 6
    pt.star(10, 202, 28, lift=LIFT)
    pt.star(10, 78, 26, lift=LIFT)
    pt.flower(-70, 222, 40, P["pink"], P["yellow"], inner=P["pink_light"])
    pt.flower(-72, 76, 44, P["pink"], P["yellow"], inner=P["pink_light"], lift=LIFT)
    pt.flower(84, 74, 44, P["pink"], P["yellow"], inner=P["pink_light"], lift=LIFT)
    pt.flower(108, 142, 22, P["pink"], P["yellow"])
    pt.vine([(-84, 190), (-100, 150), (-104, 110)], every=2)
    pt.vine([(-60, 50), (-40, 22), (0, 16), (40, 22), (70, 45)], r=2.8, every=2, leaf_len=14, leaf_w=7)
    pt.vine([(104, 110), (112, 60), (100, 30)], every=2, side=-1)
    pt.scroll((-30, 250), 160, 26, 11, handed=-1, r=4.0, mat=swirl)
    pt.scroll((40, 255), 10, 22, 10, handed=1, r=4.0, mat=swirl)
    pt.scroll((-40, 24), 190, 30, 11, handed=-1, r=4.0, mat=swirl)
    pt.dots([(-60, 190), (80, 190), (-60, 100), (84, 100)], 3.0)
    pt.sparkles(26, (-125, 125, 15, 300), [(-90, 105, 55, 225)], 2.4, P["white"], seed=13)
    return root


# ---------------------------------------------------------------------------
# 6. Love Blooms (heart bottle on a round foot)
# ---------------------------------------------------------------------------

def love_blooms():
    P = palette()
    name = "SM_Prop_FairyPotion_LoveBlooms"
    root = C.empty(name)
    body_mat = _glossy("LoveBloomsBody", "#F4478B", sparkle=0.2)
    half = [(0, 258), (22, 272), (52, 283), (84, 282), (110, 270), (129, 248), (137, 218), (136, 186),
            (128, 152), (114, 120), (98, 90), (86, 64), (80, 46), (60, 40)]
    right = np.array(half, dtype=np.float64)
    left = right[::-1].copy()
    left[:, 0] *= -1
    outline = np.vstack([right[:-1], [(0, 38)], left[1:-1]])
    outline = S.smooth_closed(px(outline), 8)
    heart = C.inflate(f"{name}_Body", outline, px(56), center=(0.0, px(170)), rings=14, power=0.55,
                      mat=body_mat, count=160)
    C.subsurf(heart, 1)
    heart.parent = root
    foot = C.lathe(f"{name}_Foot", _profile([(0, 0), (74, 0), (80, 5), (80, 14), (74, 22), (66, 30), (62, 40),
                                             (60, 50), (0, 50)]), segments=64, depth_ratio=0.8, mat=body_mat,
                   smooth_samples=4)
    C.subsurf(foot, 1)
    foot.parent = root
    neck = C.lathe(f"{name}_Neck", _profile([(0, 240), (40, 240), (40, 290), (0, 290)]), segments=32,
                   mat=body_mat)
    neck.parent = root
    col = D.collar(f"{name}_Collar", px(56), px(52), px(280), P["gold"], lip=px(9), flare=px(5), parent=root)
    csurf = C.Surface(col[:1])
    for i in range(10):
        th = math.tau * i / 10
        o = C.torus(f"{name}_CollarRing", px(7), px(2.2), 16, 6, mat=P["gold"]) if i % 2 else \
            C.ridge_star(f"{name}_CollarDiamond", px(9), px(6), px(3), points=2, mat=P["gold"])
        loc, nor = csurf.radial(th, px(305))
        q = Vector((0, 0, 1)).rotation_difference(nor)
        C.transform_mesh(o, Matrix.Translation(loc + nor * px(0.8)) @ q.to_matrix().to_4x4())
        o.parent = root
    gem_mat = M.glass("M_FairyKit_GemPinkHeart", "#FF6FBE", ior=1.9, glow=0.08, transmission=0.7)
    gem = C.faceted_outline_gem(f"{name}_Stopper", S.heart(px(124), px(100), plump=0.6), crown=0.26,
                                pavilion=0.26, table=0.55, girdle=0.06, facet_points=20, mat=gem_mat)
    C.face_front(gem)
    C.transform_mesh(gem, Matrix.Translation((0, 0, px(356))))
    gem.parent = root
    charm_mat = M.plastic("M_FairyKit_CharmHeartPink", "#FF4FA3", rough=0.18, coat=1.0, variation=0.0)
    surf = C.Surface([heart])
    D.heart_charm(f"{name}_Charm", px(66), charm_mat, _hang(surf, 14, 298, 250, 10), parent=root, tilt=0.0)
    pt = Painter(surf, root, P, name, dv=30)
    pt.label("Love\nBlooms", "#C8106E", 150, 150, 8, 126, 48, shape=S.heart(px(150), px(150), n=160, dip=1.15),
             text_offset=(0.0, px(16)))
    LIFT = 5
    pt.heart(8, 76, 32, lift=LIFT)
    pt.flower(-72, 212, 54, P["pink_light"], P["yellow"], inner=P["white"])
    pt.flower(98, 202, 48, P["pink"], P["yellow"], inner=P["pink_light"])
    pt.flower(-60, 94, 42, P["pink_light"], P["yellow"], inner=P["white"], lift=2)
    pt.flower(78, 92, 40, P["pink"], P["yellow"], inner=P["pink_light"], lift=2)
    pt.vine([(-72, 190), (-84, 150), (-100, 120)], every=2, r=3.2)
    pt.vine([(98, 180), (104, 140), (100, 110)], every=2, r=3.2, side=-1)
    pt.vine([(-60, 72), (-40, 52), (-10, 46)], every=2, r=2.6)
    pt.vine([(78, 70), (50, 52), (22, 46)], every=2, r=2.6, side=-1)
    pt.flower(-30, 250, 18, P["pink_light"], P["yellow"])
    pt.flower(60, 262, 16, P["pink_light"], P["yellow"])
    pt.flower(-110, 140, 18, P["pink_light"], P["yellow"])
    pt.sparkles(30, (-130, 130, 50, 280), [(-75, 85, 45, 210)], 2.4, P["pink_light"], seed=17)
    return root


# ---------------------------------------------------------------------------
# 7. Pixie Dust (rounded-square bottle)
# ---------------------------------------------------------------------------

def pixie_dust():
    P = palette()
    name = "SM_Prop_FairyPotion_PixieDust"
    root = C.empty(name)
    body_mat = _glossy("PixieDustBody", "#20C6BE", sparkle=0.25)
    prof = [(0, 0), (74, 0), (80, 4), (83, 12), (84, 30), (84, 190), (83, 205), (78, 218), (68, 228),
            (56, 234), (50, 238), (0, 238)]
    body = _body(f"{name}_Body", prof, body_mat, depth_ratio=0.86, exponent=4.0, parent=root)
    D.collar(f"{name}_Collar", px(57), px(28), px(234), P["gold"], lip=px(6), flare=px(3), beads=10,
             bead_r=px(4.5), parent=root)
    gold_star_topper(f"{name}", px(56), px(256), root)
    surf = C.Surface([body])
    pt = Painter(surf, root, P, name, dv=0)
    bead = M.plastic("M_FairyKit_PixieBead", "#7EE6DC", rough=0.3, coat=0.7, variation=0.02)
    pt.dots([(x, 184 + 4 * math.sin(x * 0.3)) for x in np.linspace(-70, 70, 13)], 5.0, bead)
    pt.vine([(-70, 170), (-40, 176), (0, 172), (40, 176), (70, 170)], r=2.4, leaves=False, mat=bead)
    pt.label("Pixie\nDust", "#1A56B8", 123, 136, 6, 88, 48, cartouche_kw=dict(peak=0.07, notch=0.08, side_notch=0.03))
    LIFT = 5
    pt.star(6, 140, 22, lift=LIFT)
    pt.star(6, 38, 24, lift=LIFT)
    pt.flower(-54, 48, 34, P["pink_light"], P["yellow"], inner=P["white"], lift=LIFT)
    pt.flower(56, 46, 32, P["pink_light"], P["yellow"], inner=P["white"], lift=LIFT)
    pt.vine([(-76, 60), (-78, 110), (-72, 150)], r=2.8, every=2)
    pt.vine([(78, 60), (80, 110), (74, 150)], r=2.8, every=2, side=-1)
    pt.sparkles(18, (-84, 84, 10, 225), [(-70, 75, 12, 165)], 2.2, P["white"], seed=19)
    return root


# ---------------------------------------------------------------------------
# 8. Dreamcap Essence (jar with mushroom-cap lid)
# ---------------------------------------------------------------------------

def dreamcap_essence():
    P = palette()
    name = "SM_Prop_FairyPotion_DreamcapEssence"
    root = C.empty(name)
    body_mat = _glossy("DreamcapJar", "#8954CF", sparkle=0.2)
    cap_mat = _glossy("DreamcapCap", "#FF4A9C", sparkle=0.1)
    spot_mat = M.plastic("M_FairyKit_DreamcapSpot", "#FFC0DD", rough=0.3, coat=0.7, variation=0.02)
    prof = [(0, 0), (106, 0), (113, 4), (116, 12), (117, 30), (117, 180), (115, 192), (110, 200), (0, 200)]
    body = _body(f"{name}_Jar", prof, body_mat, depth_ratio=0.92, parent=root)
    rim = C.torus(f"{name}_JarRim", px(108), px(12), 72, 12, mat=body_mat)
    C.transform_mesh(rim, Matrix.Translation((0, 0, px(212))) @ Matrix.Diagonal((1, 0.92, 1.1, 1)))
    rim.parent = root
    cap = C.lathe(f"{name}_Cap", _profile([(0, 224), (100, 222), (114, 225), (120, 232), (121, 241), (117, 251),
                                           (106, 264), (90, 278), (68, 291), (40, 300), (0, 305)]),
                  segments=72, depth_ratio=0.95, mat=cap_mat, smooth_samples=5)
    C.subsurf(cap, 2)
    cap.parent = root
    csurf = C.Surface([cap])
    spots = [(-53, 294, 44), (17, 276, 52), (72, 300, 38), (-103, 266, 34), (-43, 250, 38), (89, 256, 40)]
    for u, v, s in spots:
        dec = D.Decal()
        dec.add(C.relief(f"{name}_Spot", S.ellipse(px(s) / 2, px(s) / 2 * 0.8, 40), px(3.5), radius=px(6),
                         mat=spot_mat, skirt=0.003))
        dec.place(csurf, u=px(u), v=px(v), parent=root, tilt=0.35)
    for th in (2.2, 2.9, -2.4, 3.9):
        dec = D.Decal()
        dec.add(C.relief(f"{name}_Spot", S.ellipse(px(20), px(16), 40), px(3.5), radius=px(6), mat=spot_mat,
                         skirt=0.003))
        dec.place(csurf, u=0.0, v=px(270), theta=th, parent=root, tilt=0.4)
    surf = C.Surface([body])
    pt = Painter(surf, root, P, name, dv=22)
    pt.label("Dreamcap\nEssence", "#4A1F90", 156, 145, 0, 110, 42,
             cartouche_kw=dict(peak=0.07, notch=0.08, side_notch=0.03))
    LIFT = 5
    pt.star(6, 166, 26, lift=LIFT)
    stem_mat = M.plastic("M_FairyKit_MushroomStem", "#FFE9E4", rough=0.3, coat=0.6, variation=0.02)
    for u, v, s, lift in ((-74, 72, 62, LIFT), (4, 52, 36, LIFT)):
        dec = D.Decal()
        capo = np.vstack([S.ellipse(px(s) / 2, px(s) * 0.36, 48)])
        capo = capo[capo[:, 1] >= -1e-9]
        capo = np.vstack([capo, [[-px(s) / 2, 0.0]]])
        dec.add(C.relief(f"{name}_MushCap", C.ccw(capo), px(s) * 0.12, radius=px(s) * 0.2, mat=cap_mat,
                         skirt=0.003), offset=(0.0, px(s) * 0.1))
        dec.add(C.relief(f"{name}_MushStem", S.rounded_rect(px(s) * 0.34, px(s) * 0.5, px(s) * 0.12),
                         px(s) * 0.08, radius=px(s) * 0.1, mat=stem_mat, skirt=0.003), offset=(0.0, -px(s) * 0.12))
        for dx, dy, r in ((-0.22, 0.28, 0.06), (0.1, 0.33, 0.05), (0.26, 0.2, 0.045)):
            dec.add(C.relief(f"{name}_MushSpot", S.circle(px(s) * r, 20), px(s) * 0.03, radius=px(s) * 0.05,
                             mat=spot_mat, skirt=0.002), lift=px(s) * 0.1, offset=(px(s) * dx, px(s) * dy))
        dec.place(surf, u=px(u), v=px(v + pt.dv), lift=px(lift), parent=root)
    pt.flower(-80, 172, 30, P["pink_light"], P["yellow"], inner=P["white"])
    pt.flower(94, 166, 34, P["pink_light"], P["yellow"], inner=P["white"])
    pt.flower(68, 64, 40, P["pink"], P["yellow"], inner=P["pink_light"], lift=LIFT)
    pt.vine([(-100, 60), (-104, 100), (-98, 140)], r=2.8, every=2)
    pt.vine([(40, 20), (70, 30), (100, 40)], r=2.8, every=2, side=-1)
    pt.vine([(102, 100), (106, 140)], r=2.6, every=2)
    pt.sparkles(20, (-115, 115, 10, 195), [(-85, 90, 30, 190)], 2.2, P["lilac"], seed=23)
    return root


# ---------------------------------------------------------------------------
# 9. Tranquil Tears (round flask, gold flower stopper)
# ---------------------------------------------------------------------------

def tranquil_tears():
    P = palette()
    name = "SM_Prop_FairyPotion_TranquilTears"
    root = C.empty(name)
    body_mat = _glossy("TranquilTearsBody", "#0DA1E6", sparkle=0.2)
    swirl = M.plastic("M_FairyKit_TranquilSwirl", "#4CC4F2", rough=0.3, coat=0.7, variation=0.02)
    prof = [(0, 0), (40, 0), (58, 4), (72, 12), (84, 25), (94, 45), (101, 68), (105, 95), (105, 115), (102, 135),
            (96, 152), (87, 166), (76, 178), (64, 187), (56, 194), (55, 200), (0, 200)]
    body = _body(f"{name}_Body", prof, body_mat, depth_ratio=0.82, parent=root)
    gold = P["gold"]
    cup = C.lathe(f"{name}_Collar", _profile([(0, 194), (55, 194), (58, 198), (57, 206), (59, 216), (62, 224),
                                              (61, 229), (57, 231), (0, 231)]), segments=64, mat=gold,
                  smooth_samples=3)
    C.subsurf(cup, 1)
    cup.parent = root
    for i in range(12):
        a = math.tau * i / 12
        C.uv_sphere(f"{name}_CollarBead", px(4), 10, 6, mat=gold,
                    location=(px(60) * math.sin(a), -px(60) * math.cos(a), px(218))).parent = root
    flower_topper(f"{name}", px(50), px(228), root)
    surf = C.Surface([body])
    pt = Painter(surf, root, P, name, dv=24)
    pt.label("Tranquil\nTears", "#1558A8", 119, 117, 0, 96, 40, cartouche_kw=dict(peak=0.08, notch=0.08, side_notch=0.03))
    LIFT = 4.5
    pt.star(0, 138, 22, lift=LIFT)
    pt.star(0, 46, 22, lift=LIFT)
    pt.flower(-60, 64, 40, P["pink"], P["yellow"], inner=P["pink_light"], lift=LIFT)
    pt.flower(76, 84, 30, P["purple"], P["yellow"], inner=P["lilac"])
    pt.flower(62, 40, 22, P["blue"], P["yellow"])
    pt.vine([(-84, 100), (-92, 130), (-84, 160)], r=2.6, every=2)
    pt.vine([(80, 110), (90, 140), (80, 165)], r=2.6, every=2, side=-1)
    pt.vine([(-40, 30), (-10, 22), (30, 26)], r=2.4, every=2)
    pt.scroll((-40, 172), 170, 18, 9, handed=-1, r=3.4, mat=swirl)
    pt.scroll((30, 176), 10, 18, 9, handed=1, r=3.4, mat=swirl)
    pt.scroll((-90, 60), 250, 14, 8, handed=-1, r=3.2, mat=swirl)
    pt.sparkles(16, (-100, 100, 10, 190), [(-70, 70, 25, 165)], 2.0, P["white"], seed=29)
    return root


BOTTLES = {
    "PetalGlow": petal_glow,
    "MoonlightMist": moonlight_mist,
    "FairyWings": fairy_wings,
    "ForestBreath": forest_breath,
    "StardustElixir": stardust_elixir,
    "LoveBlooms": love_blooms,
    "PixieDust": pixie_dust,
    "DreamcapEssence": dreamcap_essence,
    "TranquilTears": tranquil_tears,
}
