"""Non-bottle props: spell book, gem bowl, loose gems, star wand, gold scoop and wing clip.
Dimensions are reference-photo pixels converted with `px()` (see decor.PX)."""

import math

import numpy as np
from mathutils import Matrix, Vector

from . import core as C
from . import decor as D
from . import materials as M
from . import shapes2d as S
from .bottles import palette
from .decor import px


def _prism(name, outline, z0, z1, mat, rings=4):
    """Extrude a closed 2D outline (in XY) along Z with a few intermediate rings."""
    P = np.asarray(outline, dtype=np.float64)
    n = len(P)
    verts, faces = [], []
    zs = np.linspace(z0, z1, rings)
    for z in zs:
        verts += [(p[0], p[1], z) for p in P]
    for j in range(rings - 1):
        for k in range(n):
            k2 = (k + 1) % n
            faces.append((j * n + k, j * n + k2, (j + 1) * n + k2, (j + 1) * n + k))
    faces.append(tuple(reversed(range(n))))
    faces.append(tuple((rings - 1) * n + k for k in range(n)))
    return C.mesh_object(name, verts, faces, mat=mat)


def _box(name, sx, sy, sz, center, mat, bevel_segments=2):
    obj = C.mesh_object(name, [(-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1), (-1, -1, 1), (1, -1, 1),
                               (1, 1, 1), (-1, 1, 1)],
                        [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)],
                        mat=mat, smooth=False)
    C.transform_mesh(obj, Matrix.Translation(center) @ Matrix.Diagonal((sx / 2, sy / 2, sz / 2, 1)))
    bev = obj.modifiers.new("Bevel", "BEVEL")
    bev.width = min(sx, sy, sz) * 0.08
    bev.segments = bevel_segments
    C.set_smooth(obj, True)
    return obj


# ---------------------------------------------------------------------------
# Spell book "Fairy Potions"
# ---------------------------------------------------------------------------

def fairy_book():
    P = palette()
    name = "SM_Prop_FairyPotion_SpellBook"
    root = C.empty(name)
    cover = M.velvet("M_FairyKit_BookCover", "#6C2C9E")
    spine_mat = M.velvet("M_FairyKit_BookSpine", "#5B2289")
    pages = M.plastic("M_FairyKit_BookPages", "#F6E7D2", rough=0.6, coat=0.1, variation=0.04, bump=0.05)
    gold = P["gold"]
    W, H, T, B = 356.0, 356.0, 92.0, 14.0  # width, height, thickness, board thickness (px)
    x0 = -W / 2  # spine side
    board = S.rounded_rect(px(W), px(H), px(10))
    for y, nm in ((-T / 2 + B / 2, "FrontCover"), (T / 2 - B / 2, "BackCover")):
        b = D.two_sided(f"{name}_{nm}", board, px(B), cover, radius=px(5), spacing=px(10))
        C.transform_mesh(b, Matrix.Translation((0, px(y), px(H / 2))))
        b.parent = root
    _box(f"{name}_Pages", px(W - 22), px(T - 2 * B + 2), px(H - 16), Vector((px(4), 0, px(H / 2))), pages).parent = root
    # rounded spine
    R_out, R_in = T / 2 + 2, T / 2 - 12
    arc = [(x0 - 4 - R_out * 0.55 * math.sin(a), R_out * math.cos(a)) for a in np.linspace(0, math.pi, 24)]
    arc += [(x0 + 6 - R_in * 0.4 * math.sin(a), R_in * math.cos(a)) for a in np.linspace(math.pi, 0, 24)]
    sp = _prism(f"{name}_Spine", px(np.array(arc)), px(1), px(H - 1), spine_mat, rings=6)
    C.subsurf(sp, 1)
    sp.parent = root
    # gold bands and stars on the spine
    for z in (20, 125, 290, 334):
        pts = [(x0 - 4 - (R_out + 1) * 0.55 * math.sin(a), (R_out + 1) * math.cos(a), z) for a in
               np.linspace(0.42, math.pi - 0.42, 30)]
        C.tube(f"{name}_SpineBand", px(np.array(pts)), px(6.5), sides=10, mat=gold, flatten=0.7).parent = root
    for z in (55, 175, 215, 255):
        st = C.ridge_star(f"{name}_SpineStar", px(15), px(7.5), px(4), wall=px(3), mat=gold)
        C.transform_mesh(st, Matrix.Translation((px(x0 - 4 - (R_out + 1) * 0.55), px(-4), px(z)))
                         @ Matrix.Rotation(-math.pi / 2, 4, "Y") @ Matrix.Rotation(math.pi / 2, 4, "Z"))
        st.parent = root

    # ---- front cover artwork (decal space, then laid flat on the cover) ----
    front = Matrix.Translation((0, px(-T / 2), px(H / 2)))
    dec = D.Decal()

    def add(obj, lift=0.0, off=(0.0, 0.0)):
        dec.add(obj, lift=px(lift), offset=(px(off[0]), px(off[1])))

    # border line + corner filigree
    inset = S.rounded_rect(px(W - 44), px(H - 44), px(16))
    border = D.stem_relief(np.vstack([C.resample(inset, count=200), C.resample(inset, count=200)[:1]]), px(2.2), gold,
                           name=f"{name}_Border")
    add(border)
    for sx in (-1, 1):
        for sz in (-1, 1):
            cx, cz = sx * (W / 2 - 34), sz * (H / 2 - 34)
            # one scroll running along the top/bottom edge, one down/up the side; both curl inwards
            for heading in ((0.0 if sx < 0 else math.pi), (-math.pi / 2 if sz > 0 else math.pi / 2)):
                path = S.scroll((cx, cz), heading, 38, 9, turns=1.1, handed=sx * sz, n=40)
                add(D.stem_relief(px(path), px(2.4), gold, name=f"{name}_Filigree"))
            add(C.relief(f"{name}_CornerDot", S.circle(px(5), 16, center=(px(cx), px(cz))), px(3), radius=px(5),
                         mat=gold, skirt=0.002))
    # title
    title = C.text_relief(f"{name}_Title", "Fairy\nPotions", D.FONT_TITLE, px(100), px(13), px(3.5), line_spacing=0.74,
                          mat=gold)
    add(title, off=(0, 60))
    # fairy silhouette (union of flat gold reliefs of equal height)
    fx, fz, fh = 40.0, -76.0, 146.0

    def fp(x, y):
        return px(fx + x * fh), px(fz + y * fh)

    def blob(outline, nm):
        add(C.relief(f"{name}_Fairy{nm}", outline, px(4), radius=px(2.5), mat=gold, skirt=0.002))

    blob(S.circle(px(0.09 * fh), 32, center=fp(-0.02, 0.35)), "Head")
    for cx_, cy_, r_ in ((0.07, 0.41, 0.085), (-0.07, 0.43, 0.065), (0.11, 0.32, 0.07), (0.02, 0.47, 0.07),
                         (-0.11, 0.36, 0.045), (0.13, 0.22, 0.05)):
        blob(S.circle(px(r_ * fh), 24, center=fp(cx_, cy_)), "Hair")
    blob(S.ellipse(px(0.08 * fh), px(0.12 * fh), 32, center=fp(0.0, 0.16)), "Torso")
    hem = [(-0.3 + 0.6 * t, -0.3 - 0.03 * math.sin(t * math.pi * 7)) for t in np.linspace(0, 1, 30)]
    dress = [(0.07, 0.12), (0.09, 0.06), (0.3, -0.28)] + [(x, y) for x, y in hem[::-1]] + [(-0.09, 0.06),
                                                                                           (-0.07, 0.12)]
    blob(C.ccw(np.array([fp(x, y) for x, y in dress])), "Dress")
    for (a, b, r) in (((-0.05, -0.3), (-0.06, -0.48), 0.032), ((0.06, -0.3), (0.1, -0.47), 0.032),
                      ((-0.06, 0.22), (-0.27, 0.05), 0.034), ((0.06, 0.22), (0.17, 0.06), 0.032),
                      ((0.0, 0.25), (0.0, 0.3), 0.04)):
        path = np.array([fp(*a), fp(*b)])
        path = np.vstack([path[0] + (path[1] - path[0]) * t for t in np.linspace(0, 1, 8)])
        blob_obj = D.stem_relief(path, px(r * fh), gold, name=f"{name}_FairyLimb")
        add(blob_obj)
    wand = np.array([fp(-0.25, 0.04), fp(-0.37, -0.02)])
    wand = np.vstack([wand[0] + (wand[1] - wand[0]) * t for t in np.linspace(0, 1, 6)])
    add(D.stem_relief(wand, px(1.8), gold, name=f"{name}_FairyWand"))
    wing_mat = M.iridescent("M_FairyKit_BookWing", stops=["#EE86CF", "#BE86EE", "#F09AD8", "#C58AEE"], glitter=True)
    for L, nm in ((S.round_lobe(px(0.72 * fh), px(0.42 * fh), angle=math.radians(40)), "WingUp"),
                  (S.round_lobe(px(0.46 * fh), px(0.28 * fh), angle=math.radians(-16)), "WingLow")):
        L = L + np.array(fp(0.05, 0.2))
        add(C.relief(f"{name}_Fairy{nm}", L, px(2.5), radius=px(4), mat=wing_mat, skirt=0.002))
        rim = C.resample(L, count=80)
        add(D.stem_relief(np.vstack([rim, rim[:1]]), px(1.6), gold, name=f"{name}_FairyWingRim"), lift=0.5)
    # flowers, sprigs, stars and hearts
    teal = P["teal"]
    for (cx, cz, s, pm, im) in ((-74, -84, 56, teal, None), (143, -64, 50, P["pink_light"], P["pink"])):
        stem = np.array([(cx, cz - s * 0.3), (cx + 4, cz - s * 0.9), (cx + 8, cz - s * 1.5)])
        stem = C.catmull_rom(stem, 6)
        add(D.stem_relief(px(stem), px(2.6), P["leaf"], name=f"{name}_Stem"))
        for side, t in ((1, 0.55), (-1, 0.8)):
            q = stem[int(len(stem) * t) - 1]
            lf = C.rot2(S.leaf(px(22), px(11), bend=0.1), -side * 0.9) + px(q)
            add(C.relief(f"{name}_Leaf", lf, px(3), radius=px(5), mat=P["leaf"], skirt=0.002))
        fl = D.flower(px(s), pm, P["yellow"], inner_mat=im, name=f"{name}_Flower")
        dec.extend(fl, offset=(px(cx), px(cz)))
    for (cx, cz, a, n) in ((-106, -30, -1.3, 4), (142, -2, 1.2, 3)):
        stem = np.array([(cx, cz - 40), (cx - 3, cz), (cx + 6, cz + 36)])
        stem = C.catmull_rom(stem, 5)
        add(D.stem_relief(px(stem), px(2.0), P["leaf"], name=f"{name}_Sprig"))
        for i in range(n):
            q = stem[int((i + 0.5) / n * (len(stem) - 1))]
            side = 1 if i % 2 else -1
            lf = C.rot2(S.leaf(px(18), px(9), bend=0.1), side * 0.9) + px(q)
            add(C.relief(f"{name}_Leaf", lf, px(3), radius=px(4.5), mat=P["leaf"], skirt=0.002))
    pink = M.plastic("M_FairyKit_BookPinkStar", "#FF7FC0", rough=0.3, coat=0.6, variation=0.02)
    for (cx, cz, s, m) in ((-109, 32, 30, gold), (114, 85, 32, pink), (-40, -50, 18, pink), (111, -120, 22, pink)):
        dec.extend(D.sparkle_star(px(s), m, name=f"{name}_Star"), offset=(px(cx), px(cz)))
    for (cx, cz, s) in ((66, 130, 24), (-19, -138, 24)):
        dec.extend(D.heart_decal(px(s), pink, name=f"{name}_Heart"), offset=(px(cx), px(cz)))
    rng = np.random.default_rng(31)
    for i in range(22):
        cx, cz = rng.uniform(-150, 150), rng.uniform(-150, 150)
        if -120 < cx < 120 and 0 < cz < 120:
            continue
        sp4 = S.star(px(rng.uniform(3.5, 6.5)), px(1.2), points=4)
        add(C.relief(f"{name}_Sparkle", sp4 + px(np.array([cx, cz])), px(1.8), radius=px(1.5), mat=gold,
                     skirt=0.002))
    dec.flat(front, parent=root)
    return root


# ---------------------------------------------------------------------------
# Scalloped gem bowl
# ---------------------------------------------------------------------------

GEM_COLORS = {
    "pink": ("#FF6DBE", 0.0), "magenta": ("#E8198F", 0.0), "blue": ("#1EA8F0", 0.0),
    "aqua": ("#9EEFEA", 380.0), "purple": ("#9B30E0", 0.0), "teal": ("#1FD6C8", 0.0),
    "clear": ("#EDE2FF", 450.0),
}


def gem_mat(key):
    hexc, film = GEM_COLORS[key]
    # partly opaque resin: keeps each stone's colour distinct when stones overlap in the bowl
    return M.glass(f"M_FairyKit_Gem_{key.capitalize()}", hexc, ior=1.85, film=film, glow=0.2, transmission=0.82)


def gem_bowl():
    name = "SM_Prop_FairyPotion_GemBowl"
    root = C.empty(name)
    mat = M.plastic("M_FairyKit_Bowl", "#B98AEF", rough=0.3, coat=0.45, variation=0.03, sparkle=0.05)
    lobes = 11
    deep = 1.3  # the photo's dish has taller fluted walls than a shallow saucer
    H = 64.0 * deep
    prof = [(r, z * deep) for r, z in
            [(0, 0), (60, 0), (68, 3), (72, 9), (84, 14), (106, 24), (126, 37), (141, 50), (151, 60), (154, 65),
             (151, 69), (145, 67), (138, 60), (124, 49), (104, 39), (78, 32), (44, 28), (0, 27)]]

    def rmod(th, z, r):
        f = min(max(z / px(H), 0.0), 1.0) ** 1.4
        return 1.0 + 0.07 * f * (abs(math.cos(lobes * th / 2)) ** 0.8 * 2 - 1.2)

    def zmod(th, z, r):
        f = min(max((z - px(50 * deep)) / px(19 * deep), 0.0), 1.0)
        return px(5) * f * (abs(math.cos(lobes * th / 2)) ** 0.8)

    bowl = C.lathe(f"{name}_Bowl", [(px(r), px(z)) for r, z in prof], segments=lobes * 14, radial_mod=rmod,
                   z_mod=zmod, mat=mat, smooth_samples=5)
    C.subsurf(bowl, 1)
    bowl.parent = root
    rng = np.random.default_rng(42)
    layout = [  # (x, y, z, radius, colour)
        (-70, -40, 42, 40, "blue"), (-20, -62, 40, 40, "purple"), (40, -60, 42, 42, "blue"), (92, -30, 44, 38, "teal"),
        (-100, 10, 46, 36, "blue"), (100, 25, 46, 36, "blue"), (-55, 40, 44, 40, "pink"), (0, 55, 44, 40, "magenta"),
        (60, 45, 44, 40, "clear"), (-10, -5, 40, 40, "teal"),
        (-50, -20, 78, 46, "pink"), (10, -35, 80, 44, "blue"), (58, -5, 80, 42, "aqua"), (-5, 25, 86, 44, "magenta"),
        (-78, 8, 76, 36, "purple"), (84, 10, 76, 36, "clear"), (30, 30, 100, 40, "clear"), (-35, -52, 72, 38, "pink"),
        (70, -48, 70, 34, "teal"),
    ]
    for i, (x, y, z, r, key) in enumerate(layout):
        g = C.nugget_gem(f"{name}_Gem", px(r * 1.12), seed=100 + i, points=16, squash=(1.0, 0.95, 0.85), mat=gem_mat(key))
        rot = Matrix.Rotation(rng.uniform(0, math.tau), 4, "Z") @ Matrix.Rotation(rng.uniform(-0.6, 0.6), 4, "X")
        C.transform_mesh(g, Matrix.Translation((px(x), px(y), px(z + (deep - 1.0) * 40))) @ rot)
        g.parent = root
    return root


# ---------------------------------------------------------------------------
# Loose faceted gems (hearts and stars)
# ---------------------------------------------------------------------------

def loose_gem(kind, color, size_px, name):
    mat = M.glass(f"M_FairyKit_LooseGem_{color.strip('#')}", color, ior=1.85, glow=0.2, transmission=0.82)
    if kind == "heart":
        g = C.faceted_outline_gem(name, S.heart(px(size_px), px(size_px * 0.86), plump=0.5), crown=0.28, pavilion=0.3,
                                  table=0.58, girdle=0.1, facet_points=22, mat=mat)
    else:
        g = C.ridge_star(name, px(size_px / 2), px(size_px / 2 * 0.58), px(size_px * 0.16), table=0.32,
                         back=px(size_px * 0.14), wall=px(size_px * 0.1), mat=mat)
    return g


def loose_gems():
    name = "SM_Prop_FairyPotion_LooseGems"
    root = C.empty(name)
    specs = [("heart", "#FF4FAE", 104, "PinkHeart"), ("star", "#B774EC", 92, "PurpleStar"),
             ("star", "#5CC8F6", 86, "BlueStar"), ("heart", "#B04BE2", 104, "PurpleHeart"),
             ("heart", "#38D9CC", 104, "TealHeart")]
    for kind, col, size, nm in specs:
        g = loose_gem(kind, col, size, f"{name}_{nm}")
        g.parent = root
    return root


# ---------------------------------------------------------------------------
# Star wand
# ---------------------------------------------------------------------------

def star_wand():
    P = palette()
    name = "SM_Prop_FairyPotion_StarWand"
    root = C.empty(name)
    gold = P["gold"]
    handle = M.plastic("M_FairyKit_WandHandle", "#FAA2CB", rough=0.28, coat=0.8, variation=0.02)
    petal = M.plastic("M_FairyKit_WandPetal", "#FF8FC4", rough=0.28, coat=0.8, variation=0.02)
    gem = M.glass("M_FairyKit_GemWandPink", "#FF5CB8", ior=1.85, glow=0.15)
    R = 88.0
    outer = S.star(px(R), px(R * 0.54), rounding=0.4)
    hole = S.star(px(R * 0.68), px(R * 0.68 * 0.54), rounding=0.2)
    frame = D.two_sided(f"{name}_StarFrame", outer, px(26), gold, radius=px(11), holes=[hole], spacing=px(2.2))
    C.transform_mesh(frame, Matrix.Rotation(-math.pi / 2, 4, "X"))  # lie flat, face up (+Z)
    frame.parent = root
    st = C.ridge_star(f"{name}_StarGem", px(R * 0.7), px(R * 0.7 * 0.54), px(10), table=0.3, back=px(8), wall=px(8),
                      mat=gem)
    st.parent = root
    for i in range(5):
        a = math.pi / 2 + math.pi / 5 + i * math.tau / 5
        C.uv_sphere(f"{name}_Rivet", px(4.5), 12, 7, mat=gold,
                    location=(px(R * 0.62 * math.cos(a)), px(R * 0.62 * math.sin(a)), px(13))).parent = root
    # handle along +X/-Y (down-right in the photo), starting at the lower-right inner corner
    ang = math.pi / 2 + math.pi / 5 + 3 * math.tau / 5  # inner vertex between the two lower-right points
    d = Vector((math.cos(ang), math.sin(ang), 0.0))
    start = d * px(R * 0.5)
    prof =[(0, 0), (22, 0), (24, 6), (22, 30), (20, 36), (19, 60), (19, 310), (21, 322), (26, 336), (27, 348),
            (24, 360), (14, 367), (0, 369)]
    shaft = C.lathe(f"{name}_Handle", [(px(r), px(z)) for r, z in prof], segments=32, mat=handle, smooth_samples=4)
    C.subsurf(shaft, 1)
    cap = C.lathe(f"{name}_Cap", [(0, px(-6)), (px(21), px(-6)), (px(23), px(4)), (px(22), px(36)), (px(19), px(42)),
                                  (0, px(42))], segments=32, mat=gold, smooth_samples=3)
    C.subsurf(cap, 1)
    groove = C.torus(f"{name}_Groove", px(22), px(2.6), 32, 8, mat=handle)
    C.transform_mesh(groove, Matrix.Translation((0, 0, px(326))))
    bow = []
    for k in range(5):
        a = k * math.tau / 5
        p = C.uv_sphere(f"{name}_BowPetal", 1.0, 16, 8, mat=petal, scale=(px(17), px(11), px(9)))
        C.transform_mesh(p, Matrix.Translation((0, 0, px(78))) @ Matrix.Rotation(a, 4, "Z")
                         @ Matrix.Translation((px(20), 0, 0)) @ Matrix.Rotation(0.3, 4, "Y"))
        bow.append(p)
    ring = C.lathe(f"{name}_BowRing", [(0, px(64)), (px(21), px(64)), (px(23), px(72)), (px(23), px(84)),
                                       (px(21), px(92)), (0, px(92))], segments=32, mat=gold, smooth_samples=3)
    bud = C.uv_sphere(f"{name}_BowBud", px(8), 12, 8, mat=gold, location=(0, -px(28), px(78)))
    parts = [shaft, cap, groove, ring, bud] + bow
    orient = Vector((0, 0, 1)).rotation_difference(d).to_matrix().to_4x4()
    for o in parts:
        C.transform_mesh(o, Matrix.Translation(start) @ orient)
        o.parent = root
    return root


# ---------------------------------------------------------------------------
# Gold scoop spoon
# ---------------------------------------------------------------------------

def gold_spoon():
    name = "SM_Prop_FairyPotion_GoldScoop"
    root = C.empty(name)
    gold = M.gold()
    prof = [(0, 0), (40, 2), (60, 8), (72, 16), (78, 25), (77, 30), (72, 29), (64, 22), (48, 15), (26, 11), (0, 10)]
    bowl = C.lathe(f"{name}_Bowl", [(px(r), px(z)) for r, z in prof], segments=64, depth_ratio=0.84, mat=gold,
                   smooth_samples=5)
    C.subsurf(bowl, 1)
    bowl.parent = root
    surf = C.Surface([bowl])
    dec = D.Decal()
    dec.add(C.relief(f"{name}_Heart", S.heart(px(52), px(46)), px(4.5), radius=px(6), mat=gold, skirt=0.002))
    dec.place(surf, u=px(-6), v=0.0, tilt=math.pi / 2, parent=root, lift=0.0)
    for i in range(11):  # ring of raised scroll loops around the inside of the rim
        a = math.radians(-150 + i * 30)
        cx, cy = math.cos(a) * px(55), math.sin(a) * px(55) * 0.84
        loop = S.scroll((0, 0), a + 1.2, px(5), px(7.5), turns=0.95, handed=1, n=26) + np.array([cx, cy])
        d2 = D.Decal()
        d2.add(D.stem_relief(loop, px(2.6), gold, name=f"{name}_Engrave"))
        d2.place(surf, u=0.0, v=0.0, tilt=math.pi / 2, parent=root)
    # handle: flattened tube rising gently from the bowl rim
    t = np.linspace(0, 1, 40)
    path = np.stack([px(70) + px(205) * t, np.zeros_like(t), px(22) + px(12) * np.sin(t * math.pi * 0.5)], axis=1)
    radii = px(12) + px(9) * t ** 1.5
    radii[:4] = px(np.array([16, 13, 11.5, 11]))
    h = C.tube(f"{name}_Handle", path, px(12), sides=16, radii=radii, mat=gold, flatten=0.55, up=(0, 0, 1))
    C.subsurf(h, 1)
    h.parent = root
    for x, r in ((96, 13.5), (104, 13.5)):
        ring = C.torus(f"{name}_Collar", px(r), px(3), 24, 8, mat=gold)
        C.transform_mesh(ring, Matrix.Translation((px(x), 0, px(22) + px(12) * math.sin((x - 70) / 205 * math.pi / 2)))
                         @ Matrix.Rotation(math.pi / 2, 4, "Y") @ Matrix.Diagonal((0.6, 1, 1, 1)))
        ring.parent = root
    # rose / scroll embossing along the top of the handle
    hs = C.Surface([h])
    for i, x in enumerate(np.linspace(125, 255, 5)):
        loop = S.scroll((0, 0), 0.4 * i, px(3), px(6 + i * 0.8), turns=1.3, handed=1 if i % 2 else -1, n=28)
        d2 = D.Decal()
        d2.add(D.stem_relief(loop, px(1.8), gold, name=f"{name}_Rose"))
        d2.place(hs, u=px(x), v=px(60), tilt=math.pi / 2, parent=root)
    return root


# ---------------------------------------------------------------------------
# Fairy-wing hair clip
# ---------------------------------------------------------------------------

def wing_clip():
    P = palette()
    name = "SM_Prop_FairyPotion_WingClip"
    root = C.empty(name)
    frame_mat = M.plastic("M_FairyKit_WingClipFrame", "#F47AAE", rough=0.26, coat=0.85, variation=0.02)
    band_mat = M.plastic("M_FairyKit_WingClipBand", "#B25CDE", rough=0.3, coat=0.7, variation=0.03)
    petal = M.plastic("M_FairyKit_WingClipPetal", "#FFD2E6", rough=0.3, coat=0.7, variation=0.02)
    irid = M.iridescent("M_FairyKit_WingIridescent")
    gold = P["gold"]
    parts = []
    lobes = {
        "RightUp": S.round_lobe(px(240), px(150), angle=math.radians(58), skew=0.15),
        "RightLow": S.round_lobe(px(165), px(100), angle=math.radians(8)),
        "LeftUp": S.round_lobe(px(240), px(130), angle=math.radians(170), skew=-0.1),
        "LeftLow": S.round_lobe(px(160), px(92), angle=math.radians(212)),
    }
    for nm, L in lobes.items():
        L = C.ccw(L)
        ring = C.offset_closed(L, px(10))
        hole = C.offset_closed(L, -px(4))
        parts.append(C.relief(f"{name}_{nm}Frame", ring, px(9), radius=px(8), holes=[hole], two_sided=True,
                              mat=frame_mat, spacing=px(2.5)))
        mem = C.relief(f"{name}_{nm}Membrane", C.offset_closed(L, -px(2)), px(2.5), radius=px(4), two_sided=True,
                       mat=irid, spacing=px(4))
        parts.append(mem)
        pts = C.resample(L, count=200)
        tip = pts[np.argmax(np.linalg.norm(pts, axis=1))]
        base_ang = math.atan2(tip[1], tip[0])
        ln = np.linalg.norm(tip)
        for f, frac, bow in ((-0.22, 0.78, -30), (0.0, 0.9, 10), (0.24, 0.72, 30)):
            # veins bow outwards and follow the lobe like the moulded veins on the photo's clip
            a = base_ang + f
            vp = np.array([[math.cos(a) * ln * frac * t - math.sin(a) * px(bow) * math.sin(t * math.pi * 0.9),
                            -px(4),
                            math.sin(a) * ln * frac * t + math.cos(a) * px(bow) * math.sin(t * math.pi * 0.9)]
                           for t in np.linspace(0.15, 1.0, 18)])
            parts.append(C.tube(f"{name}_{nm}Vein", vp, px(2.6), sides=6, mat=gold))
            end = vp[-1]
            curl = S.scroll((0, 0), a + math.pi * 0.6, px(4), px(9), turns=0.8, handed=1 if f >= 0 else -1, n=16)
            cp = np.stack([end[0] + curl[:, 0], np.full(len(curl), -px(4)), end[2] + curl[:, 1]], axis=1)
            parts.append(C.tube(f"{name}_{nm}VeinCurl", cp, px(2.2), sides=6, mat=gold))
    # purple band arcing under the flower
    t = np.linspace(-1, 1, 40)
    band = np.stack([px(170) * t + px(20), -px(10) * np.ones_like(t),
                     -px(40) * t * t + px(10) - px(40) * (t < 0) * t * t], axis=1)
    band[:, 2] -= px(12)
    parts.append(C.tube(f"{name}_Band", band, px(15), sides=14, mat=band_mat, flatten=0.8))
    # centre flower
    fl = D.flower(px(92), petal, P["yellow"], petals=5, style="daisy", name=f"{name}_Flower", height=px(12))
    for o, lift in fl.items:
        V = C.verts_np(o)
        V[:, 1] -= lift + px(22)
        C.set_verts_np(o, V)
        parts.append(o)
    # lay flat on the table (XZ plane -> XY plane, facing up, design-up pointing away)
    for o in parts:
        C.transform_mesh(o, Matrix.Rotation(-math.pi / 2, 4, "X"))
        o.parent = root
    return root


PROPS = {
    "SpellBook": fairy_book,
    "GemBowl": gem_bowl,
    "LooseGems": loose_gems,
    "StarWand": star_wand,
    "GoldScoop": gold_spoon,
    "WingClip": wing_clip,
}
