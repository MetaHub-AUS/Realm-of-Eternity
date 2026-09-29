"""Solve the showcase layout so every asset lands where it sits in the reference photo.

For each asset the silhouette bounding box (left, top, right, bottom in reference-image
pixels, 1254x1254) is matched by projecting the asset's vertices through the hero camera
and solving location (x, y) and uniform scale with Levenberg-Marquardt. Bottoms hidden in
the photo are pinned so the rows stay physically consistent, and the solved placements are
checked for mesh intersections between assets (reported as INTERSECT lines).

    python tools/fit_layout.py            (prints an updated PLACEMENT table for fairykit/layout.py)
"""
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

import bpy  # noqa: E402
import numpy as np  # noqa: E402
from mathutils import Matrix  # noqa: E402
from mathutils.bvhtree import BVHTree  # noqa: E402

import build_fairy_potion_kit as B  # noqa: E402
from fairykit import layout, scene as SC  # noqa: E402

RES = 1254
# (left, top, right, bottom) of each asset in the reference photo; None = not constrained
TARGETS = {
    "PetalGlow": (72, 90, 318, 654),
    "MoonlightMist": (298, 178, 572, 616),
    "FairyWings": (522, 33, 743, 584),
    "ForestBreath": (702, 243, 985, 624),
    "StardustElixir": (965, 203, 1238, 682),
    "SpellBook": (22, 598, 446, 962),
    "LoveBlooms": (412, 553, 692, 948),
    "PixieDust": (None, 598, 820, 911),
    "DreamcapEssence": (815, 643, 1058, 952),
    "TranquilTears": (1025, 703, 1238, 988),
    "GemBowl": (128, 912, 518, 1090),
    "StarWand": (515, 945, 905, 1200),
    "GoldScoop": (735, 942, 1052, 1085),
    "WingClip": (834, 962, 1238, 1235),
    "LooseGems/PinkHeart": (240, 1108, 347, 1197),
    "LooseGems/PurpleStar": (368, 1128, 462, 1212),
    "LooseGems/BlueStar": (437, 1072, 527, 1147),
    "LooseGems/PurpleHeart": (528, 1112, 637, 1192),
    "LooseGems/TealHeart": (652, 1128, 762, 1212),
}


def camera_projector(cam):
    view = np.array(cam.matrix_world.inverted())
    f = cam.data.lens / cam.data.sensor_width * RES

    def project(P):
        Q = P @ view[:3, :3].T + view[:3, 3]
        z = -Q[:, 2]
        return np.stack([RES / 2 + f * Q[:, 0] / z, RES / 2 - f * Q[:, 1] / z], axis=1)
    return project


def local_points(obj, max_pts=None):
    inv = obj.matrix_world.inverted()
    pts = []
    for o in [obj] + list(obj.children_recursive):
        if o.type != "MESH":
            continue
        M = np.array(inv @ o.matrix_world)
        V = np.empty(len(o.data.vertices) * 3)
        o.data.vertices.foreach_get("co", V)
        V = V.reshape(-1, 3)
        pts.append(V @ M[:3, :3].T + M[:3, 3])
    P = np.vstack(pts)
    if max_pts and len(P) > max_pts:  # (never subsample by default: bbox extremes can be single vertices)
        P = P[np.random.default_rng(0).choice(len(P), max_pts, replace=False)]
    return P


def solve(key, obj, project):
    loc, rot, scale = layout.PLACEMENT[key]
    R = np.array(Matrix(obj.matrix_world.to_3x3().normalized()))
    local = local_points(obj)
    tgt = np.array([np.nan if t is None else t for t in TARGETS[key]], dtype=float)
    mask = ~np.isnan(tgt)

    def resid(p):
        x, y, ls = p
        W = (local * np.exp(ls)) @ R.T + np.array([x, y, 0.0])
        W[:, 2] -= W[:, 2].min()  # resting on the table
        uv = project(W)
        box = np.array([uv[:, 0].min(), uv[:, 1].min(), uv[:, 0].max(), uv[:, 1].max()])
        return (box - tgt)[mask]

    p = np.array([loc[0], loc[1], np.log(scale)], dtype=float)
    lam = 1e-2
    for _ in range(60):
        r = resid(p)
        J = np.zeros((len(r), 3))
        for i in range(3):
            dp = np.zeros(3)
            dp[i] = 1e-4
            J[:, i] = (resid(p + dp) - r) / 1e-4
        A = J.T @ J + lam * np.eye(3)
        step = -np.linalg.solve(A, J.T @ r)
        if np.sum(resid(p + step) ** 2) < np.sum(r ** 2):
            p, lam = p + step, lam * 0.5
        else:
            lam *= 4
        if np.linalg.norm(step) < 1e-6:
            break
    return p, resid(p)


def main():
    scene = SC.reset()
    fonts = os.path.join(HERE, "fonts")
    B.D.set_fonts(os.path.join(fonts, "Chewy-Regular.ttf"), os.path.join(fonts, "BerkshireSwash-Regular.ttf"))
    roots = B.build_assets()
    B.apply_layout(roots)
    B.setup_scene(scene, RES, 16)
    bpy.context.view_layer.update()
    project = camera_projector(scene.camera)
    objs = {}
    for key, root in roots.items():
        if key == "LooseGems":
            for pv in root.children:
                objs["LooseGems/" + pv.name.rsplit("_", 2)[-2]] = pv
        else:
            objs[key] = root
    print("PLACEMENT = {")
    solved = {}
    for key in layout.PLACEMENT:
        p, r = solve(key, objs[key], project)
        loc, rot, _ = layout.PLACEMENT[key]
        solved[key] = (p, loc)
        print(f'    "{key}": (({p[0]:.3f}, {p[1]:.3f}, 0.0), {tuple(rot)}, {np.exp(p[2]):.3f}),'
              f'  # residual px {np.round(r).astype(int).tolist()}')
    print("}")
    # exact check: triangle overlaps between assets at their solved placements
    for key, (p, loc) in solved.items():
        obj = objs[key]
        obj.location = (p[0], p[1], 0.0)
        obj.scale = (np.exp(p[2]),) * 3
    bpy.context.view_layer.update()
    for key, obj in objs.items():
        B.snap_to_table(obj)
    bpy.context.view_layer.update()
    trees = {}
    for key, obj in objs.items():
        verts, polys = [], []
        for o in [obj] + list(obj.children_recursive):
            if o.type != "MESH":
                continue
            base = len(verts)
            verts += [o.matrix_world @ v.co for v in o.data.vertices]
            polys += [[base + i for i in poly.vertices] for poly in o.data.polygons]
        trees[key] = BVHTree.FromPolygons(verts, polys)
    keys = list(objs)
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            if a.startswith("LooseGems") and b.startswith("LooseGems"):
                continue
            n = len(trees[a].overlap(trees[b]))
            if n:
                print(f"INTERSECT {a} / {b}: {n} triangle pairs")


if __name__ == "__main__":
    main()
