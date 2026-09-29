"""Render every asset in FairyPotionKit.blend on its own from a three-quarter view and
assemble a contact sheet (renders/FairyPotionKit_assets.png).

    blender -b FairyPotionKit.blend --python tools/render_asset_sheet.py
    # or, with the bpy module:  python tools/render_asset_sheet.py [--samples 32] [--size 420]
"""
import argparse
import math
import os
import sys

import bpy
from mathutils import Vector

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from fairykit import scene as SC  # noqa: E402

ORDER = ["PetalGlow", "MoonlightMist", "FairyWings", "ForestBreath", "StardustElixir",
         "SpellBook", "LoveBlooms", "PixieDust", "DreamcapEssence", "TranquilTears",
         "GemBowl", "LooseGems", "StarWand", "GoldScoop", "WingClip"]


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=32)
    ap.add_argument("--size", type=int, default=420)
    ap.add_argument("--out", default=os.path.join(HERE, "renders", "FairyPotionKit_assets.png"))
    args = ap.parse_args(argv)

    if not bpy.data.filepath:
        bpy.ops.wm.open_mainfile(filepath=os.path.join(HERE, "FairyPotionKit.blend"))
    scene = bpy.context.scene
    scene.render.resolution_x = scene.render.resolution_y = args.size
    scene.cycles.samples = args.samples
    kit = bpy.data.collections["FairyPotionKit"]
    tiles = []
    for key in ORDER:
        for coll in kit.children:
            coll.hide_render = coll.name != key
        coll = bpy.data.collections[key]
        pts = [o.matrix_world @ Vector(c) for o in coll.all_objects if o.type == "MESH" for c in o.bound_box]
        lo = Vector((min(p.x for p in pts), min(p.y for p in pts), min(p.z for p in pts)))
        hi = Vector((max(p.x for p in pts), max(p.y for p in pts), max(p.z for p in pts)))
        center, size = (lo + hi) / 2, max(hi - lo)
        cam = SC.orbit_camera(center, size * 2.05, 20.0, 32.0, lens=60.0, name=f"CAM_Sheet_{key}")
        scene.camera = cam
        path = os.path.join(os.path.dirname(args.out), f"_sheet_{key}.png")
        scene.render.filepath = path
        bpy.ops.render.render(write_still=True)
        tiles.append((key, path))
        bpy.data.objects.remove(cam, do_unlink=True)
    for coll in kit.children:
        coll.hide_render = False

    from PIL import Image, ImageDraw, ImageFont
    cols, s = 5, args.size
    sheet = Image.new("RGB", (cols * s, math.ceil(len(tiles) / cols) * (s + 34)), "white")
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype(os.path.join(HERE, "fonts", "Chewy-Regular.ttf"), 24)
    except OSError:
        font = None
    for i, (key, path) in enumerate(tiles):
        im = Image.open(path).convert("RGBA")
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
        bg.alpha_composite(im)
        x, y = (i % cols) * s, (i // cols) * (s + 34)
        sheet.paste(bg.convert("RGB"), (x, y))
        draw.text((x + s // 2, y + s + 16), key, fill=(90, 60, 120), font=font, anchor="mm")
        os.remove(path)
    sheet.save(args.out)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
