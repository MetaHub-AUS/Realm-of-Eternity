"""Build the Fairy Potion Kit scene procedurally in Blender.

Run with Blender (4.2+ / 5.x):
    blender --background --python build_fairy_potion_kit.py -- [options]
or with the `bpy` Python module (pip install bpy):
    python build_fairy_potion_kit.py [options]

Options:
    --blend PATH      where to save the .blend (default: FairyPotionKit.blend next to this script)
    --render PATH     also render the hero shot to PATH (PNG, composited on white)
    --samples N       Cycles samples for the render (default 128)
    --res N           square render resolution in px (default 1254, same as the reference)
    --only KEY,...    build only some assets (e.g. PetalGlow,SpellBook); layout still applies
    --export-fbx DIR  export one FBX per asset (for Unreal) into DIR
"""

import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import bpy  # noqa: E402
from mathutils import Matrix  # noqa: E402

from fairykit import bottles, core as C, decor as D, layout, props, scene as SC  # noqa: E402

ASSET_BUILDERS = {**bottles.BOTTLES, **props.PROPS}


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--blend", default=os.path.join(HERE, "FairyPotionKit.blend"))
    ap.add_argument("--render", default=None)
    ap.add_argument("--samples", type=int, default=128)
    ap.add_argument("--res", type=int, default=1254)
    ap.add_argument("--only", default=None)
    ap.add_argument("--export-fbx", default=None)
    return ap.parse_args(argv)


def build_assets(keys=None):
    kit = C.new_collection("FairyPotionKit")
    roots = {}
    for key, builder in ASSET_BUILDERS.items():
        if keys and key not in keys:
            continue
        t0 = time.time()
        coll = C.new_collection(key, kit)
        C.set_collection(coll)
        root = builder()
        roots[key] = root
        print(f"  built {key:16s} {time.time() - t0:5.1f}s")
    C.set_collection(None)
    return roots


def snap_to_table(obj):
    """Shift an asset vertically so its lowest vertex rests on the table (z = 0)."""
    bpy.context.view_layer.update()
    zmin = min((o.matrix_world @ v.co).z for o in [obj] + list(obj.children_recursive) if o.type == "MESH"
               for v in o.data.vertices)
    obj.location.z -= zmin


def apply_layout(roots):
    for key, root in roots.items():
        if key == "LooseGems":
            # each loose gem is placed individually
            for child in list(root.children):
                sub = f"LooseGems/{child.name.rsplit('_', 1)[-1]}"
                if sub in layout.PLACEMENT:
                    C.set_collection(child.users_collection[0])
                    pivot = C.empty(child.name + "_Pivot", parent=root)
                    C.set_collection(None)
                    child.parent = pivot
                    layout.apply(pivot, sub)
                    snap_to_table(pivot)
        elif key in layout.PLACEMENT:
            layout.apply(root, key)
            snap_to_table(root)


def setup_scene(scene, res, samples):
    SC.render_settings(scene, res=res, samples=samples)
    SC.world(scene)
    C.set_collection(C.new_collection("Studio"))
    cam = layout.CAMERA
    SC.orbit_camera(cam["target"], cam["distance"], cam["elevation"], cam["azimuth"], lens=cam["lens"])
    SC.studio_lights(center=cam["target"])
    SC.floor()
    C.set_collection(None)


def export_fbx(roots, out_dir):
    """One FBX per asset at its own origin and native size (the hero layout transform is not exported)."""
    os.makedirs(out_dir, exist_ok=True)
    for key, root in roots.items():
        objs = [root] + list(root.children_recursive)
        hero = root.matrix_world.copy()
        root.matrix_world = Matrix.Identity(4)
        bpy.context.view_layer.update()
        with bpy.context.temp_override(selected_objects=objs, active_object=root):
            bpy.ops.object.select_all(action="DESELECT")
            for o in objs:
                o.select_set(True)
            bpy.ops.export_scene.fbx(filepath=os.path.join(out_dir, f"{root.name}.fbx"), use_selection=True,
                                     apply_unit_scale=True, apply_scale_options="FBX_SCALE_UNITS",
                                     use_mesh_modifiers=True, mesh_smooth_type="FACE", object_types={"MESH", "EMPTY"})
        root.matrix_world = hero
        bpy.context.view_layer.update()


def render(scene, path):
    scene.render.filepath = path
    bpy.ops.render.render(write_still=True)
    try:  # composite the transparent render (with shadow-catcher shadows) onto white
        from PIL import Image
        im = Image.open(path)
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
        bg.alpha_composite(im)
        bg.convert("RGB").save(path)
    except ImportError:
        print("Pillow not available: render left with transparent background")


def main():
    args = parse_args()
    t0 = time.time()
    scene = SC.reset()
    fonts = os.path.join(HERE, "fonts")
    D.set_fonts(os.path.join(fonts, "Chewy-Regular.ttf"), os.path.join(fonts, "BerkshireSwash-Regular.ttf"))
    keys = set(args.only.split(",")) if args.only else None
    print("Building Fairy Potion Kit ...")
    roots = build_assets(keys)
    apply_layout(roots)
    setup_scene(scene, args.res, args.samples)
    bpy.context.view_layer.update()
    nverts = sum(len(o.data.vertices) for o in scene.objects if o.type == "MESH")
    print(f"Built {len(roots)} assets, {len(scene.objects)} objects, {nverts} base vertices "
          f"in {time.time() - t0:.1f}s")
    if args.blend:
        bpy.ops.file.pack_all()
        bpy.ops.wm.save_as_mainfile(filepath=args.blend, compress=True)
        print(f"Saved {args.blend}")
    if args.export_fbx:
        export_fbx(roots, args.export_fbx)
    if args.render:
        t1 = time.time()
        render(scene, args.render)
        print(f"Rendered {args.render} in {time.time() - t1:.1f}s")


if __name__ == "__main__":
    main()
