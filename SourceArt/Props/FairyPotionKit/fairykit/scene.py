"""Scene, lighting, camera and render setup for the Fairy Potion Kit showcase."""

import math

import bpy
from mathutils import Vector

from . import core as C


def reset():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.name = "FairyPotionKit"
    us = scene.unit_settings
    us.system = "METRIC"
    us.scale_length = 0.1  # 1 BU = 10 cm
    us.length_unit = "CENTIMETERS"
    return scene


def render_settings(scene, res=1254, samples=128, preview=False):
    scene.render.engine = "CYCLES"
    cy = scene.cycles
    cy.device = "CPU"
    cy.samples = 24 if preview else samples
    cy.use_adaptive_sampling = True
    cy.adaptive_threshold = 0.02
    cy.use_denoising = True
    cy.denoiser = "OPENIMAGEDENOISE"
    cy.max_bounces = 10
    cy.glossy_bounces = 6
    cy.transmission_bounces = 10
    cy.transparent_max_bounces = 8
    cy.caustics_reflective = False
    cy.caustics_refractive = False
    cy.blur_glossy = 1.0
    cy.sample_clamp_indirect = 8.0
    scene.render.resolution_x = res
    scene.render.resolution_y = res
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = True
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = -0.15


def world(scene, strength=0.16):
    w = bpy.data.worlds.new("W_FairyKit_Studio")
    scene.world = w
    w.use_nodes = True
    nt = w.node_tree
    bg = nt.nodes.get("Background") or nt.nodes.new("ShaderNodeBackground")
    out = nt.nodes.get("World Output") or nt.nodes.new("ShaderNodeOutputWorld")
    # soft vertical gradient: bright top, neutral horizon
    tc = nt.nodes.new("ShaderNodeTexCoord")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(tc.outputs["Generated"], sep.inputs["Vector"])
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].position = 0.45
    ramp.color_ramp.elements[0].color = (0.75, 0.74, 0.78, 1)
    ramp.color_ramp.elements[1].position = 0.75
    ramp.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1)
    nt.links.new(sep.outputs["Z"], ramp.inputs["Fac"])
    nt.links.new(ramp.outputs["Color"], bg.inputs["Color"])
    # diffuse sees a soft studio; glossy/refraction rays see a brighter white sweep so
    # glass gems and lacquered plastic read luminous like the photo
    lp = nt.nodes.new("ShaderNodeLightPath")
    g = nt.nodes.new("ShaderNodeMath")
    g.operation = "MULTIPLY_ADD"
    g.inputs[1].default_value = 0.3
    nt.links.new(lp.outputs["Is Glossy Ray"], g.inputs[0])
    t = nt.nodes.new("ShaderNodeMath")
    t.operation = "MULTIPLY_ADD"
    t.inputs[1].default_value = 0.8
    nt.links.new(lp.outputs["Is Transmission Ray"], t.inputs[0])
    g.inputs[2].default_value = strength
    nt.links.new(g.outputs["Value"], t.inputs[2])
    nt.links.new(t.outputs["Value"], bg.inputs["Strength"])
    nt.links.new(bg.outputs["Background"], out.inputs["Surface"])


def area_light(name, location, target, size, energy, color=(1, 1, 1), size_y=None):
    ld = bpy.data.lights.new(name, "AREA")
    ld.energy = energy
    ld.color = color
    ld.shape = "RECTANGLE" if size_y else "DISK"
    ld.size = size
    if size_y:
        ld.size_y = size_y
    obj = bpy.data.objects.new(name, ld)
    C.link(obj)
    obj.location = location
    look_at(obj, target)
    return obj


def look_at(obj, target):
    d = Vector(target) - Vector(obj.location)
    obj.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()


def studio_lights(center=(0, 0, 0.6)):
    """High soft key + top light cast the only shadows (short, soft contact shadows like a
    white-sweep product shot); fills, rim and a vertical strip add shape and highlights."""
    c = Vector(center)
    lights = [
        ("L_Key", (-3.0, -5.0, 10.0), 6.0, 700, True, (1, 1, 1)),
        ("L_Top", (0.0, 1.0, 11.0), 10.0, 200, True, (1, 1, 1)),
        ("L_Fill", (7.0, -6.0, 4.0), 8.0, 150, False, (1.0, 0.97, 0.98)),
        ("L_Rim", (0.0, 9.0, 6.0), 8.0, 380, False, (1, 1, 1)),
        ("L_Camera", (0.0, -10.0, 4.0), 8.0, 110, False, (1, 1, 1)),
        ("L_StripL", (-7.0, -3.0, 3.0), 1.2, 200, False, (1, 1, 1)),
    ]
    for name, off, size, energy, shadow, color in lights:
        o = area_light(name, c + Vector(off), c, size, energy, color=color)
        o.data.use_shadow = shadow


def floor(size=60.0):
    obj = C.mesh_object("Floor_ShadowCatcher", [(-size, -size, 0), (size, -size, 0), (size, size, 0),
                                                (-size, size, 0)], [(0, 1, 2, 3)], smooth=False)
    obj.is_shadow_catcher = True
    from . import materials as M
    obj.data.materials.append(M.plastic("M_FairyKit_Tabletop", "#FFFFFF", rough=0.6, coat=0.0, variation=0.0,
                                        bump=0.0))
    return obj


def camera(location, target, lens=50.0, name="CAM_Hero"):
    cd = bpy.data.cameras.new(name)
    cd.lens = lens
    cd.sensor_width = 36.0
    cd.clip_start = 0.05
    cd.clip_end = 200.0
    cam = bpy.data.objects.new(name, cd)
    C.link(cam)
    cam.location = location
    look_at(cam, target)
    bpy.context.scene.camera = cam
    return cam


def orbit_camera(target, distance, elevation_deg, azimuth_deg=0.0, lens=50.0, name="CAM_Hero"):
    el, az = math.radians(elevation_deg), math.radians(azimuth_deg)
    t = Vector(target)
    loc = t + Vector((distance * math.cos(el) * math.sin(az), -distance * math.cos(el) * math.cos(az),
                      distance * math.sin(el)))
    return camera(loc, t, lens, name)
