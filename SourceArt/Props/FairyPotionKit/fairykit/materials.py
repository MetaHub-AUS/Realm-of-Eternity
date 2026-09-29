"""Procedural Cycles/EEVEE materials for the Fairy Potion Kit (no image textures)."""

import bpy

_cache = {}


def srgb(hexstr, alpha=1.0):
    h = hexstr.lstrip("#")
    out = []
    for i in (0, 2, 4):
        c = int(h[i:i + 2], 16) / 255.0
        out.append(c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4)
    return (*out, alpha)


def _new(name):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    out.location = (600, 0)
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.location = (250, 0)
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    return mat, nt, bsdf


def _set(bsdf, **kw):
    names = {
        "base": "Base Color", "metallic": "Metallic", "rough": "Roughness", "ior": "IOR",
        "coat": "Coat Weight", "coat_rough": "Coat Roughness", "sheen": "Sheen Weight",
        "sheen_rough": "Sheen Roughness", "transmission": "Transmission Weight",
        "sss": "Subsurface Weight", "sss_scale": "Subsurface Scale", "film": "Thin Film Thickness",
        "film_ior": "Thin Film IOR", "spec": "Specular IOR Level", "emit": "Emission Color",
        "emit_strength": "Emission Strength", "alpha": "Alpha",
    }
    for k, v in kw.items():
        bsdf.inputs[names[k]].default_value = v


def _noise_variation(nt, bsdf, color, amount=0.06, scale=18.0, bump=0.04, bump_scale=260.0,
                     sparkle=0.0, sparkle_color=None):
    """Subtle large-scale tint variation + micro bump (+ optional glitter speckles)."""
    tc = nt.nodes.new("ShaderNodeTexCoord")
    tc.location = (-1100, 0)
    noise = nt.nodes.new("ShaderNodeTexNoise")
    noise.location = (-850, 200)
    noise.inputs["Scale"].default_value = scale
    nt.links.new(tc.outputs["Object"], noise.inputs["Vector"])
    mix = nt.nodes.new("ShaderNodeMix")
    mix.data_type = "RGBA"
    mix.blend_type = "OVERLAY"
    mix.location = (-500, 200)
    mix.inputs["Factor"].default_value = amount
    mix.inputs["A"].default_value = color
    nt.links.new(noise.outputs["Color"], mix.inputs["B"])
    col_out = mix.outputs["Result"]
    if sparkle > 0:
        vor = nt.nodes.new("ShaderNodeTexVoronoi")
        vor.location = (-850, -250)
        vor.inputs["Scale"].default_value = 900.0
        nt.links.new(tc.outputs["Object"], vor.inputs["Vector"])
        ramp = nt.nodes.new("ShaderNodeMapRange")
        ramp.location = (-650, -250)
        ramp.inputs["From Min"].default_value = 0.08
        ramp.inputs["From Max"].default_value = 0.0
        nt.links.new(vor.outputs["Distance"], ramp.inputs["Value"])
        mul = nt.nodes.new("ShaderNodeMath")
        mul.operation = "MULTIPLY"
        mul.inputs[1].default_value = sparkle
        mul.location = (-450, -250)
        nt.links.new(ramp.outputs["Result"], mul.inputs[0])
        mix2 = nt.nodes.new("ShaderNodeMix")
        mix2.data_type = "RGBA"
        mix2.location = (-250, 150)
        mix2.inputs["B"].default_value = sparkle_color or (1.0, 1.0, 1.0, 1.0)
        nt.links.new(mul.outputs["Value"], mix2.inputs["Factor"])
        nt.links.new(col_out, mix2.inputs["A"])
        col_out = mix2.outputs["Result"]
    nt.links.new(col_out, bsdf.inputs["Base Color"])
    if bump > 0:
        n2 = nt.nodes.new("ShaderNodeTexNoise")
        n2.location = (-850, -550)
        n2.inputs["Scale"].default_value = bump_scale
        n2.inputs["Detail"].default_value = 4.0
        nt.links.new(tc.outputs["Object"], n2.inputs["Vector"])
        b = nt.nodes.new("ShaderNodeBump")
        b.location = (-250, -400)
        b.inputs["Strength"].default_value = bump
        b.inputs["Distance"].default_value = 0.002
        nt.links.new(n2.outputs["Fac"], b.inputs["Height"])
        nt.links.new(b.outputs["Normal"], bsdf.inputs["Normal"])
        return b
    return None


def plastic(name, hexcolor, rough=0.3, coat=0.7, sparkle=0.0, variation=0.07, bump=0.03, sheen=0.0):
    coat = coat * 0.6  # lacquer layer: keep highlights crisp without milking out the base colour
    key = ("plastic", name)
    if key in _cache:
        return _cache[key]
    mat, nt, bsdf = _new(name)
    col = srgb(hexcolor)
    _set(bsdf, base=col, rough=rough, coat=coat, coat_rough=0.06, spec=0.5)
    if sheen:
        _set(bsdf, sheen=sheen, sheen_rough=0.4)
    _noise_variation(nt, bsdf, col, amount=variation, bump=bump, sparkle=sparkle)
    mat.diffuse_color = col
    _cache[key] = mat
    return mat


def gold(name="M_FairyKit_Gold", hexcolor="#F2B640", rough=0.2):
    key = ("gold", name)
    if key in _cache:
        return _cache[key]
    mat, nt, bsdf = _new(name)
    col = srgb(hexcolor)
    _set(bsdf, base=col, metallic=1.0, rough=rough, coat=0.25, coat_rough=0.05)
    tc = nt.nodes.new("ShaderNodeTexCoord")
    noise = nt.nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 60.0
    nt.links.new(tc.outputs["Object"], noise.inputs["Vector"])
    rng = nt.nodes.new("ShaderNodeMapRange")
    rng.inputs["To Min"].default_value = rough * 0.6
    rng.inputs["To Max"].default_value = rough * 1.5
    nt.links.new(noise.outputs["Fac"], rng.inputs["Value"])
    nt.links.new(rng.outputs["Result"], bsdf.inputs["Roughness"])
    mat.diffuse_color = col
    _cache[key] = mat
    return mat


def glass(name, hexcolor, ior=1.5, rough=0.03, film=0.0, absorb=None, glow=0.35, transmission=1.0):
    key = ("glass", name)
    if key in _cache:
        return _cache[key]
    mat, nt, bsdf = _new(name)
    col = srgb(hexcolor)
    _set(bsdf, base=col, transmission=transmission, rough=rough, ior=ior, coat=0.0)
    if glow:
        # faint internal glow: stands in for light scattering inside toy resin gems
        _set(bsdf, emit=col, emit_strength=glow)
    if film:
        _set(bsdf, film=film, film_ior=1.35)
    if absorb is not None:
        vol = nt.nodes.new("ShaderNodeVolumeAbsorption")
        vol.inputs["Color"].default_value = srgb(absorb)
        vol.inputs["Density"].default_value = 6.0
        out = [n for n in nt.nodes if n.type == "OUTPUT_MATERIAL"][0]
        nt.links.new(vol.outputs["Volume"], out.inputs["Volume"])
    mat.diffuse_color = (*col[:3], 0.8)
    mat.blend_method = "BLEND" if hasattr(mat, "blend_method") else None
    _cache[key] = mat
    return mat


def cream(name="M_FairyKit_LabelCream", hexcolor="#FDEBD9"):
    return plastic(name, hexcolor, rough=0.42, coat=0.25, variation=0.03, bump=0.02)


def cork(name="M_FairyKit_Cork"):
    key = ("cork", name)
    if key in _cache:
        return _cache[key]
    mat, nt, bsdf = _new(name)
    _set(bsdf, rough=0.85, spec=0.3)
    tc = nt.nodes.new("ShaderNodeTexCoord")
    vor = nt.nodes.new("ShaderNodeTexVoronoi")
    vor.feature = "DISTANCE_TO_EDGE"
    vor.inputs["Scale"].default_value = 55.0
    nt.links.new(tc.outputs["Object"], vor.inputs["Vector"])
    crack = nt.nodes.new("ShaderNodeMapRange")
    crack.inputs["From Min"].default_value = 0.0
    crack.inputs["From Max"].default_value = 0.06
    nt.links.new(vor.outputs["Distance"], crack.inputs["Value"])
    noise = nt.nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 90.0
    noise.inputs["Detail"].default_value = 8.0
    nt.links.new(tc.outputs["Object"], noise.inputs["Vector"])
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].position = 0.3
    ramp.color_ramp.elements[0].color = srgb("#A9612B")
    ramp.color_ramp.elements[1].position = 0.75
    ramp.color_ramp.elements[1].color = srgb("#D99A5C")
    nt.links.new(noise.outputs["Fac"], ramp.inputs["Fac"])
    mix = nt.nodes.new("ShaderNodeMix")
    mix.data_type = "RGBA"
    mix.inputs["A"].default_value = srgb("#6E3814")
    nt.links.new(crack.outputs["Result"], mix.inputs["Factor"])
    nt.links.new(ramp.outputs["Color"], mix.inputs["B"])
    nt.links.new(mix.outputs["Result"], bsdf.inputs["Base Color"])
    add = nt.nodes.new("ShaderNodeMath")
    add.operation = "ADD"
    nt.links.new(crack.outputs["Result"], add.inputs[0])
    mul = nt.nodes.new("ShaderNodeMath")
    mul.operation = "MULTIPLY"
    mul.inputs[1].default_value = 0.3
    nt.links.new(noise.outputs["Fac"], mul.inputs[0])
    nt.links.new(mul.outputs["Value"], add.inputs[1])
    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.6
    bump.inputs["Distance"].default_value = 0.01
    nt.links.new(add.outputs["Value"], bump.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    mat.diffuse_color = srgb("#C98A4B")
    _cache[key] = mat
    return mat


def velvet(name, hexcolor):
    key = ("velvet", name)
    if key in _cache:
        return _cache[key]
    mat, nt, bsdf = _new(name)
    col = srgb(hexcolor)
    _set(bsdf, base=col, rough=0.55, coat=0.12, coat_rough=0.2, sheen=0.5, sheen_rough=0.35)
    _noise_variation(nt, bsdf, col, amount=0.08, scale=10.0, bump=0.12, bump_scale=420.0)
    mat.diffuse_color = col
    _cache[key] = mat
    return mat


def iridescent(name="M_FairyKit_WingIridescent", stops=None, glitter=True, scale_axis="X",
               transmission=0.0):
    """Pastel rainbow membrane with glitter flecks (fairy wings)."""
    key = ("irid", name)
    if key in _cache:
        return _cache[key]
    mat, nt, bsdf = _new(name)
    _set(bsdf, rough=0.25, coat=1.0, coat_rough=0.03, film=320.0, film_ior=1.4, transmission=transmission)
    tc = nt.nodes.new("ShaderNodeTexCoord")
    noise = nt.nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 3.0
    noise.inputs["Detail"].default_value = 2.0
    nt.links.new(tc.outputs["Object"], noise.inputs["Vector"])
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(tc.outputs["Object"], sep.inputs["Vector"])
    add = nt.nodes.new("ShaderNodeMath")
    add.operation = "MULTIPLY_ADD"
    add.inputs[1].default_value = 1.6
    nt.links.new(sep.outputs[scale_axis], add.inputs[0])
    nt.links.new(noise.outputs["Fac"], add.inputs[2])
    frac = nt.nodes.new("ShaderNodeMath")
    frac.operation = "PINGPONG"
    frac.inputs[1].default_value = 1.0
    nt.links.new(add.outputs["Value"], frac.inputs[0])
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    stops = stops or ["#F48ED0", "#B98DF0", "#6FC4F4", "#8EE8D2", "#F0A4E2"]
    cr = ramp.color_ramp
    for i, hexc in enumerate(stops):
        pos = i / (len(stops) - 1)
        el = cr.elements[i] if i < 2 else cr.elements.new(pos)
        el.position = pos
        el.color = srgb(hexc)
    nt.links.new(frac.outputs["Value"], ramp.inputs["Fac"])
    col_out = ramp.outputs["Color"]
    if glitter:
        vor = nt.nodes.new("ShaderNodeTexVoronoi")
        vor.inputs["Scale"].default_value = 140.0
        nt.links.new(tc.outputs["Object"], vor.inputs["Vector"])
        mr = nt.nodes.new("ShaderNodeMapRange")
        mr.inputs["From Min"].default_value = 0.12
        mr.inputs["From Max"].default_value = 0.02
        nt.links.new(vor.outputs["Distance"], mr.inputs["Value"])
        mix = nt.nodes.new("ShaderNodeMix")
        mix.data_type = "RGBA"
        mix.inputs["B"].default_value = srgb("#FFF3C8")
        nt.links.new(mr.outputs["Result"], mix.inputs["Factor"])
        nt.links.new(col_out, mix.inputs["A"])
        col_out = mix.outputs["Result"]
        nt.links.new(mr.outputs["Result"], bsdf.inputs["Metallic"])
    nt.links.new(col_out, bsdf.inputs["Base Color"])
    mat.diffuse_color = srgb(stops[1])
    _cache[key] = mat
    return mat
