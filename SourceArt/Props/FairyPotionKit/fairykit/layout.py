"""Showcase arrangement matching the reference photo.

Each asset is modelled at the origin; the hero layout below places it on the
table. Positions/scales were solved by projecting each asset through the hero
camera and matching its silhouette bounding box to the reference photo
(see tools/fit_layout.py).
"""

import math

from mathutils import Euler, Vector

# Hero camera (85 mm on a 36 mm sensor, ~18 degrees above the table)
CAMERA = dict(target=(0.0, 0.35, 0.55), distance=10.2, elevation=18.0, azimuth=0.0, lens=85.0)

# key: (location, rotation_degrees(XYZ), uniform scale)
PLACEMENT = {
    "PetalGlow": ((-1.690, 1.948, 0.0), (0.0, 0.0, 0.0), 1.201),
    "MoonlightMist": ((-0.783, 2.327, 0.0), (0.0, 0.0, 0.0), 1.202),
    "FairyWings": ((0.017, 2.841, 0.0), (0.0, 0.0, 0.0), 1.166),
    "ForestBreath": ((0.868, 2.111, 0.0), (0.0, 0.0, 0.0), 1.117),
    "StardustElixir": ((1.753, 1.669, 0.0), (0.0, 0.0, 0.0), 1.090),
    "SpellBook": ((-1.133, -0.964, 0.0), (0.0, 0.0, 30.0), 0.865),
    "LoveBlooms": ((-0.227, -0.947, 0.0), (0.0, 0.0, 0.0), 0.893),
    "PixieDust": ((0.357, -0.640, 0.0), (0.0, 0.0, 0.0), 0.844),
    "DreamcapEssence": ((0.943, -0.870, 0.0), (0.0, 0.0, 0.0), 0.859),
    "TranquilTears": ((1.488, -1.260, 0.0), (0.0, 0.0, 0.0), 0.912),
    "GemBowl": ((-0.832, -1.924, 0.0), (0.0, 0.0, 0.0), 0.948),
    "StarWand": ((-0.120, -1.836, 0.0), (16.0, 0.0, -29.0), 0.842),
    "GoldScoop": ((0.505, -1.700, 0.0), (30.0, 0.0, -30.0), 0.784),
    "WingClip": ((1.078, -2.628, 0.0), (35.0, 10.0, 0.0), 0.705),
    "LooseGems/PinkHeart": ((-0.851, -2.594, 0.0), (55.0, 0.0, 0.0), 0.772),
    "LooseGems/PurpleStar": ((-0.535, -2.676, 0.0), (50.0, 0.0, 0.0), 0.781),
    "LooseGems/BlueStar": ((-0.382, -2.342, 0.0), (50.0, 0.0, 0.0), 0.811),
    "LooseGems/PurpleHeart": ((-0.113, -2.587, 0.0), (55.0, 0.0, 0.0), 0.752),
    "LooseGems/TealHeart": ((0.202, -2.684, 0.0), (55.0, 0.0, 0.0), 0.760),
}


def apply(obj, key):
    loc, rot, scale = PLACEMENT[key]
    obj.location = Vector(loc)
    obj.rotation_euler = Euler([math.radians(a) for a in rot], "XYZ")
    obj.scale = (scale, scale, scale)
