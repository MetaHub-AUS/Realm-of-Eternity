"""2D outlines (closed, CCW, as (N, 2) numpy arrays) used for reliefs, gems and frames."""

import math

import numpy as np

from .core import catmull_rom, ccw, resample

TAU = math.tau


def chaikin(P, iterations=3, closed=True):
    P = np.asarray(P, dtype=np.float64)
    for _ in range(iterations):
        Q = np.roll(P, -1, axis=0) if closed else P[1:]
        A = P if closed else P[:-1]
        new = np.empty((len(A) * 2, 2))
        new[0::2] = 0.75 * A + 0.25 * Q
        new[1::2] = 0.25 * A + 0.75 * Q
        if not closed:
            new = np.vstack([P[:1], new, P[-1:]])
        P = new
    return P


def circle(r, n=48, center=(0.0, 0.0)):
    t = np.linspace(0, TAU, n, endpoint=False)
    return np.stack([center[0] + r * np.cos(t), center[1] + r * np.sin(t)], axis=1)


def ellipse(rx, ry, n=48, center=(0.0, 0.0), angle=0.0):
    t = np.linspace(0, TAU, n, endpoint=False)
    P = np.stack([rx * np.cos(t), ry * np.sin(t)], axis=1)
    c, s = math.cos(angle), math.sin(angle)
    P = P @ np.array([[c, s], [-s, c]])
    return P + np.asarray(center)


def superellipse2d(rx, ry, exponent=4.0, n=96, center=(0.0, 0.0)):
    t = np.linspace(0, TAU, n, endpoint=False)
    c, s = np.cos(t), np.sin(t)
    e = 2.0 / exponent
    return np.stack([center[0] + rx * np.sign(c) * np.abs(c) ** e,
                     center[1] + ry * np.sign(s) * np.abs(s) ** e], axis=1)


def heart(width, height=None, n=96, center=(0.0, 0.0), plump=0.0, dip=1.0):
    """Cute heart centred on its bounding box. plump > 0 rounds the lower point."""
    height = width * 0.9 if height is None else height
    t = np.linspace(0, TAU, n, endpoint=False)
    x = 16 * np.sin(t) ** 3
    y = 13 * np.cos(t) - 5 * dip * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t)
    if plump:
        y = y + plump * 4.0 * (1 - np.cos(t)) ** 2 * 0.25 * (y < -5)
    P = np.stack([x, y], axis=1)
    mn, mx = P.min(axis=0), P.max(axis=0)
    P = (P - (mn + mx) / 2) / (mx - mn) * np.array([width, height])
    return ccw(P + np.asarray(center))


def star(r_out, r_in=None, points=5, rounding=0.0, n_round=3, center=(0.0, 0.0), rotation=0.0):
    r_in = r_out * 0.48 if r_in is None else r_in
    P = []
    for i in range(points * 2):
        a = math.pi / 2 + rotation + math.pi * i / points
        r = r_out if i % 2 == 0 else r_in
        P.append((r * math.cos(a), r * math.sin(a)))
    P = np.array(P)
    if rounding > 0:
        # blend towards chaikin-smoothed version for softly rounded tips
        dense = resample(P, count=points * 40)
        sm = chaikin(P, n_round)
        sm = resample(sm, count=points * 40)
        P = dense * (1 - rounding) + sm * rounding
    return ccw(P + np.asarray(center))


def flower(petals=5, r_out=1.0, r_in=0.45, sharp=0.55, n=180, rotation=0.0, center=(0.0, 0.0)):
    t = np.linspace(0, TAU, n, endpoint=False)
    k = np.abs(np.cos(petals * (t - rotation) / 2.0)) ** sharp
    r = r_in + (r_out - r_in) * k
    return np.stack([center[0] + r * np.cos(t), center[1] + r * np.sin(t)], axis=1)


def petal(length, width, n=40, tip=0.0):
    """Petal from the origin pointing along +Y (rounded tip, narrow base)."""
    t = np.linspace(0, TAU, n, endpoint=False)
    y = (np.sin(t - math.pi / 2) + 1) * 0.5 * length  # 0..length
    w = width * 0.5 * np.sin(np.clip(y / length, 0, 1) * math.pi) ** 0.55
    w = w * (1 + tip * (y / length))
    x = np.where(np.cos(t - math.pi / 2) >= 0, 1, -1) * w
    P = np.stack([x, y], axis=1)
    return ccw(P)


def leaf(length, width, n=48, bend=0.0):
    """Pointed leaf from origin along +Y; `bend` curves it sideways."""
    s = np.linspace(0, 1, n // 2)
    w = width * 0.5 * np.sin(s * math.pi) ** 0.9 * (1 - 0.25 * s)
    y = s * length
    off = bend * length * s * s
    right = np.stack([off + w, y], axis=1)
    left = np.stack([off - w, y], axis=1)[::-1][1:-1]
    return ccw(np.vstack([right, left]))


def crescent(r, thickness=0.42, offset_angle=0.0, n=96, tip_round=0.2):
    """Crescent moon opening to the right; r = outer radius."""
    d = r * thickness * 1.25
    r2 = r * 0.86
    c2 = np.array([d, r * 0.08])
    # intersections of |p| = r and |p - c2| = r2
    dd = np.linalg.norm(c2)
    a = (r * r - r2 * r2 + dd * dd) / (2 * dd)
    h = math.sqrt(max(r * r - a * a, 1e-9))
    base = c2 * a / dd
    perp = np.array([-c2[1], c2[0]]) / dd
    i1, i2 = base + perp * h, base - perp * h
    a1 = math.atan2(i1[1], i1[0])
    a2 = math.atan2(i2[1], i2[0])
    if a2 < a1:
        a2 += TAU
    outer = np.array([[r * math.cos(t), r * math.sin(t)] for t in np.linspace(a1, a2, n)])
    b1 = math.atan2(*(i2 - c2)[::-1])
    b2 = math.atan2(*(i1 - c2)[::-1])
    if b2 > b1:
        b2 -= TAU
    inner = np.array([c2 + r2 * np.array([math.cos(t), math.sin(t)]) for t in np.linspace(b1, b2, n)])
    P = np.vstack([outer, inner[1:-1]])
    P = chaikin(P, 2) if tip_round else P
    c, s = math.cos(offset_angle), math.sin(offset_angle)
    P = P @ np.array([[c, s], [-s, c]])
    return ccw(P)


def cartouche(w, h, peak=0.06, notch=0.05, side_notch=0.025, bottom_peak=None, exponent=6.0,
              n=480, top_arch=0.0):
    """Decorative label frame: rounded rectangle with a centre peak, scalloped corners
    and small side notches (the 'ornate potion label' shape)."""
    W, H = w / 2.0, h / 2.0
    bottom_peak = peak if bottom_peak is None else bottom_peak
    t = np.linspace(0, TAU, n, endpoint=False)
    c, s = np.cos(t), np.sin(t)
    e = 2.0 / exponent
    x = W * np.sign(c) * np.abs(c) ** e
    y = H * np.sign(s) * np.abs(s) ** e
    P = np.stack([x, y], axis=1)
    T = np.roll(P, -1, axis=0) - np.roll(P, 1, axis=0)
    T /= np.linalg.norm(T, axis=1, keepdims=True)
    N = np.stack([T[:, 1], -T[:, 0]], axis=1)
    disp = np.zeros(n)
    top = y > 0
    xs = x / W
    # centre peaks (ogee point)
    disp += np.where(top, peak * h * np.exp(-(xs / 0.13) ** 2), bottom_peak * h * np.exp(-(xs / 0.13) ** 2))
    # shallow shoulder bumps either side of the peak
    disp += np.where(top, 0.35 * peak * h * np.exp(-((np.abs(xs) - 0.42) / 0.1) ** 2), 0.0)
    disp += top_arch * h * np.where(top, np.cos(np.clip(xs, -1, 1) * math.pi / 2), 0.0)
    # scalloped corners
    ang = np.arctan2(y / H, x / W)
    for ca in (math.pi / 4, 3 * math.pi / 4, -math.pi / 4, -3 * math.pi / 4):
        da = np.angle(np.exp(1j * (ang - ca)))
        disp -= notch * min(w, h) * np.exp(-(da / 0.16) ** 2)
    # side notches
    side = np.abs(xs) > 0.9
    disp -= np.where(side, side_notch * w * np.exp(-((y / H) / 0.1) ** 2), 0.0)
    P = P + N * disp[:, None]
    return ccw(P)


def wing_lobe(length, width, n=64, angle=0.0, droop=0.0):
    """Teardrop fairy-wing lobe from the origin, pointing along `angle` (radians from +X)."""
    s = np.linspace(0, 1, n // 2)
    w = width * 0.5 * np.sin(np.clip(s, 0, 1) * math.pi) ** 0.7 * (0.35 + 0.65 * s ** 0.6)
    w = w * np.where(s > 0.75, np.sqrt(np.maximum(0, 1 - ((s - 0.75) / 0.25) ** 2)) * 0.6 + 0.4 * (1 - (s - 0.75) / 0.25), 1)
    xx = s * length
    yy = droop * length * s * s
    top = np.stack([xx, yy + w], axis=1)
    bot = np.stack([xx, yy - w], axis=1)[::-1]
    P = np.vstack([top, bot[1:-1]])
    P = chaikin(P, 2)
    c, sn = math.cos(angle), math.sin(angle)
    return ccw(P @ np.array([[c, sn], [-sn, c]]))


def round_lobe(length, width, angle=0.0, n=72, skew=0.0, base=0.18):
    """Rounded butterfly lobe: narrow at the origin, broad round end (teardrop)."""
    R = width / 2
    cx = length - R
    t = np.linspace(-math.pi * 0.62, math.pi * 0.62, n)
    arc = np.stack([cx + R * np.cos(t), R * np.sin(t) * (1 + skew * np.sin(t))], axis=1)
    b = width * base * 0.5
    P = np.vstack([arc, [[length * 0.12, b], [0.0, 0.0], [length * 0.12, -b]]])
    P = chaikin(P, 3)
    c, sn = math.cos(angle), math.sin(angle)
    return ccw(P @ np.array([[c, sn], [-sn, c]]))


def butterfly_wing(upper_len, upper_w, lower_len, lower_w, spread=0.55, lower_angle=-0.75):
    """Outline of one butterfly half (right side) as the union hull of two lobes."""
    up = wing_lobe(upper_len, upper_w, angle=spread)
    lo = wing_lobe(lower_len, lower_w, angle=lower_angle)
    return up, lo


def rounded_rect(w, h, r, n_corner=8, center=(0.0, 0.0)):
    W, H = w / 2 - r, h / 2 - r
    pts = []
    for cx, cy, a0 in ((W, H, 0), (-W, H, 90), (-W, -H, 180), (W, -H, 270)):
        for i in range(n_corner + 1):
            a = math.radians(a0 + 90 * i / n_corner)
            pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    return ccw(np.array(pts) + np.asarray(center))


def smooth_closed(points, samples=10):
    return ccw(catmull_rom(np.asarray(points, dtype=np.float64), samples, closed=True))


def scroll(start, heading, length, curl_r, turns=1.25, handed=1, n=60):
    """Open 2D path: a gentle stem that ends in a spiral curl (filigree / vine tendril)."""
    pts = []
    x, y = start
    hd = heading
    stem_n = n // 2
    for i in range(stem_n):
        pts.append((x, y))
        x += math.cos(hd) * length / stem_n
        y += math.sin(hd) * length / stem_n
        hd += handed * 0.25 / stem_n
    # spiral: radius shrinking
    cx = x - handed * math.sin(hd) * -curl_r
    cy = y + handed * math.cos(hd) * curl_r
    a0 = math.atan2(y - cy, x - cx)
    for i in range(n - stem_n):
        f = i / (n - stem_n - 1)
        a = a0 + handed * f * turns * TAU
        rr = curl_r * (1 - 0.72 * f)
        pts.append((cx + rr * math.cos(a), cy + rr * math.sin(a)))
    return np.array(pts)
