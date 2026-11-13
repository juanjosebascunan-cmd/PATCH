"""Desert canyon: heightfield, strata, rocks, cacti, distant mesas, FX meshes.
Layout: road runs along +z at x~2. West wall (x<0) is a climbable scree slope to a
rim at y~18 where the ranger stands; east wall is terraced cliff. Sun is low in the
south-west, so looking toward -z puts characters against the sun.
"""
import numpy as np
from geo import box, sphere, cylinder, tapered_box, transform, rot_x, rot_y, rot_z
from rig import merge, emissive

RIM_Y = 18.0

def _n(x, z):
    return (np.sin(x * 0.21 + z * 0.13) * 0.5 + np.sin(x * 0.47 - z * 0.31) * 0.25
            + np.sin(x * 1.1 + z * 0.9) * 0.12 + np.sin(z * 0.07 + 1.3) * 0.6)

def _terrace(h, s=3.2):
    f = h / s
    k = np.floor(f)
    fr = f - k
    return (k + fr ** 6) * s

def height(x, z):
    x = np.asarray(x, 'f8'); z = np.asarray(z, 'f8')
    floor = 0.12 * np.sin(x * 0.4) * np.sin(z * 0.3)
    # west scree slope -> rim plateau
    tw = np.clip((-x - 12.0) / 28.0, 0, 1)
    sw = tw * tw * (3 - 2 * tw)
    hw = RIM_Y * sw + _n(x, z) * 1.2 * tw * (1 - tw) * 4 + np.where(tw >= 1, 0.4 * _n(x * 0.5, z * 0.5), 0)
    # east terraced cliffs
    te = np.clip((x - 13.0) / 10.0, 0, 1)
    he = _terrace(28.0 * te ** 0.6) + np.where(te > 0, 0.6 * _n(x, z) * te, 0)
    # far plateau undulation
    far = np.clip((np.abs(x) - 60) / 40, 0, 1) * 4 * (1 + np.sin(z * 0.02))
    return floor + hw + he + far

STRATA = [(0.86, 0.47, 0.27), (0.74, 0.33, 0.22), (0.90, 0.60, 0.35), (0.66, 0.30, 0.24)]
FLOOR = (0.88, 0.58, 0.36)

def terrain_mesh(x0=-110, x1=110, z0=-240, z1=170, dx=1.6, dz=2.0):
    xs = np.arange(x0, x1 + 1e-6, dx); zs = np.arange(z0, z1 + 1e-6, dz)
    # finer sampling near the action
    X, Z = np.meshgrid(xs, zs)
    Y = height(X, Z)
    P = np.stack([X, Y, Z], -1)
    a = P[:-1, :-1]; b = P[:-1, 1:]; c = P[1:, 1:]; d = P[1:, :-1]
    # alternate diagonal for less regular facets
    tris = np.concatenate([np.stack([a, d, c], -2), np.stack([a, c, b], -2)], 0).reshape(-1, 3, 3)
    cen = tris.mean(1)
    nrm = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
    slope = 1 - np.abs(nrm[:, 1])
    band = (np.floor(cen[:, 1] / 2.6 + 0.3 * np.sin(cen[:, 2] * 0.05))).astype(int) % len(STRATA)
    col = np.array(STRATA, 'f4')[band]
    flat = (slope < 0.12)[:, None]
    low = (cen[:, 1] < 0.6)[:, None]
    col = np.where(flat & ~low, col * 0.92 + np.array([0.08, 0.06, 0.03]), col)
    col = np.where(low, np.array(FLOOR, 'f4'), col)
    # sparse darker scrub specks on floor
    rng = np.random.default_rng(3)
    speck = (rng.random(len(col)) < 0.03)[:, None] & low
    col = np.where(speck, np.array([0.62, 0.40, 0.26]), col)
    v = tris.reshape(-1, 3).astype('f4')
    c = np.repeat(col, 3, 0).astype('f4')
    flag = np.repeat(np.where(low[:, 0], 0.0, 2.0), 3)[:, None].astype('f4')
    return {'v': v, 'c': c, 'flag': flag}

def road_mesh(z0=-240, z1=170):
    """Two darker tire ruts along x~0.9 and x~3.1."""
    parts = []
    for xr in (1.0, 3.0):
        zs = np.arange(z0, z1, 4.0)
        for z in zs:
            w = 0.5
            p = [(xr - w / 2, 0, z), (xr + w / 2, 0, z), (xr + w / 2, 0, z + 4), (xr - w / 2, 0, z + 4)]
            p = [(px, height(px, pz) + 0.03, pz) for px, py, pz in p]
            parts.append({'v': np.array([p[0], p[2], p[1], p[0], p[3], p[2]], 'f4'),
                          'c': np.tile(np.array([0.74, 0.46, 0.30], 'f4'), (6, 1))})
    return merge(parts)

def rock(r, seed, color=(0.70, 0.36, 0.24)):
    rng = np.random.default_rng(seed)
    s = sphere(r, color, n=5, squash=(1.0, 0.7, 1.1))
    v = s['v'].copy()
    key = np.round(v * 100).astype(int)
    _, inv = np.unique(key, axis=0, return_inverse=True)
    jit = rng.uniform(0.75, 1.2, inv.max() + 1)
    v *= jit[inv.ravel()][:, None]
    v[:, 1] = np.maximum(v[:, 1], -0.15 * r)
    return {'v': v.astype('f4'), 'c': s['c']}

def cactus(h, seed):
    g = (0.36, 0.48, 0.30)
    rng = np.random.default_rng(seed)
    parts = [cylinder(0.22, h, g, n=7)]
    for side in (1, -1):
        if rng.random() < 0.8:
            ah = rng.uniform(0.35, 0.6) * h
            parts.append(transform(cylinder(0.15, 0.5, g, n=6), rot_z(np.pi / 2), (0.0, ah, 0)))
            parts.append(transform(cylinder(0.15, rng.uniform(0.6, 1.1), g, n=6), None, (0.5 * side, ah, 0)))
    return merge(parts)

def dead_tree():
    b = (0.30, 0.20, 0.18)
    parts = [cylinder(0.14, 2.4, b, n=6, r_top=0.08)]
    for a, h, l in ((0.7, 1.6, 1.2), (-0.9, 2.0, 0.9), (0.3, 2.3, 0.8), (-0.4, 1.2, 0.7)):
        parts.append(transform(cylinder(0.06, l, b, n=5, r_top=0.03), rot_z(a), (0, h, 0)))
    return merge(parts)

BOULDER_POS = (10.5, 0.0, 6.5)   # truck 1 crashes into this

CLEAR = [(6.6, -104.0, 9.0), (5.0, -104.0, 9.0), (-9.5, 17.0, 6.0), (-4.0, 14.0, 6.0)]

def _clear(x, z):
    return all((x - cx) ** 2 + (z - cz) ** 2 > r * r for cx, cz, r in CLEAR)

def static_world():
    parts = [terrain_mesh(), road_mesh()]
    rng = np.random.default_rng(11)
    # scattered rocks
    for i in range(140):
        x = rng.uniform(-90, 90); z = rng.uniform(-220, 150)
        if abs(x - 2) < 5 or not _clear(x, z):
            continue
        r = rng.uniform(0.4, 2.2)
        parts.append(transform(rock(r, i), rot_y(rng.uniform(0, 6)), (x, height(x, z) - 0.2 * r, z)))
    # the crash boulder + friends
    bx, by, bz = BOULDER_POS
    parts.append(transform(rock(2.6, 999, (0.66, 0.32, 0.22)), rot_y(0.4), (bx + 1.2, height(bx, bz) - 0.3, bz)))
    parts.append(transform(rock(1.3, 998), rot_y(1.4), (bx + 2.5, height(bx, bz), bz - 3)))
    # cacti on floor & lower slopes
    for i in range(45):
        x = rng.uniform(-30, 20); z = rng.uniform(-200, 120)
        if abs(x - 2) < 4.5 or abs(z - 5) < 10 and abs(x) < 14 or not _clear(x, z) or (abs(z + 5) < 22 and -16 < x < 14):
            continue
        parts.append(transform(cactus(rng.uniform(2.0, 4.0), i), rot_y(rng.uniform(0, 6)), (x, height(x, z) - 0.1, z)))
    # rim props near the ranger lookout
    parts.append(transform(dead_tree(), rot_y(0.5), (-45.5, height(-45.5, 1.0) - 0.1, 1.0)))
    parts.append(transform(rock(1.0, 501), None, (-40.5, height(-40.5, 7.5) - 0.3, 7.5)))
    # distant mesas / buttes (violet by haze)
    for i in range(16):
        ang = rng.uniform(-np.pi, np.pi)
        dist = rng.uniform(260, 520)
        x = np.cos(ang) * dist; z = np.sin(ang) * dist
        h = rng.uniform(40, 120); w = rng.uniform(40, 140)
        col = STRATA[i % 4]
        m = tapered_box(w, w * 0.8, w * 0.7, w * 0.55, h, col, cy=-2)
        m2 = tapered_box(w * 1.3, w * 1.0, w, w * 0.85, h * 0.35, STRATA[(i + 1) % 4], cy=-2)
        parts.append(transform(merge([m, m2]), rot_y(rng.uniform(0, 3)), (x, 0, z)))
    return merge(parts)


# ------------------------------------------------------------------ FX ------
DUST = (0.96, 0.72, 0.52)

def puff(center, r, color=DUST, seed=0, dust=True):
    s = sphere(r, color, n=6, squash=(1.0, 0.82, 1.0)) if dust else rock(r, seed, color)
    m = transform(s, rot_y(seed * 1.7), center)
    if dust:
        m['flag'] = np.full((len(m['v']), 1), 3.0, 'f4')
    return m

def dust_cloud(origin, age, n=6, spread=1.0, rise=0.6, size=0.6, seed=0, life=2.0, color=DUST):
    """Expanding puffs; age in seconds since emission."""
    if age < 0 or age > life:
        return []
    rng = np.random.default_rng(seed)
    out = []
    k = age / life
    for i in range(n):
        d = rng.normal(size=3); d[1] = abs(d[1]) * 0.5 + 0.2; d /= np.linalg.norm(d)
        p = np.asarray(origin) + d * spread * (1 - (1 - k) ** 2) * rng.uniform(0.6, 1.4) + np.array([0, rise * k, 0])
        r = size * rng.uniform(0.45, 1.5) * (0.4 + 1.2 * k ** 0.5) * (1 - k ** 3)
        p = p + rng.normal(size=3) * size * np.array([0.5, 0.25, 0.5])
        if r > 0.02:
            out.append(puff(p, r, color, seed * 31 + i))
    return out

def flash_star(center, size, toward, color=(1.0, 0.95, 0.65), spikes=7, seed=0):
    """Emissive spiky star billboard facing `toward` (camera position)."""
    c = np.asarray(center, 'f4'); f = np.asarray(toward, 'f4') - c; f /= np.linalg.norm(f)
    r_ = np.cross(f, [0, 1, 0]); r_ /= max(np.linalg.norm(r_), 1e-6); u = np.cross(r_, f)
    rng = np.random.default_rng(seed)
    tris = []
    for i in range(spikes):
        a = 2 * np.pi * i / spikes + rng.uniform(-0.2, 0.2)
        L = size * rng.uniform(0.6, 1.3)
        tip = c + (np.cos(a) * r_ + np.sin(a) * u) * L
        w = 0.35 * size
        b1 = c + (np.cos(a + 0.5) * r_ + np.sin(a + 0.5) * u) * w * 0.5
        b2 = c + (np.cos(a - 0.5) * r_ + np.sin(a - 0.5) * u) * w * 0.5
        tris += [c, b2, tip, c, tip, b1]
    v = np.array(tris, 'f4')
    m = {'v': v, 'c': np.tile(np.asarray(color, 'f4'), (len(v), 1))}
    core = flash_core(center, size * 0.35, f, r_, u)
    return [emissive(m), emissive(core)]

def flash_core(c, r, f, r_, u, color=(1, 1, 0.92)):
    tris = []
    n = 10
    for i in range(n):
        a0 = 2 * np.pi * i / n; a1 = 2 * np.pi * (i + 1) / n
        tris += [c, c + (np.cos(a1) * r_ + np.sin(a1) * u) * r, c + (np.cos(a0) * r_ + np.sin(a0) * u) * r]
    v = np.array(tris, 'f4') + f * 0.02
    return {'v': v, 'c': np.tile(np.asarray(color, 'f4'), (len(v), 1))}

def explosion(origin, age, seed=0, scale=1.0):
    """Fireball (emissive, banded) -> dark smoke (shaded)."""
    out = []
    if age < 0 or age > 6:
        return out
    rng = np.random.default_rng(seed)
    o = np.asarray(origin, 'f4')
    for i in range(16):
        d = rng.normal(size=3); d[1] = abs(d[1]) + 0.3; d /= np.linalg.norm(d)
        sp = rng.uniform(2.5, 6.0) * scale
        tt = min(age, 1.2)
        p = o + d * sp * (1 - np.exp(-tt * 3.0)) + np.array([0, 0.9 * age * scale, 0])
        if age < 1.1:
            r = scale * rng.uniform(0.7, 1.5) * (0.3 + age * 1.8) * (1 - age / 1.3)
            k = age / 1.1
            col = [(1.0, 0.96, 0.70), (1.0, 0.72, 0.18), (0.98, 0.42, 0.12), (0.85, 0.20, 0.12)][min(3, int(k * 3.2 + (i % 2)))]
            if r > 0.05:
                out.append(emissive(puff(p, r, col, seed * 100 + i)))
        r2 = scale * rng.uniform(0.8, 1.6) * min(1.0, age * 1.4) * (1 - max(0, age - 4.5) / 1.5)
        if age > 0.35 and r2 > 0.05:
            sm = puff(p + np.array([0, 0.5 * age, 0]), r2, (0.34, 0.22, 0.32), seed * 200 + i, dust=True)
            out.append(sm)
    # flying debris chunks
    for i in range(10):
        d = rng.normal(size=3); d[1] = abs(d[1]) * 1.5 + 0.5
        v0 = d * rng.uniform(4, 9) * scale
        p = o + v0 * age + np.array([0, -4.9 * age * age, 0])
        if p[1] > 0:
            out.append(transform(box(0.25, 0.12, 0.3, (0.15, 0.12, 0.14)), rot_x(age * 9 + i), p))
    return out

def sparks(origin, age, toward, n=10, seed=0, life=0.45):
    if age < 0 or age > life:
        return []
    rng = np.random.default_rng(seed)
    out = []
    o = np.asarray(origin, 'f4')
    for i in range(n):
        d = rng.normal(size=3); d /= np.linalg.norm(d)
        p0 = o + d * age * rng.uniform(5, 12)
        p1 = p0 + d * 0.25
        w = 0.03
        side = np.cross(d, [0, 1, 0]); side = side / max(np.linalg.norm(side), 1e-6) * w
        v = np.array([p0 - side, p1, p0 + side], 'f4')
        out.append(emissive({'v': v, 'c': np.tile(np.array([1.0, 0.85, 0.4], 'f4'), (3, 1))}))
        out.append(emissive({'v': v[::-1].copy(), 'c': np.tile(np.array([1.0, 0.85, 0.4], 'f4'), (3, 1))}))
    return out
