"""Skeletons + rigid-skinned toon characters and props.
Bones: local offset from parent, pose = Euler (rx, ry, rz) applied as Ry@Rx@Rz.
Conventions (character faces +z, left = +x):
  limbs hang along -y; rx < 0 swings a limb FORWARD, rx > 0 backward
  spine/neck/head rx > 0 leans forward; dog body rx > 0 pitches nose down
"""
import numpy as np
from geo import box, sphere, cylinder, tapered_box, rot_x, rot_y, rot_z, transform, _soup

def emissive(part):
    part['flag'] = np.ones((len(part['v']), 1), 'f4')
    return part

def merge(parts):
    v = np.concatenate([p['v'] for p in parts]).astype('f4')
    c = np.concatenate([p['c'] for p in parts]).astype('f4')
    f = np.concatenate([p.get('flag', np.zeros((len(p['v']), 1), 'f4')) for p in parts]).astype('f4')
    return {'v': v, 'c': c, 'flag': f}

def limb(w0, d0, w1, d1, L, color, cx=0.0, cz=0.0):
    """segment hanging from origin down to -L, top section (w0,d0) -> bottom (w1,d1)."""
    m = tapered_box(w1, d1, w0, d0, L, color, cy=-L)
    m['v'] = m['v'] + np.array([cx, 0, cz], 'f4')
    return m

def euler(r):
    rx, ry, rz = r
    return rot_y(ry) @ rot_x(rx) @ rot_z(rz)


class Rig:
    def __init__(self):
        self.bones = {}   # name -> (parent, offset)
        self.order = []
        self.meshes = {}  # name -> list of (tag, mesh)

    def bone(self, name, parent, offset):
        self.bones[name] = (parent, np.asarray(offset, 'f4'))
        self.order.append(name)
        self.meshes[name] = []

    def attach(self, name, mesh, tag=None):
        self.meshes[name].append((tag, mesh))

    def world(self, pose, root_pos=(0, 0, 0), yaw=0.0, scale=1.0, root_off=(0, 0, 0)):
        Ms = {}
        for b in self.order:
            parent, off = self.bones[b]
            r = pose.get(b, (0, 0, 0))
            L = np.eye(4, dtype='f4')
            L[:3, :3] = euler(r)
            L[:3, 3] = off
            if parent is None:
                L[:3, 3] = off + np.asarray(root_off, 'f4')
                Rt = np.eye(4, dtype='f4')
                Rt[:3, :3] = rot_y(yaw) * scale
                Rt[:3, 3] = root_pos
                Ms[b] = Rt @ L
            else:
                Ms[b] = Ms[parent] @ L
        return Ms

    def build(self, pose, root_pos=(0, 0, 0), yaw=0.0, scale=1.0, hide=(), root_off=(0, 0, 0), extra=None):
        Ms = self.world(pose, root_pos, yaw, scale, root_off)
        out = []
        for b in self.order:
            M = Ms[b]
            for tag, m in self.meshes[b]:
                if tag in hide:
                    continue
                v = m['v'] @ M[:3, :3].T + M[:3, 3]
                out.append({'v': v.astype('f4'), 'c': m['c'], 'flag': m.get('flag', np.zeros((len(v), 1), 'f4'))})
        if extra:
            out += extra
        return merge(out), Ms


# ---------------------------------------------------------------- humanoid --
def humanoid(kind='ranger'):
    R = Rig()
    R.bone('hips', None, (0, 1.0, 0))
    R.bone('spine', 'hips', (0, 0.12, 0))
    R.bone('chest', 'spine', (0, 0.25, 0))
    R.bone('neck', 'chest', (0, 0.30, 0))
    R.bone('head', 'neck', (0, 0.08, 0))
    for s, sx in (('L', 1), ('R', -1)):
        R.bone('uarm' + s, 'chest', (0.24 * sx, 0.24, 0))
        R.bone('farm' + s, 'uarm' + s, (0, -0.30, 0))
        R.bone('hand' + s, 'farm' + s, (0, -0.27, 0))
        R.bone('thigh' + s, 'hips', (0.11 * sx, -0.04, 0))
        R.bone('shin' + s, 'thigh' + s, (0, -0.46, 0))
        R.bone('foot' + s, 'shin' + s, (0, -0.44, 0))

    if kind == 'ranger':
        skin = (0.80, 0.52, 0.38); coat = (0.46, 0.30, 0.20); shirt = (0.56, 0.46, 0.34)
        pants = (0.26, 0.24, 0.30); boots = (0.20, 0.13, 0.10); hat = (0.38, 0.27, 0.19)
        scarf = (0.72, 0.18, 0.14); beard = (0.62, 0.60, 0.58)
    else:
        boss = kind == 'boss'
        skin = (0.70, 0.45, 0.33); coat = (0.24, 0.26, 0.20) if not boss else (0.32, 0.12, 0.12)
        shirt = (0.30, 0.30, 0.24); pants = (0.20, 0.20, 0.18); boots = (0.12, 0.10, 0.09)
        hat = (0.18, 0.17, 0.15); scarf = (0.55, 0.12, 0.10); beard = None

    A = R.attach
    A('hips', box(0.36, 0.22, 0.22, pants, cy=0.0))
    A('hips', box(0.38, 0.06, 0.24, boots, cy=0.09))           # belt
    A('hips', box(0.08, 0.06, 0.05, (0.85, 0.70, 0.30), cz=0.12, cy=0.09))  # buckle
    A('spine', box(0.34, 0.26, 0.21, shirt, cy=0.12))
    A('chest', tapered_box(0.34, 0.22, 0.54, 0.27, 0.32, coat, cy=-0.02))
    A('chest', box(0.20, 0.07, 0.20, coat, cx=0.22, cy=0.27))
    A('chest', box(0.20, 0.07, 0.20, coat, cx=-0.22, cy=0.27))
    A('chest', box(0.16, 0.30, 0.02, shirt, cz=0.125, cy=0.14))   # shirt opening
    A('neck', cylinder(0.065, 0.12, skin, n=6, cy=-0.02))
    A('neck', cylinder(0.11, 0.09, scarf, n=7, cy=-0.02, r_top=0.09))

    # head
    A('head', tapered_box(0.15, 0.17, 0.22, 0.24, 0.20, skin, cy=0.0))
    A('head', box(0.22, 0.08, 0.24, skin, cy=0.23))
    A('head', box(0.06, 0.07, 0.06, skin, cy=0.12, cz=0.14))       # nose
    if kind == 'ranger':
        A('head', box(0.23, 0.10, 0.10, beard, cy=0.04, cz=0.08))   # beard
        A('head', box(0.06, 0.03, 0.02, (0.08, 0.05, 0.06), cx=0.055, cy=0.17, cz=0.122))  # eyes
        A('head', box(0.06, 0.03, 0.02, (0.08, 0.05, 0.06), cx=-0.055, cy=0.17, cz=0.122))
        A('head', box(0.07, 0.02, 0.02, beard, cx=0.055, cy=0.205, cz=0.122))  # brows
        A('head', box(0.07, 0.02, 0.02, beard, cx=-0.055, cy=0.205, cz=0.122))
        # wide-brim hat
        A('head', cylinder(0.30, 0.025, hat, n=12, cy=0.245))
        A('head', cylinder(0.145, 0.15, hat, n=8, cy=0.26, r_top=0.12))
        A('head', cylinder(0.148, 0.03, (0.22, 0.15, 0.10), n=8, cy=0.27))
    else:
        mask = (0.90, 0.86, 0.78) if kind != 'boss' else (0.80, 0.20, 0.16)
        A('head', box(0.25, 0.28, 0.06, mask, cy=0.13, cz=0.11))         # mask plate
        A('head', box(0.07, 0.05, 0.02, (0.05, 0.03, 0.04), cx=0.055, cy=0.17, cz=0.145))
        A('head', box(0.07, 0.05, 0.02, (0.05, 0.03, 0.04), cx=-0.055, cy=0.17, cz=0.145))
        A('head', emissive(box(0.03, 0.02, 0.01, (1.0, 0.25, 0.15), cx=0.055, cy=0.17, cz=0.157)))
        A('head', emissive(box(0.03, 0.02, 0.01, (1.0, 0.25, 0.15), cx=-0.055, cy=0.17, cz=0.157)))
        A('head', box(0.16, 0.02, 0.02, (0.05, 0.03, 0.04), cy=0.07, cz=0.145))   # teeth slit
        for k in range(4):
            A('head', box(0.012, 0.05, 0.02, mask, cx=-0.06 + k * 0.04, cy=0.07, cz=0.146))
        A('head', tapered_box(0.30, 0.30, 0.18, 0.20, 0.22, hat, cy=0.12))     # hood
        if kind == 'boss':
            A('head', transform(tapered_box(0.06, 0.06, 0.01, 0.01, 0.24, (0.88, 0.82, 0.70)), rot_z(-0.6), (0.12, 0.26, 0)))
            A('head', transform(tapered_box(0.06, 0.06, 0.01, 0.01, 0.24, (0.88, 0.82, 0.70)), rot_z(0.6), (-0.12, 0.26, 0)))

    for s, sx in (('L', 1), ('R', -1)):
        A('uarm' + s, limb(0.16, 0.15, 0.11, 0.11, 0.31, coat))
        A('farm' + s, limb(0.12, 0.12, 0.09, 0.09, 0.27, coat if kind == 'ranger' else shirt))
        A('farm' + s, box(0.13, 0.06, 0.13, boots, cy=-0.25))   # glove cuff
        A('hand' + s, box(0.10, 0.11, 0.09, boots if kind != 'ranger' else skin, cy=-0.05))
        A('thigh' + s, limb(0.19, 0.19, 0.13, 0.14, 0.47, pants))
        A('shin' + s, limb(0.13, 0.14, 0.10, 0.11, 0.30, pants))
        A('shin' + s, box(0.15, 0.20, 0.15, boots, cy=-0.34))
        A('foot' + s, box(0.13, 0.08, 0.27, boots, cy=-0.02, cz=0.06))
        if kind == 'ranger':
            # duster coat flaps ride on thighs
            A('thigh' + s, box(0.22, 0.62, 0.05, coat, cx=0.04 * sx, cy=-0.24, cz=-0.10))
            A('thigh' + s, box(0.05, 0.60, 0.20, coat, cx=0.12 * sx, cy=-0.24, cz=-0.01))
        if kind == 'boss':
            A('thigh' + s, box(0.20, 0.40, 0.05, coat, cx=0.03 * sx, cy=-0.18, cz=-0.10))

    # weapons, tagged so they can be hidden
    metal = (0.30, 0.30, 0.36)
    if kind == 'ranger':
        gun = merge([box(0.05, 0.12, 0.08, (0.35, 0.20, 0.14), cy=-0.06),
                     box(0.045, 0.26, 0.06, metal, cy=-0.20, cz=0.03),
                     box(0.06, 0.08, 0.07, metal, cy=-0.12, cz=0.03)])
        A('handR', gun, tag='gun')
        A('hips', box(0.08, 0.16, 0.10, (0.30, 0.19, 0.13), cx=-0.20, cy=-0.03), tag='holster')
    else:
        wood = (0.42, 0.24, 0.14)
        rifle = merge([box(0.05, 0.10, 0.07, wood, cy=-0.05),
                       box(0.05, 0.36, 0.08, wood, cy=0.08, cz=-0.03),     # stock (behind hand)
                       box(0.045, 0.62, 0.05, metal, cy=-0.38, cz=0.02),    # barrel
                       box(0.05, 0.14, 0.07, metal, cy=-0.12, cz=0.04)])
        A('handR', rifle, tag='gun')
    return R


# ---------------------------------------------------------------- dog -------
def dog(pup=False):
    R = Rig()
    s = 1.0
    if pup:
        fur = (0.86, 0.66, 0.42); fur2 = (0.98, 0.88, 0.70); dark = (0.40, 0.25, 0.18)
    else:
        fur = (0.48, 0.42, 0.40); fur2 = (0.80, 0.78, 0.74); dark = (0.22, 0.18, 0.20)
    leg = 0.24
    R.bone('body', None, (0, 0.55, 0))
    R.bone('chest', 'body', (0, 0.02, 0.22))
    R.bone('neck', 'chest', (0, 0.10, 0.16))
    R.bone('head', 'neck', (0, 0.16, 0.06))
    R.bone('jaw', 'head', (0, -0.05, 0.12))
    R.bone('earL', 'head', (0.08, 0.10, -0.02))
    R.bone('earR', 'head', (-0.08, 0.10, -0.02))
    R.bone('tail1', 'body', (0, 0.10, -0.32))
    R.bone('tail2', 'tail1', (0, 0, -0.18))
    for nm, par, off in (('fL', 'chest', (0.11, -0.05, 0.06)), ('fR', 'chest', (-0.11, -0.05, 0.06)),
                         ('bL', 'body', (0.11, -0.02, -0.26)), ('bR', 'body', (-0.11, -0.02, -0.26))):
        R.bone(nm + '1', par, off)
        R.bone(nm + '2', nm + '1', (0, -leg, 0))
        R.bone(nm + '3', nm + '2', (0, -leg + 0.02, 0))
    A = R.attach
    A('body', box(0.30, 0.28, 0.50, fur, cz=-0.08))
    A('body', box(0.26, 0.10, 0.44, fur2, cy=-0.13, cz=-0.06))     # belly
    A('chest', box(0.32, 0.34, 0.26, fur, cy=0.02))
    A('chest', box(0.22, 0.24, 0.06, fur2, cy=-0.02, cz=0.12))     # chest bib
    A('neck', transform(box(0.18, 0.22, 0.18, fur), rot_x(0.5), (0, 0.04, 0.02)))
    hs = 1.0
    A('head', box(0.24 * hs, 0.20 * hs, 0.22 * hs, fur, cy=0.03))
    A('head', box(0.14, 0.11, 0.17, fur2, cy=-0.02, cz=0.16))      # muzzle
    A('head', box(0.07, 0.05, 0.04, (0.06, 0.04, 0.05), cy=0.02, cz=0.25))  # nose
    A('jaw', box(0.12, 0.04, 0.14, fur2, cy=-0.02, cz=0.05))
    A('jaw', box(0.10, 0.02, 0.10, (0.75, 0.30, 0.32), cy=0.005, cz=0.05))   # tongue/mouth
    eye_y, eye_z = 0.07, 0.112
    if pup:
        for ex in (0.065, -0.065):
            A('head', box(0.055, 0.06, 0.02, (0.06, 0.04, 0.05), cx=ex, cy=eye_y, cz=eye_z))
            A('head', emissive(box(0.02, 0.02, 0.01, (1, 1, 1), cx=ex + 0.012, cy=eye_y + 0.015, cz=eye_z + 0.012)))
    else:
        # one good eye (right side = -x) glowing amber; scar + patch over the left
        A('head', emissive(box(0.05, 0.035, 0.02, (1.0, 0.78, 0.25), cx=-0.065, cy=eye_y, cz=eye_z)), tag='eye_open')
        A('head', box(0.022, 0.03, 0.01, (0.05, 0.03, 0.04), cx=-0.065, cy=eye_y, cz=eye_z + 0.012), tag='eye_open')
        A('head', emissive(box(0.055, 0.012, 0.02, (1.0, 0.78, 0.25), cx=-0.065, cy=eye_y - 0.004, cz=eye_z)), tag='eye_squint')
        A('head', transform(box(0.08, 0.025, 0.03, dark), rot_z(0.35), (-0.065, eye_y + 0.02, eye_z)), tag='eye_squint')
        A('head', box(0.08, 0.075, 0.03, dark, cx=0.065, cy=eye_y, cz=eye_z))          # patch
        A('head', transform(box(0.015, 0.14, 0.02, (0.78, 0.55, 0.52)), rot_z(0.5), (0.06, 0.07, 0.125)))  # scar
        A('head', transform(box(0.30, 0.02, 0.02, dark), rot_z(-0.25), (0, 0.065, 0.12)))     # patch strap
        A('neck', cylinder(0.11, 0.04, (0.60, 0.15, 0.12), n=8, cy=0.0))   # red collar
    ear = tapered_box(0.09, 0.04, 0.01, 0.01, 0.14 if not pup else 0.11, dark)
    A('earL', transform(ear, rot_z(-0.25)))
    A('earR', transform(ear, rot_z(0.25)))
    A('tail1', transform(box(0.07, 0.07, 0.20, fur), None, (0, 0, -0.09)))
    A('tail2', transform(box(0.06, 0.06, 0.18, fur2), None, (0, 0, -0.08)))
    for nm in ('fL', 'fR', 'bL', 'bR'):
        A(nm + '1', box(0.10, leg + 0.04, 0.12, fur, cy=-leg / 2))
        A(nm + '2', box(0.075, leg, 0.08, fur, cy=-leg / 2))
        A(nm + '3', box(0.09, 0.05, 0.12, fur2 if not pup else dark, cy=-0.025, cz=0.03))
    R.scale_default = 0.52 if pup else 1.0
    return R


# ---------------------------------------------------------------- vehicles --
def truck(variant=0):
    """Returns Rig with chassis root + 4 wheel bones; faces +z."""
    R = Rig()
    paint = (0.55, 0.50, 0.42) if variant == 0 else (0.42, 0.40, 0.30)
    rust = (0.55, 0.28, 0.18)
    dark = (0.12, 0.10, 0.12)
    R.bone('chassis', None, (0, 0.55, 0))
    for nm, off in (('wFL', (0.95, -0.12, 1.45)), ('wFR', (-0.95, -0.12, 1.45)),
                    ('wBL', (0.95, -0.12, -1.35)), ('wBR', (-0.95, -0.12, -1.35))):
        R.bone(nm, 'chassis', off)
        w = transform(cylinder(0.43, 0.32, dark, n=10, cy=-0.16), rot_z(np.pi / 2))
        hub = transform(cylinder(0.18, 0.34, (0.55, 0.52, 0.50), n=6, cy=-0.17), rot_z(np.pi / 2))
        R.attach(nm, merge([w, hub]))
    A = R.attach
    A('chassis', box(2.0, 0.30, 4.9, dark, cy=0.05))                         # frame
    A('chassis', box(2.05, 0.65, 1.7, paint, cy=0.48, cz=1.55))              # hood
    A('chassis', box(2.1, 0.30, 0.25, (0.25, 0.24, 0.26), cy=0.30, cz=2.45)) # bumper/ram
    for k in range(5):
        A('chassis', box(0.06, 0.55, 0.05, (0.28, 0.27, 0.30), cx=-0.8 + 0.4 * k, cy=0.55, cz=2.48))
    A('chassis', box(1.95, 0.85, 1.3, paint, cy=1.18, cz=0.2))               # cab
    A('chassis', box(1.75, 0.45, 0.04, (0.20, 0.14, 0.28), cy=1.28, cz=0.86))  # windshield
    A('chassis', box(2.0, 0.08, 0.05, rust, cy=0.82, cz=0.95))
    A('chassis', emissive(box(0.30, 0.18, 0.04, (1.0, 0.86, 0.55), cx=0.72, cy=0.62, cz=2.42)))
    A('chassis', emissive(box(0.30, 0.18, 0.04, (1.0, 0.86, 0.55), cx=-0.72, cy=0.62, cz=2.42)))
    A('chassis', box(2.05, 0.55, 2.4, paint, cy=0.48, cz=-1.35))             # bed
    A('chassis', box(0.06, 0.6, 2.2, rust, cx=1.03, cy=0.58, cz=-1.35))
    A('chassis', box(0.06, 0.6, 2.2, rust, cx=-1.03, cy=0.58, cz=-1.35))
    # skull-ish warpaint stripe on the doors
    for sx in (1, -1):
        for k in range(3):
            A('chassis', transform(box(0.02, 0.62, 0.07, (0.80, 0.16, 0.12)), rot_x(0.5), (1.0 * sx, 1.0, -0.05 + 0.2 * k)))
    A('chassis', transform(cylinder(0.26, 0.62, (0.62, 0.22, 0.14), n=8), None, (0.62, 0.76, -0.45)), tag='drum')   # fuel drum
    A('chassis', transform(cylinder(0.27, 0.05, (0.85, 0.75, 0.30), n=8), None, (0.62, 1.08, -0.45)), tag='drum')
    return R


def cage(door_angle=0.0):
    """Small wire cage (~0.9 m). Door on +z face hinged on +x edge. Origin at bottom center."""
    bar = (0.22, 0.22, 0.26)
    parts = [box(0.92, 0.05, 0.72, (0.35, 0.25, 0.18), cy=0.025)]
    for x in np.linspace(-0.44, 0.44, 7):
        parts.append(box(0.025, 0.62, 0.025, bar, cx=x, cy=0.34, cz=-0.34))
    for z in np.linspace(-0.34, 0.34, 6):
        parts.append(box(0.025, 0.62, 0.025, bar, cx=-0.44, cy=0.34, cz=z))
        parts.append(box(0.025, 0.62, 0.025, bar, cx=0.44, cy=0.34, cz=z))
    parts.append(box(0.92, 0.04, 0.72, bar, cy=0.66))
    parts.append(box(0.92, 0.04, 0.04, bar, cy=0.40, cz=-0.34))
    # door
    door = [box(0.025, 0.62, 0.025, bar, cx=-x, cy=0.34) for x in np.linspace(0.0, 0.88, 7)]
    door += [box(0.9, 0.04, 0.03, bar, cx=-0.44, cy=0.06), box(0.9, 0.04, 0.03, bar, cx=-0.44, cy=0.62)]
    door.append(box(0.08, 0.10, 0.06, (0.80, 0.62, 0.25), cx=-0.84, cy=0.36, cz=0.02))   # latch
    d = transform(merge(door), rot_y(-door_angle), (0.44, 0, 0.36))
    parts.append(d)
    return merge(parts)


# ---------------------------------------------------------------- poses -----
def H(**kw):
    """humanoid pose helper; values in radians as (rx,ry,rz) or scalar rx."""
    p = {}
    for k, v in kw.items():
        p[k] = (v, 0.0, 0.0) if np.isscalar(v) else tuple(v)
    return p

def blend(a, b, t):
    keys = set(a) | set(b)
    out = {}
    for k in keys:
        va = np.asarray(a.get(k, (0, 0, 0)), 'f4'); vb = np.asarray(b.get(k, (0, 0, 0)), 'f4')
        out[k] = tuple((va * (1 - t) + vb * t).tolist())
    return out

def add(a, b):
    out = dict(a)
    for k, v in b.items():
        va = np.asarray(out.get(k, (0, 0, 0)), 'f4')
        out[k] = tuple((va + np.asarray(v, 'f4')).tolist())
    return out

STAND = H(uarmL=(0.05, 0, 0.12), uarmR=(0.05, 0, -0.12), farmL=-0.15, farmR=-0.15)

def run_cycle(p, amp=1.0, lean=0.28):
    s, c = np.sin(p), np.cos(p)
    return H(hips=(0, 0.12 * s * amp, 0), spine=(lean, -0.1 * s * amp, 0), chest=(0.05, -0.15 * s * amp, 0),
             head=-lean * 0.6,
             thighL=-0.95 * s * amp - 0.15, thighR=0.95 * s * amp - 0.15,
             shinL=0.25 + 1.1 * max(0, np.sin(p - 1.2)) * amp + 0.3,
             shinR=0.25 + 1.1 * max(0, np.sin(p + np.pi - 1.2)) * amp + 0.3,
             footL=-0.2, footR=-0.2,
             uarmL=(0.85 * s * amp, 0, 0.15), uarmR=(-0.85 * s * amp, 0, -0.15),
             farmL=-1.5, farmR=-1.5)

def walk_cycle(p, amp=1.0):
    s = np.sin(p)
    return H(hips=(0, 0.06 * s, 0), spine=0.05, thighL=-0.45 * s * amp, thighR=0.45 * s * amp,
             shinL=0.1 + 0.6 * max(0, np.sin(p - 1.0)) * amp, shinR=0.1 + 0.6 * max(0, np.sin(p + np.pi - 1.0)) * amp,
             uarmL=(0.35 * s * amp, 0, 0.12), uarmR=(-0.35 * s * amp, 0, -0.12), farmL=-0.3, farmR=-0.3)

def run_bob(p, amp=1.0):
    return 0.07 * abs(np.cos(p)) * amp

def dog_gallop(p, amp=1.0):
    s = np.sin
    pose = {
        'body': (0.06 * s(p) * amp, 0, 0), 'chest': (-0.08 * s(p) * amp, 0, 0),
        'neck': (-0.2 + 0.12 * s(p + 1), 0, 0), 'head': (0.25 - 0.1 * s(p + 1), 0, 0),
        'tail1': (0.35 + 0.15 * s(p * 2), 0.2 * s(p), 0), 'tail2': (-0.2, 0, 0),
        'earL': (0.6, 0, 0), 'earR': (0.6, 0, 0),
    }
    for nm, ph in (('fL', 0.0), ('fR', 0.45), ('bL', np.pi), ('bR', np.pi + 0.45)):
        a = s(p + ph)
        pose[nm + '1'] = (-0.75 * a * amp, 0, 0)
        knee = 0.9 * max(0, s(p + ph - 1.3)) * amp
        pose[nm + '2'] = ((knee if nm[0] == 'b' else -knee), 0, 0)
        pose[nm + '3'] = (0.2 * a, 0, 0)
    return pose

def dog_trot(p, amp=1.0):
    s = np.sin
    pose = {'neck': (-0.1, 0, 0), 'head': (0.12 + 0.04 * s(2 * p), 0, 0),
            'tail1': (0.5, 0.25 * s(p), 0), 'tail2': (-0.1, 0.2 * s(p - 0.5), 0),
            'earL': (0.2, 0, 0), 'earR': (0.2, 0, 0)}
    for nm, ph in (('fL', 0.0), ('bR', 0.0), ('fR', np.pi), ('bL', np.pi)):
        a = s(p + ph)
        pose[nm + '1'] = (-0.45 * a * amp, 0, 0)
        knee = 0.6 * max(0, s(p + ph - 1.2)) * amp
        pose[nm + '2'] = ((knee if nm[0] == 'b' else -knee), 0, 0)
    return pose

DOG_SIT = {'body': (-0.55, 0, 0), 'chest': (-0.1, 0, 0), 'neck': (0.25, 0, 0), 'head': (0.25, 0, 0),
           'fL1': (0.55, 0, 0), 'fR1': (0.55, 0, 0), 'bL1': (-1.2, 0, 0), 'bR1': (-1.2, 0, 0),
           'bL2': (1.9, 0, 0), 'bR2': (1.9, 0, 0), 'bL3': (-0.6, 0, 0), 'bR3': (-0.6, 0, 0),
           'tail1': (-0.6, 0, 0), 'earL': (0.1, 0, 0), 'earR': (0.1, 0, 0)}

DOG_CROUCH = {'body': (0.0, 0, 0), 'chest': (0.15, 0, 0), 'neck': (-0.5, 0, 0), 'head': (0.55, 0, 0),
              'fL1': (-0.5, 0, 0), 'fR1': (-0.5, 0, 0), 'fL2': (1.0, 0, 0), 'fR2': (1.0, 0, 0),
              'fL3': (-0.5, 0, 0), 'fR3': (-0.5, 0, 0),
              'bL1': (-0.7, 0, 0), 'bR1': (-0.7, 0, 0), 'bL2': (1.3, 0, 0), 'bR2': (1.3, 0, 0),
              'tail1': (-0.1, 0, 0), 'earL': (0.9, 0, 0), 'earR': (0.9, 0, 0)}

DOG_STAND = {'neck': (-0.15, 0, 0), 'head': (0.1, 0, 0), 'tail1': (0.4, 0, 0), 'earL': (0.0, 0, 0), 'earR': (0.0, 0, 0)}
