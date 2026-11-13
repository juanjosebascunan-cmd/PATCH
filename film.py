"""The short: shot list, blocking, animation, cameras, subtitles, sound cues.
Global time T in seconds (0..60). frame(T) -> dict(dyn mesh, cam, light, flash, black, shot).
"""
import numpy as np
from geo import transform, rot_y, rot_x, rot_z, box
from rig import (humanoid, dog, truck, cage, merge, H, add, blend, STAND, run_cycle, walk_cycle,
                 run_bob, dog_gallop, dog_trot, DOG_SIT, DOG_CROUCH, DOG_STAND, emissive)
from world import height, dust_cloud, flash_star, explosion, sparks, puff, BOULDER_POS, DUST

DUR = 60.0
RANGER, HUNTER, BOSS = humanoid('ranger'), humanoid('hunter'), humanoid('boss')
DOG, PUP = dog(False), dog(True)
TRUCK1, TRUCK2 = truck(0), truck(1)
PUP_S = 0.52

def gy(x, z):
    return float(height(x, z))

def ease(u):
    u = np.clip(u, 0, 1)
    return u * u * (3 - 2 * u)

def eout(u):
    u = np.clip(u, 0, 1)
    return 1 - (1 - u) ** 3

def lerp(a, b, u):
    return np.asarray(a, 'f8') * (1 - u) + np.asarray(b, 'f8') * u

def kf(t, keys, fn=ease):
    """keys: [(t, value), ...] sorted. Eased piecewise interpolation."""
    if t <= keys[0][0]:
        return np.asarray(keys[0][1], 'f8')
    for (t0, v0), (t1, v1) in zip(keys, keys[1:]):
        if t <= t1:
            return lerp(v0, v1, fn((t - t0) / (t1 - t0)))
    return np.asarray(keys[-1][1], 'f8')

def yaw_to(src, dst):
    d = np.asarray(dst, 'f8') - np.asarray(src, 'f8')
    return float(np.arctan2(d[0], d[2]))

def shake(T, amp, seed=0.0):
    return np.array([np.sin(T * 37 + seed) * 0.6 + np.sin(T * 23.3 + 1 + seed) * 0.4,
                     np.sin(T * 31 + 2 + seed) * 0.6 + np.sin(T * 17.1 + seed) * 0.4,
                     np.sin(T * 29 + 4 + seed) * 0.5]) * amp

def handheld(T, amp=0.03):
    return np.array([np.sin(T * 1.3) + 0.5 * np.sin(T * 2.9 + 1), np.sin(T * 1.7 + 2) + 0.4 * np.sin(T * 3.3),
                     np.sin(T * 1.1 + 4)]) * amp

def onfloor(p):
    return np.array([p[0], gy(p[0], p[2]), p[2]])

# ---------------------------------------------------------------- poses ----
CROUCH = add(STAND, H(hips=(0, 0.25, 0), spine=0.35, chest=0.1, head=-0.25,
                       thighL=-1.45, shinL=1.45, footL=0.0, thighR=0.25, shinR=1.45, footR=0.7,
                       uarmL=(-0.55, 0, 0.1), farmL=-0.9, uarmR=(-0.2, 0, -0.25), farmR=-0.9))
CROUCH_OFF = (0, -0.43, 0)
AIM = add(STAND, H(chest=(0, -0.15, 0), head=(0, 0.15, 0), uarmR=(-1.52, 0.10, 0.12), farmR=-0.02, handR=0.05,
                   uarmL=(0.15, 0, 0.25), farmL=-0.4, thighL=-0.15, thighR=0.18, shinR=0.1))
AIM2 = add(AIM, H(uarmL=(-1.25, -0.4, -0.5), farmL=-0.7))  # two-handed
DRAW = add(STAND, H(uarmR=(0.25, 0, -0.35), farmR=-0.9))
SLIDE = H(hips=(0.0, 1.2, 0), spine=0.38, chest=(0.1, -0.4, 0), head=(-0.2, -0.6, 0),
          thighL=-0.95, shinL=1.35, footL=-0.3, thighR=-0.25, shinR=1.0, footR=-0.5,
          uarmL=(-0.4, 0, 1.25), farmL=-0.4, uarmR=(0.35, 0, -0.9), farmR=-0.3)
SLIDE_OFF = (0, -0.33, 0)
PUNCH_WIND = add(STAND, H(spine=0.15, chest=(0, 0.55, 0), uarmL=(0.3, 0, 0.5), farmL=-2.0,
                          thighL=-0.4, shinL=0.4, thighR=0.25, uarmR=(-0.4, 0, -0.3), farmR=-1.2))
PUNCH_HIT = add(STAND, H(spine=0.3, chest=(0.05, -0.6, 0), head=(0, 0.4, 0), uarmL=(-1.6, 0.3, -0.1), farmL=-0.05,
                         thighL=-0.6, shinL=0.5, thighR=0.45, shinR=0.2, uarmR=(-0.2, 0, -0.4), farmR=-1.4))
RIFLE_PORT = add(STAND, H(uarmR=(-0.45, 0, -0.1), farmR=-1.25, uarmL=(-0.75, 0, -0.3), farmL=-1.3))
RIFLE_AIM = add(STAND, H(chest=(0, -0.25, 0), head=(0.05, 0.2, 0), uarmR=(-1.40, 0.15, 0.10), farmR=-0.35,
                         uarmL=(-1.35, -0.25, -0.45), farmL=-0.55, thighL=-0.25, thighR=0.2))
HANDS_UP = add(STAND, H(spine=-0.15, head=-0.15, uarmL=(-2.8, 0, 0.35), farmL=-0.4,
                        uarmR=(-2.8, 0, -0.35), farmR=-0.4))
YELL = add(STAND, H(spine=0.2, chest=0.15, head=-0.35, uarmR=(-1.3, 0.3, 0.2), farmR=-0.2,
                    uarmL=(0.3, 0, 0.6), farmL=-1.0, thighL=-0.3, thighR=0.2))
KNEEL = add(HANDS_UP, H(hips=0.1, thighL=0.1, shinL=1.5, footL=0.4, thighR=0.05, shinR=1.5, footR=0.4))
KNEEL_OFF = (0, -0.44, 0)
SUPINE = H(hips=(-1.45, 0, 0), spine=-0.1, head=0.3, uarmL=(-0.2, 0, 1.3), farmL=-0.5, uarmR=(-0.1, 0, -1.2),
           farmR=-0.4, thighL=-0.3, shinL=0.5, thighR=-0.1, shinR=0.2)
SUPINE_OFF = (0, -0.82, 0)
STAGGER = add(STAND, H(spine=-0.35, chest=-0.2, head=-0.35, uarmL=(-0.4, 0, 0.9), farmL=-0.6,
                       uarmR=(-0.3, 0, -0.9), farmR=-0.6, thighL=-0.4, thighR=0.15))
DIVE = H(hips=(1.35, 0, 0), spine=0.1, head=-0.6, uarmL=(-2.6, 0, 0.3), uarmR=(-2.6, 0, -0.3),
         thighL=0.3, thighR=0.1)
DIVE_OFF = (0, -0.78, 0)

PUP_DANGLE = {'body': (0.1, 0, 0), 'neck': (-0.6, 0, 0), 'head': (0.5, 0, 0),
              'fL1': (0.35, 0, 0), 'fR1': (0.25, 0, 0), 'bL1': (0.2, 0, 0), 'bR1': (0.3, 0, 0),
              'fL2': (-0.2, 0, 0), 'fR2': (-0.2, 0, 0), 'tail1': (-0.9, 0, 0), 'earL': (0.8, 0, 0), 'earR': (0.8, 0, 0)}
DOG_LEAP = {'body': (-0.25, 0, 0), 'neck': (-0.35, 0, 0), 'head': (0.3, 0, 0),
            'fL1': (-1.3, 0, 0), 'fR1': (-1.2, 0, 0), 'fL2': (0.3, 0, 0), 'fR2': (0.3, 0, 0),
            'bL1': (1.0, 0, 0), 'bR1': (0.9, 0, 0), 'bL2': (0.4, 0, 0), 'bR2': (0.4, 0, 0),
            'tail1': (0.1, 0, 0), 'earL': (1.0, 0, 0), 'earR': (1.0, 0, 0)}
DOG_LAND = {'body': (0.25, 0, 0), 'chest': (0.1, 0, 0), 'neck': (-0.2, 0, 0), 'head': (0.15, 0, 0),
            'fL1': (-0.4, 0, 0), 'fR1': (-0.3, 0, 0), 'fL2': (0.6, 0, 0), 'fR2': (0.6, 0, 0),
            'bL1': (-0.9, 0, 0), 'bR1': (-0.9, 0, 0), 'bL2': (1.4, 0, 0), 'bR2': (1.4, 0, 0)}
DOG_GROWL = dict(DOG_CROUCH, jaw=(0.25, 0, 0), earL=(1.1, 0, 0), earR=(1.1, 0, 0))
PUP_REAR = {'body': (-1.0, 0, 0), 'chest': (-0.1, 0, 0), 'neck': (0.4, 0, 0), 'head': (0.5, 0, 0),
            'fL1': (-0.9, 0, 0), 'fR1': (-0.7, 0, 0), 'fL2': (-0.6, 0, 0), 'fR2': (-0.6, 0, 0),
            'bL1': (-1.0, 0, 0), 'bR1': (-1.0, 0, 0), 'bL2': (0.6, 0, 0), 'bR2': (0.6, 0, 0),
            'tail1': (0.6, 0, 0), 'jaw': (0.3, 0, 0)}

def dog_tailwag(pose, T, amp=0.5, speed=14.0):
    p = dict(pose)
    t1 = np.asarray(p.get('tail1', (0.4, 0, 0)), 'f8').copy(); t1[1] = amp * np.sin(T * speed); t1[0] = max(t1[0], 0.5)
    p['tail1'] = tuple(t1)
    return p

# ---------------------------------------------------------- vehicles -------
T_SHOT = 24.80          # ranger's shot hits truck 1's front-left tire
CRASH_T = 26.0
V1 = 8.5

def truck1_state(T):
    """returns pos, yaw, roll, pitch, wheel_angle, blown"""
    if T < T_SHOT:
        z = -6.0 + V1 * (T - T_SHOT)
        return np.array([2.0, 0, z]), 0.0, 0.0, 0.0, (z + 200) / 0.43, False
    u = min((T - T_SHOT) / (CRASH_T - T_SHOT), 1.0)
    P0, P1, P2, P3 = map(np.array, ((2.0, -6.0), (2.0, -0.6), (6.0, 1.6), (8.0, 3.6)))
    s = eout(u) * 0.96 + u * 0.04
    b = lambda s: (1 - s) ** 3 * P0 + 3 * (1 - s) ** 2 * s * P1 + 3 * (1 - s) * s * s * P2 + s ** 3 * P3
    p = b(s); dp = b(min(s + 0.01, 1)) - b(max(s - 0.01, 0))
    yaw = float(np.arctan2(dp[0], dp[1])) + 0.35 * np.sin(np.pi * u)   # fishtail
    roll = 0.10 * np.sin(np.pi * u)
    pitch = 0.0
    if T >= CRASH_T:
        a = T - CRASH_T
        yaw = float(np.arctan2(*(P3 - P2))) + 0.08
        pitch = 0.10 * np.exp(-a * 5) * np.cos(a * 22) + 0.05      # nose dips into the rock
        roll = 0.06 * np.exp(-a * 4) * np.sin(a * 18)
    wheel = (-6 + 200) / 0.43 + s * 12 / 0.43
    return np.array([p[0], 0, p[1]]), yaw, roll, pitch, wheel, True

T2_BRAKE = 24.95
T2_STOP_DUR = 1.7

def truck2_state(T):
    if T < T2_BRAKE:
        z = -6.0 + V1 * (T - T_SHOT) - 13.0
        return np.array([2.4, 0, z]), 0.0, 0.0, (z + 200) / 0.43
    a = min(T - T2_BRAKE, T2_STOP_DUR)
    z0 = -6.0 + V1 * (T2_BRAKE - T_SHOT) - 13.0
    z = z0 + V1 * a - 0.5 * (V1 / T2_STOP_DUR) * a * a
    yaw = -0.18 * ease(a / T2_STOP_DUR)
    pitch = 0.05 * np.sin(np.pi * a / T2_STOP_DUR) if T < T2_BRAKE + T2_STOP_DUR else 0.03 * np.exp(-(T - T2_BRAKE - T2_STOP_DUR) * 6) * np.cos((T - T2_BRAKE - T2_STOP_DUR) * 25)
    return np.array([2.4 + 0.3 * ease(a / T2_STOP_DUR), 0, z]), yaw, pitch, (z + 200) / 0.43

def truck_mesh(rig, pos, yaw, wheel, roll=0.0, pitch=0.0, hide=(), blown=False, lift=0.0):
    pose = {'chassis': (pitch, 0, roll)}
    for w in ('wFL', 'wFR', 'wBL', 'wBR'):
        pose[w] = (wheel, 0, 0)
    off = (0, lift, 0)
    m, Ms = rig.build(pose, pos, yaw, hide=hide, root_off=off)
    return m, Ms

def local_to_world(Ms, bone, p):
    M = Ms[bone]
    return M[:3, :3] @ np.asarray(p, 'f4') + M[:3, 3]

BED_Y = 0.755          # bed top relative to chassis bone
CAGE_LOCAL = (0.42, BED_Y, -1.20)
DRUM_LOCAL = (0.62, 1.05, -0.45)

def cage_world(Ms2, yaw2, door):
    """cage on truck 2 bed, door facing the tailgate."""
    M = Ms2['chassis']
    c = cage(-door)
    R = M[:3, :3] @ rot_y(np.pi)
    v = c['v'] @ R.T + (M[:3, :3] @ np.asarray(CAGE_LOCAL, 'f4') + M[:3, 3])
    return {'v': v.astype('f4'), 'c': c['c'], 'flag': c['flag']}

def cage_frame(Ms2):
    """world matrix (R, t) of cage's local frame (origin bed, +z = toward door/tailgate)."""
    M = Ms2['chassis']
    R = M[:3, :3] @ rot_y(np.pi)
    t = M[:3, :3] @ np.asarray(CAGE_LOCAL, 'f4') + M[:3, 3]
    return R, t

def in_frame(Rt, p):
    R, t = Rt
    return R @ np.asarray(p, 'f4') + t

def frame_yaw(Rt, local_yaw):
    R, _ = Rt
    f = R @ np.array([np.sin(local_yaw), 0, np.cos(local_yaw)], 'f4')
    return float(np.arctan2(f[0], f[2]))

# ----------------------------------------------------------- blocking ------
RIM = np.array([-40.4, 0, 4.0]); RIM[1] = gy(RIM[0], RIM[2])
DOG_RIM = np.array([-39.6, 0, 2.7]); DOG_RIM[1] = gy(DOG_RIM[0], DOG_RIM[2])
SLIDE_A = np.array([-39.6, 0, 4.4])
SLIDE_B = np.array([-12.5, 0, 3.0])
FIRE_POS = np.array([-11.8, 0, 2.6])
RP = np.array([-7.0, 0, -0.5])            # ranger's fighting position
BOSS_P = np.array([0.0, 0, -9.3])
H3_P = np.array([-0.8, 0, -7.0])
H2_P = np.array([-1.6, 0, -12.0])
H1_P = np.array([5.8, 0, 1.2])

def ranger_mesh(T, pose, pos, yaw, gun=True, root_off=(0, 0, 0)):
    hide = () if gun else ('gun',)
    m, Ms = RANGER.build(pose, pos, yaw, hide=hide, root_off=root_off)
    return m, Ms

def muzzle(Ms, cam_eye, size=0.45, seed=0):
    p = local_to_world(Ms, 'handR', (0, -0.42, 0.03))
    return flash_star(p, size, cam_eye, seed=seed), p

def hunter_mesh(rig, pose, pos, yaw, gun=True, root_off=(0, 0, 0)):
    return rig.build(pose, pos, yaw, hide=() if gun else ('gun',), root_off=root_off)

def rifle_free(pos, rot):
    """a loose rifle tumbling (same parts as hunter rifle)."""
    wood = (0.42, 0.24, 0.14); metal = (0.30, 0.30, 0.36)
    r = merge([box(0.05, 0.36, 0.08, wood, cy=0.08), box(0.045, 0.62, 0.05, metal, cy=-0.38),
               box(0.05, 0.14, 0.07, metal, cy=-0.12)])
    return transform(r, rot, pos)

SUBS = [
    (7.0, 9.4, "RANGER: Hear that, Patch?"),
    (13.5, 15.0, "[pup whimpering]"),
    (15.2, 16.9, "[low growl]"),
    (17.3, 19.0, "RANGER: Yeah. Let's go to work."),
    (28.5, 30.0, "MASKED BOSS: Who's out there?!"),
    (40.7, 42.5, "MASKED BOSS: That pup's mine, old dog."),
    (44.3, 46.0, "RANGER: Not anymore."),
    (47.5, 48.7, "RANGER: Go, Patch!"),
    (54.2, 56.8, "RANGER: Looks like you've got a partner, Patch."),
]

# sound cue list (time, kind) consumed by the audio builder
CUES = []

def cue(t, kind, **kw):
    CUES.append(dict(t=t, kind=kind, **kw))

cue(0.0, 'wind'); cue(T_SHOT - 0.02, 'gunshot'); cue(T_SHOT + 0.03, 'tire_pop'); cue(T_SHOT + 0.05, 'skid', dur=1.2)
cue(CRASH_T, 'crash'); cue(T2_BRAKE, 'brake', dur=T2_STOP_DUR)
cue(30.9, 'gunshot'); cue(30.95, 'ricochet'); cue(33.45, 'punch'); cue(33.95, 'body_fall')
cue(35.6, 'dog_leap'); cue(36.2, 'thud_small'); cue(36.9, 'latch'); cue(37.2, 'cage_creak'); cue(38.6, 'pup_yip')
cue(42.6, 'gunshot'); cue(43.25, 'ricochet'); cue(43.3, 'metal_clatter'); cue(46.0, 'dog_leap')
cue(48.55, 'gunshot'); cue(48.8, 'explosion'); cue(8.0, 'engine_far'); cue(15.3, 'growl', dur=1.6)
cue(13.7, 'whimper'); cue(55.3, 'pup_yip'); cue(36.5, 'growl', dur=0.6); cue(28.6, 'yell'); cue(40.7, 'yell_low')

# ---------------------------------------------------------------- shots ----
def S01(T):
    """0-6 establishing: silhouettes on the rim against the sun."""
    u = T / 6.0
    dyn = []
    m, _ = ranger_mesh(T, add(STAND, H(head=(0.05, 0.25 * np.sin(T * 0.4), 0))), RIM, yaw=1.35, gun=False); dyn.append(m)
    m, _ = DOG.build(dog_tailwag(DOG_SIT, T, 0.15, 3), DOG_RIM, 1.6); dyn.append(m)
    from engine import SUN_DIR
    e0 = RIM + np.array([-SUN_DIR[0], 0, -SUN_DIR[2]]) / np.hypot(SUN_DIR[0], SUN_DIR[2]) * 60.0
    e0[1] = gy(e0[0], e0[2]) + 9.0
    e1 = RIM + np.array([-SUN_DIR[0], 0, -SUN_DIR[2]]) / np.hypot(SUN_DIR[0], SUN_DIR[2]) * 44.0 + np.array([2.0, 0, 0])
    e1[1] = gy(e1[0], e1[2]) + 10.5
    eye = lerp(e0, e1, ease(u)) + handheld(T, 0.05)
    tgt = lerp(RIM + np.array([1.5, 4.0, 0]), RIM + np.array([0.5, 1.3, 0]), ease(u))
    return dict(dyn=dyn, cam=dict(eye=eye, target=tgt, fov=lerp(30, 24, ease(u))), light=(RIM, 30))

def S02(T):
    """6-10 ranger crouched on rim, dog hears engines."""
    t = T - 6.0
    dyn = []
    m, _ = ranger_mesh(T, add(CROUCH, H(head=(0.1, 0.25 * ease((t - 2.4) / 0.6), 0))), RIM, 1.5, gun=False, root_off=CROUCH_OFF)
    dyn.append(m)
    turn = ease((t - 1.8) / 0.5)
    dp = dict(DOG_SIT)
    dp['neck'] = (0.25, 0.9 * turn, 0); dp['head'] = (0.15 - 0.2 * turn, 0.5 * turn, 0)
    perk = ease((t - 1.6) / 0.15)
    dp['earL'] = (0.1 - 0.5 * perk, 0, 0); dp['earR'] = (0.1 - 0.5 * perk, 0, 0)
    m, _ = DOG.build(dp, DOG_RIM, 1.6); dyn.append(m)
    eye = lerp(RIM + np.array([3.4, 0.95, 3.0]), RIM + np.array([2.7, 0.9, 2.5]), ease(t / 4)) + handheld(T, 0.02)
    tgt = lerp(RIM + np.array([0.6, 0.7, 0.0]), RIM + np.array([0.5, 0.75, -0.2]), ease(t / 4))
    return dict(dyn=dyn, cam=dict(eye=eye, target=tgt, fov=40), light=(RIM, 12))

def convoy(T, hide_drum=False, door=0.0, pup_in=True, hunters_in_bed=True, extra_hide=()):
    """both trucks + cargo; returns meshes, truck2 matrices."""
    dyn = []
    p1, y1, r1, pi1, w1, blown = truck1_state(T)
    m, Ms1 = truck_mesh(TRUCK1, p1, y1, w1, r1, pi1); dyn.append(m)
    p2, y2, pi2, w2 = truck2_state(T)
    hide = ('drum',) if hide_drum else ()
    m, Ms2 = truck_mesh(TRUCK2, p2, y2, w2, 0.0, pi2, hide=hide); dyn.append(m)
    dyn.append(cage_world(Ms2, y2, door))
    CF = cage_frame(Ms2)
    if pup_in:
        pp = in_frame(CF, (0.0, 0.02, -0.05))
        m, _ = PUP.build(dict(DOG_SIT, head=(0.35, 0.3 * np.sin(T * 0.9), 0), earL=(0.7, 0, 0), earR=(0.7, 0, 0)),
                         pp, frame_yaw(CF, 0.0), scale=PUP_S)
        dyn.append(m)
    if hunters_in_bed:
        for k, (lx, lz, rig, ph) in enumerate(((-0.55, -0.35, HUNTER, 0.0), (-0.6, -1.25, HUNTER, 1.3))):
            pos = local_to_world(Ms2, 'chassis', (lx, BED_Y, lz))
            sway = 0.08 * np.sin(T * 3 + ph)
            m, _ = rig.build(add(RIFLE_PORT, H(spine=(sway, 0, sway * 0.5))), pos, y2 + (0.9 if lx < 0 else -0.6))
            dyn.append(m)
    return dyn, Ms1, Ms2, (p1, y1), (p2, y2)

def truck_dust(T, Ms, seed, rate=7, life=1.6, size=0.5):
    """trailing dust from rear wheels: emit puffs at past wheel positions (re-simulated)."""
    out = []
    for k in range(int(life * rate)):
        te = T - k / rate
        tq = np.floor(te * rate) / rate
        age = T - tq
        if age < 0 or age > life:
            continue
        out += dust_cloud(Ms(tq), age, n=2, spread=1.3, rise=0.6, size=size, seed=seed + int(tq * rate), life=life)
    return out

def S03(T):
    """10-15 low angle: the convoy roars past; the pup in the cage."""
    t = T - 10.0
    dyn, Ms1, Ms2, (p1, y1), (p2, y2) = convoy(T)
    dyn += truck_dust(T, lambda tq: truck1_state(tq)[0] + np.array([0, 0.3, -2.0]), 100)
    dyn += truck_dust(T, lambda tq: truck2_state(tq)[0] + np.array([0, 0.3, -2.0]), 300)
    camz = -104.0
    eye = np.array([6.6, 0.45, camz]) + handheld(T, 0.04)
    # pan: watch truck1 approach, then hold on truck2 cargo as it passes
    tg1 = p1 + np.array([0, 1.4, 0]); tg2 = p2 + np.array([0, 1.7, -1.5])
    w = ease((t - 2.0) / 1.6)
    tgt = lerp(tg1, tg2, w)
    tgt = np.where(np.array([True, True, True]), tgt, tgt)
    if p2[2] > camz + 6:   # after it passes keep pan smooth
        tgt = lerp(tgt, np.array([2.4, 1.8, camz + 14]), ease((p2[2] - camz - 6) / 8))
    sh = shake(T, 0.035 * np.exp(-((p1[2] - camz) ** 2) / 40) + 0.035 * np.exp(-((p2[2] - camz) ** 2) / 40))
    return dict(dyn=dyn, cam=dict(eye=eye + sh, target=tgt, fov=50), light=(np.array([2, 0, camz]), 40))

def S04a(T):
    """15-17 extreme close on Patch's good eye."""
    t = T - 15.0
    hide = ('eye_open',) if t > 0.55 else ('eye_squint',)
    pose = blend(DOG_SIT, dict(DOG_SIT, neck=(0.35, 0.5, 0), head=(0.05, 0.25, 0), earL=(0.9, 0, 0), earR=(0.9, 0, 0),
                                jaw=(0.12, 0, 0)), ease(t / 0.6))
    m, Ms = DOG.build(pose, DOG_RIM, 1.6, hide=hide)
    eyep = local_to_world(Ms, 'head', (-0.065, 0.07, 0.112))
    fwd = local_to_world(Ms, 'head', (-0.065, 0.07, 1.0)) - eyep
    side = local_to_world(Ms, 'head', (-1.0, 0.07, 0.112)) - eyep
    fwd /= np.linalg.norm(fwd); side /= np.linalg.norm(side)
    dist = lerp(1.25, 0.85, ease(t / 2.0))
    eye = eyep + fwd * dist + side * 0.12 + np.array([0, 0.04, 0]) + handheld(T, 0.004)
    return dict(dyn=[m], cam=dict(eye=eye, target=eyep + side * 0.07 + np.array([0, -0.03, 0]), fov=28), light=(DOG_RIM, 6))

def S04b(T):
    """17-19 ranger rises and draws, low angle against the sky."""
    t = T - 17.0
    u = ease(t / 0.9)
    pose = blend(CROUCH, STAND, u)
    off = lerp(CROUCH_OFF, (0, 0, 0), u)
    dr = ease((t - 1.0) / 0.35)
    pose = blend(pose, DRAW, dr) if t < 1.35 else blend(DRAW, add(STAND, H(uarmR=(-0.25, 0, -0.15), farmR=-0.5)), ease((t - 1.35) / 0.3))
    pose = add(pose, H(head=(0.0, 0.35, 0)))
    m, Ms = ranger_mesh(T, pose, RIM, 1.5, gun=t > 1.15, root_off=off)
    dp = dict(DOG_STAND, neck=(-0.1, 0.4, 0)) if t > 0.4 else DOG_SIT
    m2, _ = DOG.build(dog_tailwag(dp, T, 0.2, 8), DOG_RIM, 1.6)
    eye = RIM + np.array([2.5, 0.3, 1.9]) + handheld(T, 0.015)
    tgt = RIM + np.array([0.0, lerp(0.85, 1.25, u), 0.0])
    return dict(dyn=[m, m2], cam=dict(eye=eye, target=tgt, fov=50), light=(RIM, 10))

def slide_pos(T):
    u = np.clip((T - 19.2) / 4.1, 0, 1)
    s = 0.15 * u + 0.85 * (1 - (1 - u) ** 1.6)
    x = lerp(SLIDE_A[0], SLIDE_B[0], s); z = lerp(SLIDE_A[2], SLIDE_B[2], s) + 0.4 * np.sin(u * 7)
    return np.array([x, gy(x, z), z]), u

def S05(T):
    """19-24 side-tracking: the ranger surfs down the scree, Patch bounding alongside."""
    t = T - 19.0
    dyn = []
    p, u = slide_pos(T)
    if T < 19.2:
        pose = blend(STAND, SLIDE, ease((T - 19.0) / 0.2)); off = lerp((0, 0, 0), SLIDE_OFF, ease((T - 19.0) / 0.2))
        p = SLIDE_A.copy(); p[1] = gy(p[0], p[2])
    elif u < 1:
        pose = add(SLIDE, H(spine=(0.05 * np.sin(T * 9), 0, 0.08 * np.sin(T * 5)), uarmL=(0.1 * np.sin(T * 6), 0, 0)))
        off = SLIDE_OFF
    else:
        k = ease((T - 23.3) / 0.5)
        pose = blend(SLIDE, add(STAND, H(uarmR=(-0.3, 0, -0.2), farmR=-0.5)), k); off = lerp(SLIDE_OFF, (0, 0, 0), k)
    yaw_ = 1.5 + 0.1 * np.sin(T * 3)
    m, Ms = ranger_mesh(T, pose, p, yaw_ - 1.2 * (1 - ease((T - 23.3) / 0.5)) * (u > 0), gun=True, root_off=off)
    dyn.append(m)
    # dust spray from boots, re-simulated history
    for k in range(18):
        te = T - k / 14.0
        if te < 19.2 or te > 23.5:
            continue
        pe, _ = slide_pos(te)
        dyn += dust_cloud(pe + np.array([0.3, 0.1, 0]), T - te, n=2, spread=0.9, rise=0.9, size=0.30,
                          seed=7000 + int(te * 14), life=1.3)
    # dog runs beside, then peels off toward the trucks
    du = np.clip((T - 19.35) / 4.0, 0, 1)
    dx = lerp(SLIDE_A[0] + 0.5, SLIDE_B[0] + 6.0, du); dz = lerp(SLIDE_A[2] + 1.6, -4.0, du ** 2.2)
    dpos = np.array([dx, gy(dx, dz), dz])
    nxt = np.array([dx + 0.5, 0, dz - 0.5 * du * 2.2 * du])
    dyaw = 1.57 - 0.6 * du ** 2
    m, _ = DOG.build(dog_gallop(T * 15.0), dpos + np.array([0, 0.08 * abs(np.sin(T * 15)), 0]), dyaw)
    dyn.append(m)
    for k in range(10):
        te = T - k / 10.0
        if te < 19.4 or te > 23.4:
            continue
        dyn += dust_cloud(dpos + np.array([-(T - te) * 6.5, 0.1, 0]), T - te, n=1, spread=0.6, rise=0.7, size=0.32,
                          seed=8000 + int(te * 10), life=0.9)
    if u >= 1:
        dyn += dust_cloud(SLIDE_B + np.array([0.5, 0.2, 0]), T - 23.3, n=7, spread=2.0, rise=1.0, size=0.8, seed=901, life=1.6)
    cam_anchor = p + np.array([0, 1.0, 0])
    eye = cam_anchor + np.array([2.4, 0.7, 8.5]) + handheld(T, 0.05) + shake(T, 0.03)
    tgt = cam_anchor + np.array([2.0, -0.2, 0])
    return dict(dyn=dyn, cam=dict(eye=eye, target=tgt, fov=46), light=(p, 25))

def S06a(T):
    """24-25.2 over-the-shoulder: draw a bead on truck 1's tire. BANG."""
    t = T - 24.0
    dyn, Ms1, Ms2, (p1, y1), (p2, y2) = convoy(T)
    dyn += truck_dust(T, lambda tq: truck1_state(tq)[0] + np.array([0, 0.3, -2.0]), 100)
    dyn += truck_dust(T, lambda tq: truck2_state(tq)[0] + np.array([0, 0.3, -2.0]), 300)
    aim_at = np.array([3.0, 0.5, -1.0])
    yaw_r = yaw_to(FIRE_POS, aim_at)
    rise = ease(t / 0.35)
    pose = blend(add(STAND, H(uarmR=(-0.3, 0, -0.2), farmR=-0.5)), AIM, rise)
    if T > T_SHOT:
        k = np.exp(-(T - T_SHOT) * 10)
        pose = add(pose, H(uarmR=(-0.45 * k, 0, 0), farmR=-0.3 * k))
    p = FIRE_POS.copy(); p[1] = gy(p[0], p[2])
    m, Ms = ranger_mesh(T, pose, p, yaw_r); dyn.append(m)
    back = -np.array([np.sin(yaw_r), 0, np.cos(yaw_r)]); right = np.array([-np.cos(yaw_r), 0, np.sin(yaw_r)])
    eye = p + back * 1.7 + right * 0.85 + np.array([0, 1.6, 0]) + handheld(T, 0.01)
    flash = 0.0
    if 0 <= T - T_SHOT < 2 / 24:
        fs, mp = muzzle(Ms, eye, 0.3, 1); dyn += fs
        flash = 0.35
    eye = eye + shake(T, 0.03 * np.exp(-max(T - T_SHOT, 0) * 9) * (T > T_SHOT))
    tgt = lerp(p + np.array([0, 1.3, 0]) - back * 6, aim_at + np.array([0, 0.8, 0]), 0.5)
    return dict(dyn=dyn, cam=dict(eye=eye, target=tgt, fov=44), light=(p - back * 8, 25), flash=flash)

def S06b(T):
    """25.2-28 wide: blowout, fishtail, crash into the boulder; truck 2 skids to a stop."""
    dyn, Ms1, Ms2, (p1, y1), (p2, y2) = convoy(T, hunters_in_bed=T < 27.2)
    tire = local_to_world(Ms1, 'wFL', (0, 0, 0))
    dyn += sparks(tire, T - T_SHOT, (0, 0, 0), n=10, seed=5, life=0.5)
    for k in range(14):
        te = T - k / 12.0
        if te < T_SHOT or te > CRASH_T:
            continue
        pe = truck1_state(te)[0]
        dyn += dust_cloud(pe + np.array([0, 0.3, -1.5]), T - te, n=2, spread=1.6, rise=1.2, size=0.9, seed=3000 + int(te * 12), life=1.4)
    for k in range(14):
        te = T - k / 12.0
        if te < T2_BRAKE or te > T2_BRAKE + T2_STOP_DUR:
            continue
        pe = truck2_state(te)[0]
        dyn += dust_cloud(pe + np.array([0, 0.25, 1.4]), T - te, n=2, spread=1.4, rise=0.9, size=0.75, seed=4000 + int(te * 12), life=1.4)
    dyn += dust_cloud(np.array(BOULDER_POS) + np.array([-1.0, 1.0, -0.8]), T - CRASH_T, n=10, spread=3.0, rise=1.6, size=1.3, seed=77, life=2.4)
    # steam from the smashed hood
    if T > CRASH_T:
        hood = local_to_world(Ms1, 'chassis', (0, 0.9, 1.9))
        for k in range(6):
            te = T - k * 0.25
            if te > CRASH_T:
                dyn += dust_cloud(hood, T - te, n=1, spread=0.4, rise=1.8, size=0.4, seed=600 + int(te * 4), life=1.5, color=(0.86, 0.80, 0.82))
    # hunters leap off the bed of truck 2 after it stops
    for k, (lx, lz, dst, ph) in enumerate(((-0.55, -0.35, H3_P, 0.0), (-0.6, -1.25, H2_P, 0.25))):
        tj = T - 27.2 - ph
        if tj < 0:
            continue
        a = local_to_world(Ms2, 'chassis', (lx, BED_Y, lz))
        b = onfloor(np.asarray(dst) + np.array([1.2, 0, 0]))
        u = np.clip(tj / 0.5, 0, 1)
        pos = lerp(a, b, u) + np.array([0, 1.4 * np.sin(np.pi * u) * 0.5, 0])
        pose = blend(add(RIFLE_PORT, H(thighL=-1.0, shinL=1.4, thighR=-0.6, shinR=1.2)), RIFLE_PORT, ease(u)) if u < 1 else \
            blend(add(RIFLE_PORT, H(thighL=-1.0, shinL=1.6, thighR=-0.9, shinR=1.5, spine=0.4)), RIFLE_PORT, ease((tj - 0.5) / 0.4))
        off = (0, -0.25 * (1 - ease((tj - 0.5) / 0.4)) if u >= 1 else 0, 0)
        m, _ = HUNTER.build(pose, pos, yaw_to(a, b), root_off=off); dyn.append(m)
    eye = np.array([-5.5, 1.9, 15.0]); tgt = lerp(np.array([3.5, 0.8, -8.0]), np.array([5.0, 1.0, -2.0]), ease((T - 25.2) / 1.0))
    sh = shake(T, 0.12 * np.exp(-max(T - CRASH_T, 0) * 5) * (T > CRASH_T))
    flash = 0.45 * np.exp(-(T - CRASH_T) * 30) if T >= CRASH_T else 0.0
    return dict(dyn=dyn, cam=dict(eye=eye + sh, target=tgt + sh * 0.5, fov=40), light=(np.array([3, 0, -2]), 30), flash=flash)

def static_trucks(T, hide_drum=False, door=0.0, pup_in=True, lift2=0.0, pitch2=0.0):
    p1, y1, r1, pi1, w1, _ = truck1_state(T)
    m1, Ms1 = truck_mesh(TRUCK1, p1, y1, w1, r1, pi1)
    p2, y2, pi2, w2 = truck2_state(T)
    m2, Ms2 = truck_mesh(TRUCK2, p2, y2, w2, 0.0, pi2 + pitch2, hide=('drum',) if hide_drum else (), lift=lift2)
    dyn = [m1, m2, cage_world(Ms2, y2, door)]
    CF = cage_frame(Ms2)
    if pup_in:
        pp = in_frame(CF, (0.0, 0.02, -0.05))
        m, _ = PUP.build(dict(DOG_SIT, head=(0.35, 0.3 * np.sin(T * 0.9), 0), earL=(0.7, 0, 0), earR=(0.7, 0, 0)), pp,
                         frame_yaw(CF, 0.0), scale=PUP_S)
        dyn.append(m)
    # smoldering hood on truck 1
    hood = local_to_world(Ms1, 'chassis', (0, 0.9, 1.9))
    for k in range(6):
        te = T - k * 0.25
        dyn += dust_cloud(hood, T - np.floor(te * 4) / 4, n=1, spread=0.4, rise=1.8, size=0.4,
                          seed=600 + int(np.floor(te * 4)), life=1.5, color=(0.86, 0.80, 0.82))
    return dyn, Ms1, Ms2, CF, (p2, y2)

def boss_exit(T):
    u = np.clip((T - 28.0) / 0.8, 0, 1)
    a = np.array([1.3, 0, -10.6])
    pos = lerp(a, BOSS_P, ease(u))
    return pos, u

def ranger_fight_pos(T):
    """ranger advances from the slope foot to RP during 28-30."""
    u = ease((T - 28.0) / 2.0)
    p = lerp(FIRE_POS, RP, u); p[1] = gy(p[0], p[2])
    return p, u

def h1_state(T):
    """truck-1 driver: stumbles out after the crash, rifle up."""
    u = np.clip((T - 28.5) / 1.5, 0, 1)
    pos = lerp(np.array([6.3, 0, 2.8]), H1_P, ease(u))
    pose = blend(STAGGER, RIFLE_PORT, ease(u))
    return pos, pose

def S07a(T):
    """28-30 low angle: the masked boss climbs out, bellowing."""
    t = T - 28.0
    dyn, Ms1, Ms2, CF, _ = static_trucks(T)
    bp, u = boss_exit(T)
    rp, _ = ranger_fight_pos(T)
    y = yaw_to(bp, rp)
    pose = blend(add(RIFLE_PORT, walk_cycle(T * 7, 0.7 * (1 - u))), YELL, ease((t - 0.7) / 0.3))
    m, Ms = hunter_mesh(BOSS, pose, bp, y); dyn.append(m)
    for pos_, ph in ((H3_P, 0), (H2_P, 1)):
        m, _ = HUNTER.build(add(RIFLE_PORT, H(head=(0, 0.3 * np.sin(T * 2 + ph), 0))), onfloor(pos_ + np.array([1.2 * (1 - ease(t / 1.0)), 0, 0])), y + 0.2 * ph)
        dyn.append(m)
    head = local_to_world(Ms, 'head', (0, 0.15, 0))
    fwd = np.array([np.sin(y), 0, np.cos(y)]); right = np.array([np.cos(y), 0, -np.sin(y)])
    eye = bp + fwd * 2.3 - right * 1.1 + np.array([0, 0.35, 0]) + handheld(T, 0.02)
    eye = eye + shake(T, 0.03 * ease((t - 0.7) / 0.2) * np.exp(-max(t - 0.9, 0) * 2))
    tgt = lerp(head + np.array([0, -0.4, 0]), head, ease(t / 1.0))
    return dict(dyn=dyn, cam=dict(eye=eye, target=tgt, fov=42), light=(bp, 15))

def S07b(T):
    """30-32 a hunter raises his rifle; the ranger shoots it out of his hands."""
    t = T - 30.0
    dyn, Ms1, Ms2, CF, _ = static_trucks(T)
    rp = RP.copy(); rp[1] = gy(rp[0], rp[2])
    h3 = onfloor(H3_P)
    yr = yaw_to(rp, h3); yh = yaw_to(h3, rp)
    rpose = blend(add(STAND, H(uarmR=(-0.6, 0, -0.2), farmR=-0.4)), AIM, ease((t - 0.45) / 0.3))
    if T > 30.9:
        k = np.exp(-(T - 30.9) * 10); rpose = add(rpose, H(uarmR=(-0.45 * k, 0, 0), farmR=-0.3 * k))
    m, MsR = ranger_mesh(T, rpose, rp, yr); dyn.append(m)
    hit = T > 30.95
    hpose = blend(RIFLE_PORT, RIFLE_AIM, ease((t - 0.2) / 0.5))
    if hit:
        hpose = blend(hpose, add(HANDS_UP, H(spine=-0.3, uarmR=(0.6, 0, 0))), ease((T - 30.95) / 0.25))
    m, MsH = hunter_mesh(HUNTER, hpose, h3, yh, gun=not hit); dyn.append(m)
    # other hunters + boss in background
    bp, _ = boss_exit(T)
    m, _ = hunter_mesh(BOSS, YELL, bp, yaw_to(bp, rp)); dyn.append(m)
    m, _ = HUNTER.build(RIFLE_PORT, onfloor(H2_P), yaw_to(H2_P, rp)); dyn.append(m)
    hp, hpo = h1_state(T); m, _ = HUNTER.build(hpo, onfloor(hp), yaw_to(hp, rp)); dyn.append(m)
    mid = (rp + h3) / 2
    fw = np.array([np.sin(yr), 0, np.cos(yr)]); rt = np.array([-np.cos(yr), 0, np.sin(yr)])
    eye = rp - fw * 1.6 + rt * 0.75 + np.array([0, 1.5, 0]) + handheld(T, 0.015)
    tgt = h3 + np.array([0, 1.15, 0]) + rt * 1.0
    flash = 0.0
    if 0 <= T - 30.9 < 2 / 24:
        fs, _ = muzzle(MsR, eye, 0.3, 3); dyn += fs; flash = 0.3
    if hit:
        hand = local_to_world(MsH, 'handR', (0, -0.1, 0)) if T < 30.97 else None
        a = T - 30.95
        hand0 = onfloor(H3_P) + np.array([0, 1.25, 0])
        d = -np.array([np.sin(yh), 0, np.cos(yh)])
        rpos = hand0 + d * 3.0 * a + np.array([0, 4.5 * a - 4.9 * a * a, 0])
        if rpos[1] < 0.1:
            rpos[1] = 0.1
        dyn.append(rifle_free(rpos, rot_x(a * 14) @ rot_y(yh)))
        dyn += sparks(hand0, a, eye, n=12, seed=9)
        if a < 2 / 24:
            dyn += flash_star(hand0, 0.35, eye, color=(1.0, 0.8, 0.4), seed=4)
    return dict(dyn=dyn, cam=dict(eye=eye, target=tgt, fov=30), light=(mid, 18), flash=flash)

def h2_charge(T):
    start = onfloor(H2_P)
    rp = RP.copy(); rp[1] = gy(rp[0], rp[2])
    d = rp - start; d[1] = 0; L = np.linalg.norm(d); d /= L
    stop = rp - d * 0.95
    u = np.clip((T - 32.0) / 1.35, 0, 1)
    pos = lerp(start, stop, u); pos[1] = gy(pos[0], pos[2])
    return pos, d, u, rp

def S07c(T):
    """32-34.5 a second hunter charges; ranger slips it and lays him out."""
    dyn, Ms1, Ms2, CF, _ = static_trucks(T)
    pos, d, u, rp = h2_charge(T)
    yh = float(np.arctan2(d[0], d[2]))
    hit_t = 33.42
    if T < hit_t:
        hp = add(run_cycle(T * 13.0, 1.0, 0.4), H(uarmR=(-0.9, 0, -0.2), farmR=-1.2))  # rifle clubbed up
        hpos = pos + np.array([0, run_bob(T * 13.0), 0])
        hoff = (0, 0, 0)
    else:
        a = T - hit_t
        k = np.clip(a / 0.5, 0, 1)
        hpos = pos + d * (-2.2 * eout(k)) + np.array([0, 0.5 * np.sin(np.pi * k) * (k < 1), 0])
        hp = blend(STAGGER, SUPINE, ease(k)); hoff = lerp((0, 0, 0), SUPINE_OFF, ease(k))
        if a > 0.5:
            hp = add(SUPINE, H(head=0.2 * np.sin(a * 3)))
    m, MsH = HUNTER.build(hp, hpos, yh, root_off=hoff); dyn.append(m)
    yr = yh + np.pi
    if T < 33.1:
        rpose = add(STAND, H(uarmR=(-0.6, 0, -0.2), farmR=-0.4))
    elif T < 33.32:
        rpose = blend(STAND, PUNCH_WIND, ease((T - 33.1) / 0.22))
    elif T < 33.5:
        rpose = blend(PUNCH_WIND, PUNCH_HIT, ease((T - 33.32) / 0.1))
    else:
        rpose = blend(PUNCH_HIT, add(STAND, H(uarmR=(-0.6, 0, -0.2), farmR=-0.4)), ease((T - 33.6) / 0.6))
    m, MsR = ranger_mesh(T, rpose, rp, yr); dyn.append(m)
    m, _ = hunter_mesh(BOSS, YELL, onfloor(BOSS_P), yaw_to(BOSS_P, rp)); dyn.append(m)
    side = np.array([d[2], 0, -d[0]])
    eye = rp - d * 1.5 + side * 6.3 + np.array([0, 1.2, 0]) + handheld(T, 0.02)
    tgt = rp - d * 1.4 + np.array([0, 0.95, 0])
    flash = 0.0
    if 0 <= T - hit_t < 3 / 24:
        fist = local_to_world(MsR, 'handL', (0, -0.05, 0))
        dyn += flash_star(fist, 0.45, eye, color=(1.0, 0.95, 0.75), spikes=9, seed=12)
        flash = 0.25
    sh = shake(T, 0.09 * np.exp(-max(T - hit_t, 0) * 8) * (T > hit_t))
    if T > 33.9:
        dyn += dust_cloud(hpos + np.array([0, 0.1, 0]), T - 33.9, n=7, spread=1.6, rise=0.4, size=0.24, seed=33, life=1.1)
    return dict(dyn=dyn, cam=dict(eye=eye + sh, target=tgt + sh * 0.6, fov=44), light=(rp, 14), flash=flash)

def dog_on_bed_state(T, CF):
    """Patch's sneak/jump/latch sequence in cage-frame coords, returns pose, world pos, yaw, door angle."""
    door = 0.0
    if T < 35.6:
        u = np.clip((T - 34.5) / 1.1, 0, 1)
        a = np.array([-3.4, 0, -17.6]); b = in_frame(CF, (0.35, 0, 3.0)); b[1] = 0
        pos = lerp(a, b, ease(u) * 0.9 + 0.1 * u); pos[1] = gy(pos[0], pos[2])
        pose = add(dog_trot(T * 11.0, 0.8), {'body': (0.0, 0, 0), 'neck': (-0.35, 0, 0), 'head': (0.45, 0, 0)})
        pose = {k: v for k, v in pose.items()}
        for kk in ('fL1', 'fR1', 'bL1', 'bR1'):
            pose[kk] = tuple(np.asarray(pose.get(kk, (0, 0, 0))) + np.array([0.0, 0, 0]))
        pos = pos + np.array([0, -0.12, 0])
        yaw = yaw_to(a, b) if u < 0.9 else lerp(yaw_to(a, b), frame_yaw(CF, np.pi), ease((u - 0.9) / 0.1))
        return pose, pos, yaw, door
    if T < 36.2:
        u = (T - 35.6) / 0.6
        a = in_frame(CF, (0.35, 0, 3.0)); a[1] = 0
        b = in_frame(CF, (0.0, 0, 0.85))
        pos = lerp(a, b, u) + np.array([0, 1.1 * np.sin(np.pi * u), 0])
        return DOG_LEAP, pos, frame_yaw(CF, np.pi), door
    b = in_frame(CF, (0.0, 0, 0.85))
    yaw = frame_yaw(CF, np.pi)
    if T < 36.6:
        return blend(DOG_LAND, DOG_STAND, ease((T - 36.2) / 0.4)), b, yaw, door
    if T < 37.0:  # lunge at the latch
        k = ease((T - 36.6) / 0.4)
        pose = blend(DOG_STAND, dict(DOG_STAND, neck=(0.2, -0.25, 0), head=(0.75, -0.2, 0), jaw=(0.45 * (1 - k), 0, 0),
                                     fL1=(-0.3, 0, 0), fR1=(-0.3, 0, 0)), k)
        return pose, b + (in_frame(CF, (0, 0, 0.75)) - b) * k * 0.6, yaw, door
    if T < 38.0:  # yank back: door swings open
        k = ease((T - 37.0) / 0.6)
        pose = dict(DOG_STAND, neck=(0.2 - 0.3 * k, -0.25 - 0.5 * k, 0), head=(0.75 - 0.4 * k, -0.2, 0),
                    bL1=(-0.4 * k, 0, 0), bR1=(-0.4 * k, 0, 0), fL1=(-0.5 * k, 0, 0), fR1=(-0.3 * k, 0, 0))
        door = 1.9 * eout((T - 37.0) / 0.8)
        return pose, b + (in_frame(CF, (0, 0, 0.75)) - b) * 0.6 * (1 - k), yaw, door
    door = 1.9
    k = ease((T - 38.6) / 0.5)
    pose = dog_tailwag(dict(DOG_STAND, neck=(0.0, -0.4 + 0.4 * k, 0), head=(0.35 * k, 0.0, 0)), T, 0.6, 12)
    return pose, b, yaw + 0.3 * k, door

def pup_state(T, CF):
    if T < 38.0:
        pp = in_frame(CF, (0.0, 0.02, -0.05))
        return dict(DOG_SIT, head=(0.35, 0.3 * np.sin(T * 0.9), 0), earL=(0.7, 0, 0), earR=(0.7, 0, 0)), pp, frame_yaw(CF, 0.0)
    if T < 38.9:
        u = (T - 38.0) / 0.9
        pp = in_frame(CF, lerp((0.0, 0.02, -0.05), (0.32, 0.02, 0.85), ease(u)))
        return dog_trot(T * 16, 1.0), pp, frame_yaw(CF, 0.25)
    pp = in_frame(CF, (0.32, 0.02, 0.85))
    k = ease((T - 38.9) / 0.4)
    pose = dog_tailwag(dict(DOG_STAND, neck=(-0.2, -0.6 * k, 0), head=(-0.3 * k, -0.3, 0), earL=(0.5, 0, 0), earR=(0.5, 0, 0)), T, 0.9, 20)
    return pose, pp, frame_yaw(CF, 0.25 - 1.2 * k)

def S08(T):
    """34.5-40 Patch circles behind truck 2, springs onto the bed and yanks the cage open."""
    dyn, Ms1, Ms2, CF, (p2, y2) = static_trucks(T, pup_in=False)
    dpose, dpos, dyaw, door = dog_on_bed_state(T, CF)
    dyn[2] = cage_world(Ms2, y2, door)
    m, MsD = DOG.build(dpose, dpos, dyaw); dyn.append(m)
    pp, ppos, pyaw = pup_state(T, CF)
    m, _ = PUP.build(pp, ppos, pyaw, scale=PUP_S); dyn.append(m)
    # unaware hunters in front
    m, _ = hunter_mesh(BOSS, add(RIFLE_PORT, H(head=(0, 0.2 * np.sin(T), 0))), onfloor(BOSS_P), yaw_to(BOSS_P, RP)); dyn.append(m)
    rear = in_frame(CF, (0, 0.2, 0.8))
    if T < 36.2:
        eye = np.array([-1.2, 0.55, -20.6]) + handheld(T, 0.02)
        tgt = lerp(np.array([-2.0, 0.4, -16.0]), rear + np.array([0, 0.3, 0]), ease((T - 34.6) / 1.4))
        fov = 44
    else:
        # tighter on the bed: cage door + dogs
        eye = in_frame(CF, (1.35, 0.95, 2.6)) + handheld(T, 0.01)
        tgt = in_frame(CF, (0.1, 0.35, 0.35))
        fov = 40
    return dict(dyn=dyn, cam=dict(eye=eye, target=tgt, fov=fov), light=(rear, 10))

def S09a(T):
    """40-42 shot / reverse: the boss swings his rifle at the dogs; the dogs bristle on the truck bed."""
    dyn, Ms1, Ms2, CF, (p2, y2) = static_trucks(T, door=1.9, pup_in=False)
    dpose, dpos, dyaw, _ = dog_on_bed_state(39.9, CF)
    bp = onfloor(BOSS_P)
    k = ease((T - 40.3) / 0.5)
    dyaw2 = lerp(dyaw, yaw_to(dpos, bp), k)
    m, _ = DOG.build(dict(DOG_GROWL, neck=(-0.3, 0, 0), head=(0.4, 0, 0)), dpos, dyaw2,
                     hide=('eye_open',) if k > 0.5 else ('eye_squint',))
    dyn.append(m)
    pp, ppos, pyaw = pup_state(39.9, CF)
    behind = dpos + (dpos - bp) / np.linalg.norm(dpos - bp) * 0.45 + np.array([0.25, 0, 0])
    behind[1] = dpos[1]
    m, _ = PUP.build(dict(DOG_CROUCH, earL=(1.2, 0, 0), earR=(1.2, 0, 0)), lerp(ppos, behind, k), lerp(pyaw, dyaw2, k), scale=PUP_S)
    dyn.append(m)
    target = dpos + np.array([0, 0.5, 0])
    yb = lerp(yaw_to(bp, RP), yaw_to(bp, target), ease((T - 40.0) / 0.6))
    pose = blend(RIFLE_PORT, RIFLE_AIM, ease((T - 40.2) / 0.5))
    m, Ms = hunter_mesh(BOSS, pose, bp, yb); dyn.append(m)
    if T < 41.1:
        # low frontal on the boss: the barrel comes round onto us
        dd = target - bp; dd[1] = 0; dd /= np.linalg.norm(dd)
        lat = np.array([dd[2], 0, -dd[0]])
        eye = bp + dd * 2.6 + lat * 0.5 + np.array([0, 1.05, 0]) + handheld(T, 0.01)
        tgt = bp + np.array([0, 1.45, 0]) + dd * 0.4
        return dict(dyn=dyn, cam=dict(eye=eye, target=tgt, fov=40), light=(bp, 10))
    # reverse: the dogs on the bed, from the boss's side but clear of the cage
    dd = dpos - bp; dd[1] = 0; dd /= np.linalg.norm(dd)
    lat = np.array([dd[2], 0, -dd[0]])
    if lat[0] > 0:
        lat = -lat
    eye = bp - dd * 1.3 + lat * 0.7 + np.array([0, 3.5, 0]) + handheld(T, 0.01)
    tgt = dpos + np.array([0, 0.35, 0]) - dd * 0.5
    return dict(dyn=dyn, cam=dict(eye=eye, target=tgt, fov=32), light=(dpos, 10))

def S09b(T):
    """42-43.2 ranger: one shot."""
    dyn, Ms1, Ms2, CF, _ = static_trucks(T, door=1.9, pup_in=False)
    rp = RP.copy(); rp[1] = gy(rp[0], rp[2])
    bp = onfloor(BOSS_P)
    yr = yaw_to(rp, bp + np.array([0.35, 0, 0]))
    pose = blend(add(STAND, H(uarmR=(-0.6, 0, -0.2), farmR=-0.4)), AIM2, ease((T - 42.0) / 0.35))
    if T > 42.6:
        k = np.exp(-(T - 42.6) * 9); pose = add(pose, H(uarmR=(-0.22 * k, 0, 0), farmR=-0.15 * k, head=(-0.1 * k, 0, 0)))
    m, Ms = ranger_mesh(T, pose, rp, yr); dyn.append(m)
    m, _ = hunter_mesh(BOSS, RIFLE_AIM, bp, yaw_to(bp, in_frame(CF, (0.1, 0.4, 0.9)))); dyn.append(m)
    fwd = np.array([np.sin(yr), 0, np.cos(yr)]); right = np.array([np.cos(yr), 0, -np.sin(yr)])
    head = local_to_world(Ms, 'head', (0, 0.12, 0))
    eye = rp + fwd * 2.3 - right * 1.7 + np.array([0, 1.5, 0]) + handheld(T, 0.01)
    tgt = head + np.array([0, -0.3, 0]) + fwd * 0.4
    flash = 0.0
    if 0 <= T - 42.6 < 2 / 24:
        fs, _ = muzzle(Ms, eye, 0.28, 8); dyn += fs; flash = 0.22
    sh = shake(T, 0.04 * np.exp(-max(T - 42.6, 0) * 9) * (T > 42.6))
    return dict(dyn=dyn, cam=dict(eye=eye + sh, target=tgt, fov=44), light=(rp, 8), flash=flash)

def S09c(T):
    """43.2-46 the boss's rifle bursts apart; he drops to his knees. 'Not anymore.'"""
    dyn, Ms1, Ms2, CF, _ = static_trucks(T, door=1.9, pup_in=False)
    bp = onfloor(BOSS_P)
    rp = RP.copy(); rp[1] = gy(rp[0], rp[2])
    hit_t = 43.25
    tgt_dog = in_frame(CF, (0.1, 0.4, 0.9))
    yb = yaw_to(bp, tgt_dog)
    a = T - hit_t
    if a < 0:
        pose, off, gun = RIFLE_AIM, (0, 0, 0), True
    elif a < 0.5:
        pose, off, gun = blend(RIFLE_AIM, STAGGER, ease(a / 0.2)), (0, 0, 0), False
        yb = yb + 0.6 * ease(a / 0.4)
    else:
        k = ease((a - 0.5) / 0.5)
        pose, off, gun = blend(STAGGER, KNEEL, k), lerp((0, 0, 0), KNEEL_OFF, k), False
        yb = yb + 0.6 + (yaw_to(bp, rp) - yb - 0.6) * k
    m, Ms = hunter_mesh(BOSS, pose, bp, yb, gun=gun, root_off=off); dyn.append(m)
    yr = yaw_to(rp, bp)
    walk = np.clip((T - 44.2) / 1.6, 0, 1)
    rpos = lerp(rp, rp + (bp - rp) * 0.45, ease(walk)); rpos[1] = gy(rpos[0], rpos[2])
    rpose = add(AIM, walk_cycle(T * 6.5, 0.6 * (0 < walk < 1))) if walk > 0 else AIM
    m, MsR = ranger_mesh(T, rpose, rpos, yr); dyn.append(m)
    # dogs watching from the bed
    dpose, dpos, dyaw, _ = dog_on_bed_state(39.9, CF)
    m, _ = DOG.build(dog_tailwag(DOG_STAND, T, 0.2, 6), dpos, dyaw + 0.4); dyn.append(m)
    pp, ppos, pyaw = pup_state(39.9, CF)
    m, _ = PUP.build(DOG_SIT, ppos, pyaw, scale=PUP_S); dyn.append(m)
    hand0 = bp + np.array([0, 1.3, 0]) + np.array([np.sin(yb), 0, np.cos(yb)]) * 0.5
    side = np.array([np.cos(yr), 0, -np.sin(yr)])
    if side[2] > 0:
        side = -side
    mid = lerp(bp, rp, 0.22)
    eye = mid + side * 7.5 + np.array([0, 1.3, 0]) + handheld(T, 0.02)
    tgt = mid + np.array([0, 0.95, 0])
    flash = 0.0
    if a >= 0:
        dyn += sparks(hand0, a, eye, n=16, seed=21, life=0.55)
        if a < 3 / 24:
            dyn += flash_star(hand0, 0.6, eye, color=(1.0, 0.75, 0.35), seed=22); flash = 0.3
        for i in range(4):   # rifle pieces
            rng = np.random.default_rng(40 + i)
            v0 = rng.normal(size=3) * 2.5 + np.array([0, 3.5, 0])
            pp = hand0 + v0 * a + np.array([0, -4.9 * a * a, 0])
            pp[1] = max(pp[1], 0.06)
            dyn.append(transform(box(0.05, 0.25 + 0.1 * i, 0.06, (0.30, 0.30, 0.36) if i % 2 else (0.42, 0.24, 0.14)), rot_x(a * 11 * (i + 1)), pp))
    sh = shake(T, 0.05 * np.exp(-max(a, 0) * 8) * (a > 0))
    return dict(dyn=dyn, cam=dict(eye=eye + sh, target=tgt, fov=44), light=(bp, 14), flash=flash)

def carry_pup(MsD, dyaw, T):
    jaw = local_to_world(MsD, 'jaw', (0, -0.02, 0.10))
    pos = jaw + np.array([0, -0.30, 0])
    m, _ = PUP.build(dict(PUP_DANGLE, fL1=(0.35 + 0.3 * np.sin(T * 9), 0, 0), bR1=(0.3 + 0.3 * np.sin(T * 9 + 1), 0, 0)),
                     pos, dyaw + np.pi / 2, scale=PUP_S, root_off=(0, -0.27, 0))
    return m

ESC_A = np.array([1.1, 0, -15.3])
ESC_B = np.array([-13.5, 0, -6.5])

def escape_pos(T, lag=0.0, side=0.0):
    u = np.clip((T - 47.2 - lag) / 4.8, 0, 1)
    p = lerp(ESC_A, ESC_B, u)
    d = ESC_B - ESC_A; d /= np.linalg.norm(d)
    perp = np.array([d[2], 0, -d[0]])
    p = p + perp * side
    p[1] = gy(p[0], p[2])
    return p, u, float(np.arctan2(d[0], d[2]))

EXPLODE_T = 48.8

def S10a(T):
    """46-47.2 Patch leaps off the truck bed with the pup in his jaws."""
    dyn, Ms1, Ms2, CF, (p2, y2) = static_trucks(T, door=1.9, pup_in=False)
    b = in_frame(CF, (0.0, 0, 0.85))
    land = ESC_A.copy(); land[1] = gy(land[0], land[2])
    u = np.clip((T - 46.0) / 0.6, 0, 1)
    if T < 46.0:
        pos, pose = b, DOG_STAND
    elif u < 1:
        pos = lerp(b, land, u) + np.array([0, 0.9 * np.sin(np.pi * u), 0]); pose = DOG_LEAP
    else:
        k = ease((T - 46.6) / 0.35); pos = land; pose = blend(DOG_LAND, dog_gallop(T * 14), k)
    dyaw = yaw_to(b, land) if T >= 46.0 else frame_yaw(CF, np.pi)
    m, MsD = DOG.build(dict(pose, jaw=(0.05, 0, 0)), pos, dyaw); dyn.append(m)
    dyn.append(carry_pup(MsD, dyaw, T))
    if T > 46.6:
        dyn += dust_cloud(land, T - 46.6, n=6, spread=0.7, rise=0.25, size=0.09, seed=46, life=0.8)
    # boss on knees, ranger guarding
    m, _ = hunter_mesh(BOSS, KNEEL, onfloor(BOSS_P), yaw_to(BOSS_P, RP), gun=False, root_off=KNEEL_OFF); dyn.append(m)
    eye = land + np.array([-2.6, 0.35, -2.8]) + handheld(T, 0.02)
    tgt = lerp(b, land, 0.55) + np.array([0, 0.5, 0])
    return dict(dyn=dyn, cam=dict(eye=eye, target=tgt, fov=46), light=(land, 10))

def S10b(T):
    """47.2-52 run for it: the ranger caps the fuel drum; the truck goes up behind them."""
    hide = T >= EXPLODE_T
    ea = T - EXPLODE_T
    lift = 0.45 * np.sin(np.pi * np.clip(ea / 0.7, 0, 1)) if ea > 0 else 0.0
    dyn, Ms1, Ms2, CF, (p2, y2) = static_trucks(T, hide_drum=hide, door=1.9, pup_in=False, lift2=lift,
                                                pitch2=-0.12 * np.sin(np.pi * np.clip(ea / 0.7, 0, 1)) if ea > 0 else 0)
    drum = local_to_world(Ms2, 'chassis', DRUM_LOCAL)
    # dog with pup
    dp, du, dyaw = escape_pos(T, 0.0, 0.0)
    m, MsD = DOG.build(dict(dog_gallop(T * 13.5), jaw=(0.05, 0, 0)), dp + np.array([0, 0.07 * abs(np.sin(T * 13.5)), 0]), dyaw)
    dyn.append(m); dyn.append(carry_pup(MsD, dyaw, T))
    # ranger: runs, turns and fires at the drum, runs again
    rp, ru, ryaw = escape_pos(T, 0.35, -1.6)
    if T < 48.25:
        rpose = add(run_cycle(T * 12.0), H(uarmR=(0.0, 0, -0.2))); yr = ryaw; rpos = rp + np.array([0, run_bob(T * 12.0), 0])
    elif T < 48.95:
        k = ease((T - 48.25) / 0.25)
        yr = lerp(ryaw, yaw_to(rp, drum), k)
        rpose = blend(run_cycle(T * 12.0), AIM, k); rpos = rp
        if T > 48.55:
            kk = np.exp(-(T - 48.55) * 10); rpose = add(rpose, H(uarmR=(-0.5 * kk, 0, 0)))
    else:
        k = ease((T - 48.95) / 0.3)
        yr = lerp(yaw_to(rp, drum), ryaw, k); rpose = blend(AIM, run_cycle(T * 12.0), k); rpos = rp + np.array([0, run_bob(T * 12.0) * k, 0])
    m, MsR = ranger_mesh(T, rpose, rpos, yr); dyn.append(m)
    # hunters near the truck get flattened
    for (hp0, rig, dir_, seed) in ((onfloor(BOSS_P), BOSS, 1, 0), (onfloor(H3_P), HUNTER, 1, 1), (onfloor(H1_P), HUNTER, -1, 2)):
        if ea < 0:
            pose = HANDS_UP if rig is BOSS else add(RIFLE_PORT, H(head=(0, 0.3 * np.sin(T + seed), 0)))
            if rig is BOSS:
                pose = KNEEL
            m, _ = rig.build(pose, hp0, yaw_to(hp0, rp), hide=('gun',) if rig is BOSS else (), root_off=KNEEL_OFF if rig is BOSS else (0, 0, 0))
        else:
            k = ease(ea / 0.45)
            away = hp0 - drum; away[1] = 0; away /= max(np.linalg.norm(away), 1e-6)
            pos = hp0 + away * 1.5 * eout(ea / 0.6)
            m, _ = rig.build(blend(STAGGER, DIVE, k), pos, float(np.arctan2(away[0], away[2])),
                             hide=('gun',) if rig is BOSS else (), root_off=lerp((0, 0, 0), DIVE_OFF, k))
        dyn.append(m)
    m, _ = HUNTER.build(add(SUPINE, H(head=0.2)), onfloor(h2_charge(40)[0] - h2_charge(40)[1] * 2.2), 0.0, root_off=SUPINE_OFF); dyn.append(m)
    # explosion + lingering fire
    if ea >= 0:
        dyn += explosion(drum, ea, seed=5, scale=1.35)
        if ea < 3 / 24:
            dyn += flash_star(drum, 3.0, (0, 0, 0), color=(1.0, 0.95, 0.7), spikes=11, seed=5)
        for i in range(5):
            ph = (T * 2.2 + i * 0.37) % 1.0
            fp = drum + np.array([np.sin(i * 2.1) * 0.5, ph * 1.8, np.cos(i * 1.7) * 0.5])
            if ea > 1.0:
                dyn.append(emissive(puff(fp, 0.45 * (1 - ph) + 0.1, (1.0, 0.70 - 0.3 * ph, 0.25), seed=900 + i + int(T * 6) % 3)))
    if 0 <= T - 48.55 < 2 / 24:
        dyn += muzzle(MsR, (0, 0, 0), 0.5, 13)[0]
    # dust from running feet
    for k in range(8):
        te = T - k / 8.0
        if te < 47.2:
            continue
        pe, _, _ = escape_pos(te, 0, 0)
        dyn += dust_cloud(pe + np.array([0, 0.1, 0]), T - te, n=1, spread=0.5, rise=0.5, size=0.3, seed=11000 + int(te * 8), life=1.0)
    # camera: ahead of the runners, looking back at the truck
    mid = (dp + rp) / 2
    d = ESC_B - ESC_A; d /= np.linalg.norm(d)
    eye = mid + d * 5.5 + np.array([0, 1.25, 0]) + np.array([d[2], 0, -d[0]]) * 1.4 + handheld(T, 0.03)
    tgt = lerp(mid + np.array([0, 0.9, 0]), drum, 0.32)
    flash = 0.0
    if ea >= 0:
        eye = eye + shake(T, 0.22 * np.exp(-ea * 3.0))
        flash = 0.65 * np.exp(-ea * 9)
    return dict(dyn=dyn, cam=dict(eye=eye, target=tgt, fov=50), light=(mid - d * 3, 22), flash=flash)

WALK_A = np.array([-41.0, 0, 2.0])
WALK_B = np.array([-47.5, 0, -6.0])

def S11(T):
    """52-60 sundown: three silhouettes on the rim. The pup licks Patch."""
    t = T - 52.0
    dyn = []
    d = WALK_B - WALK_A; L = np.linalg.norm(d); d /= L
    yaw = float(np.arctan2(d[0], d[2]))
    perp = np.array([d[2], 0, -d[0]])
    u = np.clip(t / 8.0, 0, 1)
    rp = WALK_A + d * L * u; rp[1] = gy(rp[0], rp[2])
    m, _ = ranger_mesh(T, add(walk_cycle(T * 5.2), H(head=(0, -0.35 * ease((t - 2.0) / 0.5) * (1 - ease((t - 5.0) / 0.6)), 0))),
                       rp + np.array([0, 0.03 * abs(np.sin(T * 5.2)), 0]), yaw, gun=False); dyn.append(m)
    dp = rp + perp * 1.1 - d * 0.2; dp[1] = gy(dp[0], dp[2])
    lick = 55.0 <= T <= 56.0
    dpose = dog_tailwag(dog_trot(T * 6.5, 0.7), T, 0.7 if T > 55 else 0.3, 10)
    if lick:
        dpose = dog_tailwag(dict(DOG_STAND, neck=(-0.1, 0.55, 0), head=(0.2, 0.3, 0)), T, 0.9, 16)
    m, _ = DOG.build(dpose, dp, yaw); dyn.append(m)
    pp = dp + perp * 0.75 + d * 0.4 * np.sin(T * 1.7); pp[1] = gy(pp[0], pp[2])
    if T < 55.0:
        ppose = dog_trot(T * 11.0, 1.0); py = yaw
        pp = pp + np.array([0, 0.05 * abs(np.sin(T * 11)), 0])
    elif lick:
        k = np.sin(np.pi * (T - 55.0) / 1.0)
        ppose = blend(DOG_STAND, PUP_REAR, min(1, k * 1.6)); py = yaw - 1.57
        pp = lerp(pp, dp + perp * 0.32 + d * 0.15, min(1, k * 1.8)); pp[1] = gy(pp[0], pp[2]) + 0.05 * k
    else:
        ppose = dog_tailwag(dog_trot(T * 11.0, 1.0), T, 1.0, 22); py = yaw
    m, _ = PUP.build(ppose, pp, py, scale=PUP_S); dyn.append(m)
    # camera: low behind them, rising and pulling away as they walk into the sun
    c0 = WALK_A - d * 4.2 + perp * 2.6 + np.array([0, gy(*(WALK_A - d * 4.2)[[0, 2]]) + 0.9, 0])
    eye = lerp(c0, c0 + np.array([0, 2.5, 0]) - d * 4.0 + perp * 3.0, ease(t / 8.0)) + handheld(T, 0.03)
    tgt = lerp(dp + np.array([0, 0.9, 0]), rp + d * 4 + np.array([0, 1.4, 0]), ease(t / 8.0))
    black = ease((T - 58.6) / 1.4)
    return dict(dyn=dyn, cam=dict(eye=eye, target=tgt, fov=40), light=(rp, 14), black=black)

SHOTS = [
    ('01_establish', 0.0, 6.0, S01),
    ('02_hear', 6.0, 10.0, S02),
    ('03_convoy', 10.0, 15.0, S03),
    ('04a_eye', 15.0, 17.0, S04a),
    ('04b_draw', 17.0, 19.0, S04b),
    ('05_slide', 19.0, 24.0, S05),
    ('06a_aim', 24.0, 25.2, S06a),
    ('06b_crash', 25.2, 28.0, S06b),
    ('07a_boss', 28.0, 30.0, S07a),
    ('07b_disarm', 30.0, 32.0, S07b),
    ('07c_punch', 32.0, 34.5, S07c),
    ('08_cage', 34.5, 40.0, S08),
    ('09a_threat', 40.0, 42.0, S09a),
    ('09b_shot', 42.0, 43.2, S09b),
    ('09c_kneel', 43.2, 46.0, S09c),
    ('10a_leap', 46.0, 47.2, S10a),
    ('10b_boom', 47.2, 52.0, S10b),
    ('11_sundown', 52.0, 60.0, S11),
]

OVERLAYS = [
    (1.0, 4.6, 'card', "RED MESA CANYON"),
    (56.9, 60.0, 'title', "PATCH"),
]

def frame(T):
    for name, t0, t1, fn in SHOTS:
        if t0 <= T < t1 or (name == SHOTS[-1][0] and T >= t1):
            out = fn(T)
            out.setdefault('flash', 0.0); out.setdefault('black', 0.0)
            out['shot'] = name
            e = np.asarray(out['cam']['eye'], 'f8').copy()
            e[1] = max(e[1], gy(e[0], e[2]) + 0.3)
            out['cam']['eye'] = e
            out['dyn'] = merge(out['dyn'])
            return out
    raise ValueError(T)

def subtitle_at(T):
    for t0, t1, s in SUBS:
        if t0 <= T < t1:
            return s
    return None
