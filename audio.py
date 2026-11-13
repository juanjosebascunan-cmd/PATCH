"""Procedural soundtrack: spaghetti-western score + Foley/SFX, synced to film.CUES.
Writes soundtrack.wav (48 kHz stereo, 60 s)."""
import numpy as np
from scipy.signal import lfilter, butter, sosfilt
from scipy.io import wavfile
import film

SR = 48000
DUR = film.DUR
N = int(SR * DUR)
rng = np.random.default_rng(7)

music = np.zeros((N, 2))
sfx = np.zeros((N, 2))

def t_arr(d):
    return np.arange(int(d * SR)) / SR

def midi(m):
    return 440.0 * 2 ** ((m - 69) / 12)

def place(buf, t0, sig, pan=0.0, gain=1.0):
    i0 = int(t0 * SR)
    if i0 >= N:
        return
    if i0 < 0:
        sig = sig[-i0:]; i0 = 0
    n = min(len(sig), N - i0)
    l = np.cos((pan + 1) * np.pi / 4); r = np.sin((pan + 1) * np.pi / 4)
    buf[i0:i0 + n, 0] += sig[:n] * gain * l * 1.414
    buf[i0:i0 + n, 1] += sig[:n] * gain * r * 1.414

def lp(x, fc, order=2):
    return sosfilt(butter(order, min(fc, SR * 0.45), 'low', fs=SR, output='sos'), x)

def hp(x, fc, order=2):
    return sosfilt(butter(order, fc, 'high', fs=SR, output='sos'), x)

def bp(x, lo, hi, order=2):
    return sosfilt(butter(order, [lo, min(hi, SR * 0.45)], 'band', fs=SR, output='sos'), x)

def env(n, a=0.005, d=0.3, curve=4.0):
    t = np.arange(n) / SR
    e = np.minimum(t / max(a, 1e-4), 1.0) * np.exp(-np.maximum(t - a, 0) * curve / max(d, 1e-3))
    return e

def noise(d):
    return rng.standard_normal(int(d * SR))

# ---------------------------------------------------------------- instruments
def pluck(freq, dur=2.0, bright=0.6, decay=0.996):
    """Karplus-Strong via IIR (fast)."""
    n = int(dur * SR)
    L = max(2, int(SR / freq))
    exc = np.zeros(n)
    burst = rng.uniform(-1, 1, L)
    burst = lp(burst, 1200 + 5000 * bright)
    exc[:L] = burst
    a = np.zeros(L + 2); a[0] = 1.0; a[L] = -decay * 0.5; a[L + 1] = -decay * 0.5
    y = lfilter([1.0], a, exc)
    y *= np.exp(-np.arange(n) / SR * 0.8)
    return y / (np.max(np.abs(y)) + 1e-9)

def twang(freq, dur=1.6):
    """baritone-guitar 'twang': pluck + slight pitch dive + spring reverb flavour."""
    y = pluck(freq, dur, bright=0.85, decay=0.9975)
    y = y + 0.35 * np.tanh(3 * y)
    return y / (np.max(np.abs(y)) + 1e-9)

def whistle(freq, dur, vib=5.5):
    t = t_arr(dur)
    v = 1 + 0.012 * np.sin(2 * np.pi * vib * t) * np.minimum(t / 0.3, 1)
    ph = 2 * np.pi * np.cumsum(freq * v) / SR
    gl = np.minimum(t / 0.06, 1.0)
    y = np.sin(ph) + 0.08 * np.sin(2 * ph)
    breath = bp(noise(dur), freq * 0.9, freq * 1.3) * 0.05
    e = gl * np.minimum(1, (dur - t) / 0.12)
    return (y + breath) * np.clip(e, 0, 1)

def pad(freqs, dur, cutoff=900):
    t = t_arr(dur)
    y = np.zeros_like(t)
    for f in freqs:
        for det in (-0.004, 0.0, 0.005):
            ph = (f * (1 + det) * t + rng.random()) % 1.0
            y += 2 * ph - 1
    y = lp(y, cutoff, 2)
    e = np.minimum(t / (dur * 0.35), 1) * np.minimum(1, (dur - t) / (dur * 0.3))
    return y * e / (len(freqs) * 3)

def kick(d=0.5):
    t = t_arr(d)
    f = 45 + 110 * np.exp(-t * 28)
    y = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 7)
    return y + 0.15 * lp(noise(d), 2000) * np.exp(-t * 60)

def taiko(d=1.2, f0=72):
    t = t_arr(d)
    f = f0 + 40 * np.exp(-t * 18)
    y = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 4.0)
    y += 0.35 * lp(noise(d), 900) * np.exp(-t * 25)
    return y

def snare(d=0.35):
    t = t_arr(d)
    y = bp(noise(d), 900, 7000) * np.exp(-t * 18) * 0.7
    y += np.sin(2 * np.pi * 190 * t) * np.exp(-t * 25) * 0.5
    return y

def hat(d=0.08):
    t = t_arr(d)
    return hp(noise(d), 7000) * np.exp(-t * 70) * 0.5

def rim(d=0.06):
    t = t_arr(d)
    return (np.sin(2 * np.pi * 1700 * t) * 0.6 + hp(noise(d), 3000) * 0.4) * np.exp(-t * 90)

def bass(freq, dur):
    t = t_arr(dur)
    y = np.sin(2 * np.pi * freq * t) + 0.35 * np.sin(4 * np.pi * freq * t) + 0.12 * np.sign(np.sin(2 * np.pi * freq * t))
    y = lp(y, 500)
    return y * env(len(t), 0.01, dur * 0.8, 3.0)

def bell(freq, dur=2.5):
    t = t_arr(dur)
    y = sum(a * np.sin(2 * np.pi * freq * r * t) * np.exp(-t * k) for r, a, k in ((1, 1, 1.5), (2.76, 0.4, 3), (5.4, 0.2, 5), (8.9, 0.1, 8)))
    return y * np.minimum(t / 0.002, 1)

# ---------------------------------------------------------------- the score
D2, D3, D4, D5 = 38, 50, 62, 74
SCALE = [0, 1, 4, 5, 7, 8, 10]          # D phrygian dominant: desert/western colour

def deg(root, d):
    o, k = divmod(d, 7)
    return root + 12 * o + SCALE[k]

# intro 0-10: wind-bed drone, slow arpeggio + whistle theme
place(music, 0.0, pad([midi(D2), midi(D3 + 7)], 10.5, 500) * 0.55)
arp = [0, 4, 7, 9, 7, 4, 2, 4]
for i in range(16):
    tt = 0.6 + i * 0.6
    if tt > 9.6:
        break
    place(music, tt, pluck(midi(deg(D3, arp[i % 8])), 2.2, 0.5) * 0.28, pan=-0.35)
theme = [(1.8, 7, 1.0), (2.9, 6, 0.5), (3.4, 4, 0.5), (3.9, 5, 1.6), (6.0, 4, 0.5), (6.5, 2, 0.5), (7.0, 1, 0.8), (7.8, 0, 1.9)]
for tt, d_, dur in theme:
    place(music, tt, whistle(midi(deg(D4 + 12, d_)), dur) * 0.16, pan=0.25)
place(music, 0.2, bell(midi(D5 + 7), 4.0) * 0.06, pan=0.5)

# tension 10-19: pulses, pad swell, drum roll into the slide
for i in range(14):
    tt = 10.0 + i * 0.6
    place(music, tt, taiko(1.0, 62) * (0.30 + 0.03 * i), pan=0.0)
place(music, 10.0, pad([midi(D2), midi(D2 + 1), midi(D3 + 7)], 9.2, 650) * 0.55)
for i, tt in enumerate(np.linspace(17.0, 19.0, 26)):
    place(music, tt, snare(0.2) * (0.05 + 0.25 * i / 26), pan=0.1)
place(music, 17.0, pad([midi(D3), midi(D3 + 4), midi(D3 + 7), midi(D4 + 1)], 2.1, 1800) * 0.45)
place(music, 17.2, twang(midi(D2 + 12), 2.0) * 0.3, pan=-0.2)

# action A 19-34: 140 bpm groove, bass ostinato, twang stabs
BPM = 140.0; beat = 60.0 / BPM

def groove(t0, t1, intensity=1.0, root=D2):
    nb = int((t1 - t0) / beat)
    ost = [0, 0, 2, 0, 3, 2, 1, 0]
    for b in range(nb):
        tt = t0 + b * beat
        if b % 4 in (0, 2) or (b % 8 == 7):
            place(music, tt, kick() * 0.55 * intensity)
        if b % 4 in (1, 3):
            place(music, tt, snare() * 0.32 * intensity, pan=0.1)
        if b % 2 == 0:
            place(music, tt, taiko(0.9, 80) * 0.25 * intensity, pan=-0.3)
        for h in range(2):
            place(music, tt + h * beat / 2, hat() * 0.18 * intensity, pan=0.4)
        place(music, tt, bass(midi(deg(root + 12, ost[b % 8])), beat * 0.95) * 0.42 * intensity)
        if b % 8 == 0:
            place(music, tt, twang(midi(deg(D3, [0, 4, 3, 1][(b // 8) % 4])), 1.8) * 0.30 * intensity, pan=-0.25)

groove(19.0, 24.6, 0.9)
groove(25.9, 34.4, 1.0)
lead = [(19.86, 7, 0.8), (21.57, 8, 0.4), (22.0, 7, 0.4), (22.43, 5, 1.2), (28.6, 4, 0.4), (29.0, 5, 0.4), (29.4, 7, 1.0), (31.5, 8, 0.6), (32.2, 9, 1.2)]
for tt, d_, dur in lead:
    place(music, tt, whistle(midi(deg(D4 + 12, d_)), dur) * 0.12, pan=0.3)
# stinger hit on the crash
place(music, film.CRASH_T, taiko(2.0, 55) * 0.8)
place(music, film.CRASH_T, pad([midi(D3), midi(D3 + 1), midi(D3 + 6)], 1.4, 2500) * 0.5)

# sneak 34.5-40: pizzicato + ticking, low pad
place(music, 34.4, pad([midi(D2), midi(D3 + 3)], 5.8, 450) * 0.5)
for i in range(int(5.5 / (beat / 2))):
    tt = 34.5 + i * beat / 2
    place(music, tt, rim() * 0.20, pan=0.5 if i % 2 else -0.5)
    if i % 2 == 0:
        place(music, tt, pluck(midi(deg(D3, [0, 1, 0, 4, 3, 1, 0, -1][(i // 2) % 8])), 0.5, 0.3, 0.99) * 0.26, pan=-0.2)
place(music, 38.0, bell(midi(D5 + 4), 3.0) * 0.07, pan=0.4)    # the cage pops: hopeful chime

# standoff 40-43.2: drone + heartbeat, silence before the shot
place(music, 40.0, pad([midi(D2), midi(D2 + 1)], 2.7, 380) * 0.6)
for tt in (40.2, 41.0, 41.8):
    place(music, tt, kick(0.5) * 0.35); place(music, tt + 0.22, kick(0.4) * 0.22)
place(music, 42.6, taiko(2.4, 50) * 0.75)
place(music, 42.6, pad([midi(D3), midi(D3 + 4), midi(D3 + 7)], 3.2, 2200) * 0.40)
place(music, 44.3, twang(midi(D3), 2.5) * 0.35, pan=-0.2)

# escape 46-52: groove returns hotter, cut by the explosion
groove(46.0, film.EXPLODE_T - 0.05, 1.05)
groove(film.EXPLODE_T + 0.9, 52.0, 0.8)
for tt, d_, dur in [(46.0, 7, 0.5), (46.45, 9, 0.5), (46.9, 10, 0.9), (50.0, 9, 0.5), (50.45, 8, 0.5), (50.9, 7, 1.0)]:
    place(music, tt, whistle(midi(deg(D4 + 12, d_)), dur) * 0.12, pan=0.3)

# resolution 52-60: D major, theme returns warm, fade
place(music, 52.0, pad([midi(D2), midi(D3 + 4), midi(D3 + 7), midi(D4 + 2)], 8.0, 900) * 0.6)
maj = [0, 4, 7, 11, 14, 11, 7, 4]
for i in range(13):
    tt = 52.2 + i * 0.6
    place(music, tt, pluck(midi(D3 + maj[i % 8]), 2.4, 0.45) * 0.26, pan=-0.35)
for tt, semi, dur in [(53.0, 19, 1.0), (54.1, 17, 0.5), (54.6, 16, 0.5), (55.1, 14, 1.6), (57.0, 12, 0.5), (57.5, 11, 0.5), (58.0, 14, 1.8)]:
    place(music, tt, whistle(midi(D4 + semi), dur) * 0.15, pan=0.25)
place(music, 57.0, bell(midi(D5 + 2), 3.0) * 0.07, pan=0.5)

# ---------------------------------------------------------------- SFX
def echo(sig, delays=((0.31, 0.32), (0.66, 0.18), (1.12, 0.09)), lpf=2500):
    """canyon slap-back echoes"""
    tail = max(d for d, g in delays)
    out = np.concatenate([sig, np.zeros(int((tail + 0.1) * SR))])
    wet = lp(sig, lpf)
    for d, g in delays:
        i = int(d * SR)
        out[i:i + len(wet)] += wet * g
    return out

def gunshot():
    d = 0.6; t = t_arr(d)
    crack = hp(noise(d), 1500) * np.exp(-t * 60)
    body = lp(noise(d), 1800) * np.exp(-t * 18)
    thump = np.sin(2 * np.pi * (60 + 80 * np.exp(-t * 40)) * t) * np.exp(-t * 14)
    return echo(np.tanh(2.2 * (crack * 0.9 + body * 0.8 + thump * 1.0)))

def ricochet():
    d = 0.7; t = t_arr(d)
    f = 3400 * np.exp(-t * 1.6) + 900
    y = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 5) * 0.5
    return y + hp(noise(d), 4000) * np.exp(-t * 40) * 0.4

def engine(d, f0, f1, load=1.0):
    t = t_arr(d)
    f = np.linspace(f0, f1, len(t)) * (1 + 0.03 * np.sin(2 * np.pi * 7 * t))
    ph = np.cumsum(f) / SR
    y = 0.6 * (2 * (ph % 1) - 1) + 0.4 * np.sin(2 * np.pi * 2 * ph) + 0.25 * np.sign(np.sin(2 * np.pi * 0.5 * ph))
    y = lp(y, 700 + 500 * load) + 0.3 * lp(noise(d), 400) * load
    return np.tanh(1.5 * y)

def screech(d, fc=2600):
    t = t_arr(d)
    y = bp(noise(d), fc * 0.85, fc * 1.15, 3) * 3
    y += 0.3 * np.sin(2 * np.pi * np.cumsum(fc * (1 + 0.04 * np.sin(2 * np.pi * 11 * t))) / SR)
    return y * np.minimum(t / 0.05, 1) * np.minimum(1, (d - t) / 0.2)

def crash():
    d = 2.2; t = t_arr(d)
    y = np.tanh(3 * lp(noise(d), 1500) * np.exp(-t * 4)) * 0.8
    y += np.sin(2 * np.pi * (45 + 60 * np.exp(-t * 20)) * t) * np.exp(-t * 5) * 1.2
    for k in range(9):   # metal clanks
        f = rng.uniform(300, 2200)
        st = rng.uniform(0.0, 0.9)
        tt = np.clip(t - st, 0, None)
        y += (t > st) * np.sin(2 * np.pi * f * tt) * np.exp(-tt * rng.uniform(8, 20)) * 0.25
    y += hp(noise(d), 3000) * np.exp(-t * 3) * 0.15   # glass/debris hiss
    return echo(y, ((0.4, 0.25), (0.9, 0.12)))

def explosion():
    d = 4.0; t = t_arr(d)
    boom = np.sin(2 * np.pi * np.cumsum(30 + 70 * np.exp(-t * 6)) / SR) * np.exp(-t * 1.6) * 1.4
    roar = lp(noise(d), 600) * np.exp(-t * 1.1) * 1.4
    crack = hp(noise(d), 2000) * np.exp(-t * 14) * 0.6
    debris = np.zeros_like(t)
    for k in range(40):
        st = rng.uniform(0.2, 3.0); tt = np.clip(t - st, 0, None)
        debris += (t > st) * bp(noise(d), 1500, 6000)[:len(t)] * np.exp(-tt * 60) * rng.uniform(0.05, 0.25)
    y = np.tanh(1.6 * (boom + roar + crack)) + debris
    return echo(y, ((0.45, 0.35), (1.0, 0.2), (1.7, 0.1)), 900)

def punch():
    d = 0.4; t = t_arr(d)
    y = np.sin(2 * np.pi * (80 + 120 * np.exp(-t * 50)) * t) * np.exp(-t * 18)
    y += lp(noise(d), 2500) * np.exp(-t * 35) * 0.8
    return np.tanh(2 * y)

def thud(d=0.6, f=55):
    t = t_arr(d)
    return np.sin(2 * np.pi * (f + 40 * np.exp(-t * 30)) * t) * np.exp(-t * 10) + lp(noise(d), 800) * np.exp(-t * 14) * 0.5

def whoosh(d=0.5):
    t = t_arr(d)
    y = noise(d)
    fc = 400 + 2500 * np.sin(np.pi * t / d)
    out = np.zeros_like(y)
    for i in range(0, len(y), 2400):   # piecewise sweep
        seg = y[i:i + 2400]
        out[i:i + len(seg)] = bp(seg, max(100, fc[i] * 0.7), fc[i] * 1.3)
    return out * np.sin(np.pi * t / d) ** 2 * 0.8

def metal_click():
    d = 0.25; t = t_arr(d)
    return sum(np.sin(2 * np.pi * f * t) * np.exp(-t * k) for f, k in ((2400, 40), (3900, 60), (1300, 30))) * 0.4

def creak(d=0.8):
    t = t_arr(d)
    f = 260 + 140 * t / d + 20 * np.sin(2 * np.pi * 9 * t)
    pulses = (np.sin(2 * np.pi * 38 * t) > 0.6).astype(float)
    y = np.sin(2 * np.pi * np.cumsum(f) / SR) * (0.4 + 0.6 * pulses)
    return bp(y, 200, 2500) * np.sin(np.pi * t / d) * 0.6

def yip(f0=950, f1=1450, d=0.22):
    t = t_arr(d)
    f = f0 + (f1 - f0) * np.sin(np.pi * t / d * 0.7)
    ph = 2 * np.pi * np.cumsum(f) / SR
    y = np.sin(ph) + 0.4 * np.sin(2 * ph) + 0.2 * np.sin(3 * ph)
    return y * np.sin(np.pi * t / d) ** 1.5 * 0.5

def whimper(d=1.3):
    t = t_arr(d)
    f = 780 + 160 * np.sin(2 * np.pi * 2.2 * t) * np.exp(-t) + 60 * np.sin(2 * np.pi * 13 * t)
    ph = 2 * np.pi * np.cumsum(f) / SR
    y = (np.sin(ph) + 0.3 * np.sin(2 * ph)) * (0.5 + 0.5 * np.sin(2 * np.pi * 3.1 * t) ** 2)
    return y * np.minimum(t / 0.1, 1) * np.minimum(1, (d - t) / 0.3) * 0.35

def growl(d=1.6):
    t = t_arr(d)
    am = 0.55 + 0.45 * np.sin(2 * np.pi * 28 * t + 2 * np.sin(2 * np.pi * 3 * t))
    y = bp(noise(d), 120, 520, 3) * am * 3 + 0.4 * np.sin(2 * np.pi * np.cumsum(90 + 10 * np.sin(2 * np.pi * 2 * t)) / SR) * am
    return np.tanh(y) * np.minimum(t / 0.15, 1) * np.minimum(1, (d - t) / 0.25) * 0.6

def wind(d):
    y = noise(d)
    t = t_arr(d)
    g = 0.5 + 0.5 * np.sin(2 * np.pi * 0.11 * t) * np.sin(2 * np.pi * 0.047 * t + 1)
    lo = bp(y, 150, 900) * (0.6 + 0.4 * g)
    hi = bp(np.roll(y, 1000), 1800, 4500) * 0.25 * g ** 2
    return lo + hi

# ambience
place(sfx, 0.0, wind(DUR) * 0.11)
place(sfx, 0.0, hp(noise(DUR), 6000) * 0.006, pan=0.6)        # faint desert insects/hiss

# engines: approaching convoy, doppler pass, then idling
def truck_engine_bus():
    t = np.arange(N) / SR
    y = np.zeros(N)
    for which, offset in ((1, 0.0), (2, 1.53)):
        pass_t = 12.8 + offset
        gain = np.where(t < 10, np.clip((t - 7.5) / 2.5, 0, 1) * 0.15, 0)
        gain = gain + np.where((t >= 10) & (t < 15), 0.15 + 0.85 * np.exp(-((t - pass_t) / 0.9) ** 2), 0)
        gain = gain + np.where((t >= 15) & (t < 19), 0.10, 0)
        gain = gain + np.where((t >= 19) & (t < 24), 0.12 + 0.05 * (t - 19), 0)
        if which == 1:
            gain = gain + np.where((t >= 24) & (t < film.CRASH_T), 0.45, 0)
        else:
            gain = gain + np.where((t >= 24) & (t < 27), 0.40 * np.clip((27 - t) / 1.5, 0.25, 1), 0)
            gain = gain + np.where((t >= 27) & (t < film.EXPLODE_T), 0.07, 0)   # idling
        dop = 1 + 0.06 * np.tanh(-(t - pass_t) * 1.2) * ((t > 10) & (t < 15))
        base = 52 if which == 1 else 47
        f = base * dop * np.where(t > 27, 0.75, 1.0)
        ph = np.cumsum(f) / SR
        e = 0.6 * (2 * (ph % 1) - 1) + 0.4 * np.sin(4 * np.pi * ph) + 0.25 * np.sign(np.sin(np.pi * ph))
        y += lp(e, 900) * gain * 0.5
    y += lp(noise(DUR), 350) * 0.15 * np.exp(-((t - 13.5) / 1.6) ** 2)
    return np.tanh(1.4 * y)

eng = truck_engine_bus()
sfx[:, 0] += eng * 0.75; sfx[:, 1] += eng * 0.75

for c in film.CUES:
    t0, k = c['t'], c['kind']
    if k == 'gunshot':
        place(sfx, t0, gunshot(), gain=0.95)
    elif k == 'tire_pop':
        place(sfx, t0, punch() * 0.8 + 0, gain=0.6, pan=0.3)
    elif k == 'skid':
        place(sfx, t0, screech(c['dur'], 2300), gain=0.18, pan=0.3)
    elif k == 'brake':
        place(sfx, t0, screech(c['dur'], 3100), gain=0.14, pan=-0.3)
    elif k == 'crash':
        place(sfx, t0, crash(), gain=0.9, pan=0.2)
    elif k == 'ricochet':
        place(sfx, t0, ricochet(), gain=0.4, pan=0.4)
    elif k == 'punch':
        place(sfx, t0, punch(), gain=0.9)
    elif k == 'body_fall':
        place(sfx, t0, thud(0.7, 50), gain=0.8, pan=0.3)
    elif k == 'dog_leap':
        place(sfx, t0 - 0.1, whoosh(0.5), gain=0.35)
    elif k == 'thud_small':
        place(sfx, t0, thud(0.4, 90), gain=0.45)
    elif k == 'latch':
        place(sfx, t0, metal_click(), gain=0.6, pan=0.2)
    elif k == 'cage_creak':
        place(sfx, t0, creak(0.8), gain=0.5, pan=0.2)
    elif k == 'pup_yip':
        place(sfx, t0, yip(), gain=0.55, pan=0.15); place(sfx, t0 + 0.28, yip(1050, 1600, 0.18), gain=0.4, pan=0.15)
    elif k == 'whimper':
        place(sfx, t0, whimper(1.3), gain=0.5, pan=0.3)
    elif k == 'growl':
        place(sfx, t0, growl(c.get('dur', 1.2)), gain=0.9)
    elif k == 'metal_clatter':
        for i in range(4):
            place(sfx, t0 + 0.35 + i * 0.13, metal_click() * (0.7 - 0.12 * i), gain=0.5, pan=-0.2)
    elif k == 'explosion':
        place(sfx, t0, explosion(), gain=1.0)

# footsteps for the runs + boots on scree in the slide
for tt in np.arange(19.3, 23.4, 0.07):
    place(sfx, tt, bp(noise(0.06), 800, 5000) * env(int(0.06 * SR), 0.002, 0.04) * 0.10, pan=-0.2)
for tt in np.arange(32.0, 33.4, 0.23):
    place(sfx, tt, thud(0.15, 120) * 0.25, pan=0.3)
for tt in np.arange(47.3, 52.0, 0.25):
    place(sfx, tt, thud(0.12, 140) * 0.18, pan=-0.1)

# ---------------------------------------------------------------- mix
# duck the score under big SFX moments
t = np.arange(N) / SR
duck = np.ones(N)
for tt, depth, rel in ((film.T_SHOT, 0.45, 0.8), (film.CRASH_T, 0.4, 1.0), (30.9, 0.5, 0.6), (42.6, 0.35, 1.2), (film.EXPLODE_T, 0.15, 1.6)):
    duck *= 1 - (1 - depth) * np.exp(-np.maximum(t - tt, 0) / rel) * (t >= tt - 0.02)
music *= duck[:, None]
mix = music * 0.85 + sfx
# fade in / out
fade = np.minimum(t / 0.6, 1) * np.clip((DUR - t) / 2.5, 0, 1)
mix *= fade[:, None]
# soft master limiter
mix = np.tanh(mix * 1.1) * 0.89
peak = np.max(np.abs(mix))
mix = mix / peak * 0.70
wavfile.write('soundtrack.wav', SR, (mix * 32767).astype(np.int16))
print('soundtrack.wav written, peak', peak)
