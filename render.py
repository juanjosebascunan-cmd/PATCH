"""Render driver.
  python3 render.py sheet  <out.png> [W] [per_shot]   -> preview frame sheet
  python3 render.py shots  <outdir>  [W] [shot,...]   -> one mp4 per shot (raw pipe to ffmpeg)
"""
import sys, os, subprocess, time
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from engine import Renderer, FPS
from world import static_world
import film

ASPECT = float(os.environ.get('ASPECT', '0.75'))  # H/W
FONT_SUB = '/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed-Bold.ttf'
FONT_TITLE = '/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf'

def make_renderer(W, H):
    r = Renderer(W, H, shadow_res=4096 if W >= 2000 else 2048)
    r.upload('world', film_world())
    return r

_WORLD = None
def film_world():
    global _WORLD
    if _WORLD is None:
        _WORLD = static_world()
    return _WORLD

def draw_text_outlined(d, xy, text, font, fill, outline, ow, anchor):
    d.text(xy, text, font=font, fill=fill, anchor=anchor, stroke_width=ow, stroke_fill=outline)

def overlay(img, T):
    W, H = img.size
    sub = film.subtitle_at(T)
    over = [(a, b, k, s) for a, b, k, s in film.OVERLAYS if a <= T < b]
    if not sub and not over:
        return img
    d = ImageDraw.Draw(img)
    if sub:
        f = ImageFont.truetype(FONT_SUB, int(H * 0.040))
        if ':' in sub and sub.split(':')[0].isupper():
            who, line = sub.split(':', 1)
            text = line.strip()
            fw = ImageFont.truetype(FONT_SUB, int(H * 0.026))
            draw_text_outlined(d, (W / 2, H * 0.885), who.strip(), fw, (255, 196, 120), (20, 10, 24), max(2, int(H * 0.004)), 'ms')
        else:
            text = sub
            f = ImageFont.truetype(FONT_SUB.replace('Bold', 'BoldOblique'), int(H * 0.036))
        draw_text_outlined(d, (W / 2, H * 0.94), text, f, (255, 250, 240), (20, 10, 24), max(3, int(H * 0.0055)), 'ms')
    for a, b, kind, s in over:
        fade = min(1.0, (T - a) / 0.6, (b - T) / 0.6) if kind == 'card' else min(1.0, (T - a) / 0.5)
        alpha = int(255 * max(0, fade))
        layer = Image.new('RGBA', img.size, (0, 0, 0, 0))
        ld = ImageDraw.Draw(layer)
        if kind == 'card':
            f = ImageFont.truetype(FONT_SUB, int(H * 0.030))
            spaced = ' '.join(s)
            ld.text((W * 0.06, H * 0.90), spaced, font=f, fill=(255, 236, 210, alpha), anchor='ls',
                    stroke_width=max(2, int(H * 0.004)), stroke_fill=(30, 12, 30, alpha))
            ld.rectangle([W * 0.06, H * 0.91, W * 0.06 + W * 0.20 * fade, H * 0.914], fill=(255, 150, 70, alpha))
        else:
            f = ImageFont.truetype(FONT_TITLE, int(H * 0.16))
            ld.text((W / 2, H * 0.47), s, font=f, fill=(255, 178, 90, alpha), anchor='mm',
                    stroke_width=int(H * 0.008), stroke_fill=(24, 10, 26, alpha))
            f2 = ImageFont.truetype(FONT_SUB, int(H * 0.036))
            ld.text((W / 2, H * 0.61), 'hecho por @promptmedia', font=f2, fill=(255, 226, 200, alpha), anchor='mm',
                    stroke_width=max(2, int(H * 0.004)), stroke_fill=(24, 10, 26, alpha))
            f3 = ImageFont.truetype(FONT_SUB, int(H * 0.026))
            ld.text((W / 2, H * 0.665), 'código abierto · hecho con Claude', font=f3, fill=(240, 200, 220, alpha), anchor='mm',
                    stroke_width=max(2, int(H * 0.0035)), stroke_fill=(24, 10, 26, alpha))
        img = Image.alpha_composite(img.convert('RGBA'), layer).convert('RGB')
        d = ImageDraw.Draw(img)
    return img

def render_frame(r, T, W, H):
    fr = film.frame(T)
    r.upload('dyn', fr['dyn'], dynamic=True)
    data = r.render(['world', 'dyn'], fr['cam'], frame_t=T, light_center=fr['light'][0],
                    light_span=fr['light'][1], flash=fr['flash'], black=fr['black'])
    img = Image.frombytes('RGB', (W, H), data).transpose(Image.FLIP_TOP_BOTTOM)
    return overlay(img, T), fr['shot']

def times_sheet(out, times, W=960, cols=3):
    H = int(round(W * ASPECT / 2)) * 2
    r = make_renderer(W, H)
    tiles = []
    for T in times:
        img, name = render_frame(r, T, W, H)
        tiles.append((img, f"{name}  {T:05.2f}s"))
    _assemble(out, tiles, cols, W, H)

def _assemble(out, tiles, cols, W, H):
    tw, th = W // 2, H // 2
    rows = (len(tiles) + cols - 1) // cols
    S = Image.new('RGB', (cols * tw, rows * (th + 26)), (18, 12, 20))
    d = ImageDraw.Draw(S)
    f = ImageFont.truetype(FONT_SUB, 18)
    for i, (img, lab) in enumerate(tiles):
        x, y = (i % cols) * tw, (i // cols) * (th + 26)
        S.paste(img.resize((tw, th), Image.LANCZOS), (x, y + 26))
        d.text((x + 6, y + 4), lab, font=f, fill=(255, 210, 160))
    S.save(out, quality=90)
    print('sheet', out, S.size)

def sheet(out, W=960, per_shot=3, only=None):
    H = int(round(W * ASPECT / 2)) * 2
    r = make_renderer(W, H)
    tiles = []
    for name, t0, t1, fn in film.SHOTS:
        if only and name not in only:
            continue
        for k in range(per_shot):
            T = t0 + (t1 - t0) * (k + 0.5) / per_shot
            img, _ = render_frame(r, T, W, H)
            tiles.append((img, f"{name}  {T:05.2f}s"))
    cols = per_shot if per_shot > 1 else 4
    tw, th = W // 2, H // 2
    rows = (len(tiles) + cols - 1) // cols
    S = Image.new('RGB', (cols * tw, rows * (th + 26)), (18, 12, 20))
    d = ImageDraw.Draw(S)
    f = ImageFont.truetype(FONT_SUB, 18)
    for i, (img, lab) in enumerate(tiles):
        x, y = (i % cols) * tw, (i // cols) * (th + 26)
        S.paste(img.resize((tw, th), Image.LANCZOS), (x, y + 26))
        d.text((x + 6, y + 4), lab, font=f, fill=(255, 210, 160))
    S.save(out, quality=90)
    print('sheet', out, S.size)

def shots(outdir, W=3840, only=None):
    H = int(round(W * ASPECT / 2)) * 2
    os.makedirs(outdir, exist_ok=True)
    r = make_renderer(W, H)
    for name, t0, t1, fn in film.SHOTS:
        if only and name not in only:
            continue
        f0, f1 = int(round(t0 * FPS)), int(round(t1 * FPS))
        tmp = os.path.join(outdir, f'{name}.part.mp4'); dst = os.path.join(outdir, f'{name}.mp4')
        cmd = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}',
               '-r', str(FPS), '-i', '-', '-c:v', 'libx264', '-preset', 'faster', '-crf', '17', '-tune', 'animation',
               '-pix_fmt', 'yuv420p', '-r', str(FPS), tmp]
        p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        t = time.time()
        for fi in range(f0, f1):
            img, _ = render_frame(r, fi / FPS, W, H)
            p.stdin.write(img.tobytes())
        p.stdin.close()
        if p.wait() == 0:
            os.replace(tmp, dst)
        print(f'{name}: {f1 - f0} frames in {time.time() - t:.0f}s', flush=True)

if __name__ == '__main__':
    mode = sys.argv[1]
    if mode == 'sheet':
        only = sys.argv[5].split(',') if len(sys.argv) > 5 else None
        sheet(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 960, int(sys.argv[4]) if len(sys.argv) > 4 else 3, only)
    elif mode == 'times':
        times_sheet(sys.argv[2], [float(x) for x in sys.argv[3].split(',')], int(sys.argv[4]) if len(sys.argv) > 4 else 960)
    elif mode == 'shots':
        only = sys.argv[4].split(',') if len(sys.argv) > 4 else None
        shots(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 3840, only)
