"""Join shot files, mux soundtrack, write SRT, build the review frame sheet.
  python3 assemble.py <shotdir> <out.mp4> <sheet.jpg>
"""
import sys, os, subprocess
from PIL import Image, ImageDraw, ImageFont
import film

shotdir, out, sheet = sys.argv[1], sys.argv[2], sys.argv[3]

def tc(t):
    h = int(t // 3600); m = int(t % 3600 // 60); s = t % 60
    return f"{h:02d}:{m:02d}:{int(s):02d},{int(round((s - int(s)) * 1000)):03d}"

# subtitles also as a sidecar .srt (they are burnt into the picture too)
srt = os.path.splitext(out)[0] + '.srt'
with open(srt, 'w') as f:
    for i, (a, b, s) in enumerate(film.SUBS, 1):
        f.write(f"{i}\n{tc(a)} --> {tc(b)}\n{s.replace('RANGER: ', 'RANGER: ').strip()}\n\n")

lst = os.path.join(shotdir, 'list.txt')
with open(lst, 'w') as f:
    for name, *_ in film.SHOTS:
        f.write(f"file '{os.path.abspath(os.path.join(shotdir, name + '.mp4'))}'\n")
silent = os.path.join(shotdir, '_joined.mp4')
subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'concat', '-safe', '0', '-i', lst, '-c', 'copy', silent], check=True)
tmp = out + '.part.mp4'
subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', silent, '-i', 'soundtrack.wav', '-map', '0:v', '-map', '1:a',
                '-c:v', 'copy', '-c:a', 'aac', '-b:a', '256k', '-shortest', '-movflags', '+faststart', tmp], check=True)
os.replace(tmp, out)

# frame sheet: two frames per shot, labelled with shot number + timecode
times = []
for i, (name, a, b, fn) in enumerate(film.SHOTS, 1):
    d = b - a
    picks = [a + d * 0.3, a + d * 0.75] if d >= 2.0 else [a + d * 0.5]
    for t in picks:
        times.append((i, name, t))
tw, th = 640, 480
cols = 4
rows = (len(times) + cols - 1) // cols
pad = 34
S = Image.new('RGB', (cols * tw + (cols + 1) * 10, rows * (th + pad) + 10 + 70), (20, 13, 22))
d = ImageDraw.Draw(S)
fb = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed-Bold.ttf', 22)
ft = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf', 36)
d.text((12, 14), f"PATCH  —  {os.path.basename(out)}  —  frame sheet (shot no. · timecode)", font=ft, fill=(255, 190, 110))
for k, (i, name, t) in enumerate(times):
    fr = os.path.join(shotdir, f'_f{k}.png')
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-ss', f'{t:.3f}', '-i', out, '-frames:v', '1',
                    '-vf', f'scale={tw}:{th}:flags=lanczos', fr], check=True)
    x = 10 + (k % cols) * (tw + 10); y = 70 + (k // cols) * (th + pad)
    S.paste(Image.open(fr), (x, y + pad - 4))
    label = name.split('_', 1)[1].replace('_', ' ')
    d.text((x + 2, y + 2), f"#{i}  {label}", font=fb, fill=(255, 214, 170))
    d.text((x + tw - 4, y + 2), f"{int(t // 60):d}:{t % 60:05.2f}", font=fb, fill=(200, 170, 220), anchor='ra')
    os.remove(fr)
S.save(sheet, quality=88)
print('done', out, sheet, S.size)
