"""Procedural geometry: flat-shaded triangle soup builders (no indices).
All builders return dict(v=Nx3 f4 positions, c=Nx3 f4 colors) — normals are
derived per-face at combine time for crisp toon facets.
"""
import numpy as np

def _soup(tris, color):
    v = np.asarray(tris, dtype='f4').reshape(-1, 3)
    c = np.tile(np.asarray(color, 'f4'), (len(v), 1))
    return {'v': v, 'c': c}

def face_normals(v):
    t = v.reshape(-1, 3, 3)
    n = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    n = n / np.maximum(ln, 1e-9)
    return np.repeat(n, 3, axis=0).astype('f4')

def box(sx, sy, sz, color, cx=0, cy=0, cz=0):
    x, y, z = sx / 2, sy / 2, sz / 2
    p = np.array([[-x,-y,-z],[x,-y,-z],[x,y,-z],[-x,y,-z],
                  [-x,-y, z],[x,-y, z],[x,y, z],[-x,y, z]], 'f4')
    p += [cx, cy, cz]
    f = [(0,2,1),(0,3,2),(4,5,6),(4,6,7),(0,1,5),(0,5,4),
         (3,7,6),(3,6,2),(0,4,7),(0,7,3),(1,2,6),(1,6,5)]
    tris = p[np.array(f).ravel()]
    return _soup(tris, color)

def tapered_box(sx0, sz0, sx1, sz1, h, color, cy=0):
    """Frustum box: bottom (sx0,sz0) at y=cy, top (sx1,sz1) at y=cy+h."""
    b = [(-sx0/2,cy,-sz0/2),(sx0/2,cy,-sz0/2),(sx0/2,cy,sz0/2),(-sx0/2,cy,sz0/2)]
    t = [(-sx1/2,cy+h,-sz1/2),(sx1/2,cy+h,-sz1/2),(sx1/2,cy+h,sz1/2),(-sx1/2,cy+h,sz1/2)]
    p = np.array(b + t, 'f4')
    f = [(0,1,5),(0,5,4),(1,2,6),(1,6,5),(2,3,7),(2,7,6),(3,0,4),(3,4,7),
         (4,5,6),(4,6,7),(0,3,2),(0,2,1)]
    return _soup(p[np.array(f).ravel()], color)

def cylinder(r, h, color, n=8, cy=0, r_top=None, cap=True):
    rt = r if r_top is None else r_top
    a = np.linspace(0, 2*np.pi, n, endpoint=False)
    bot = np.stack([r*np.cos(a), np.full(n, cy), r*np.sin(a)], 1)
    top = np.stack([rt*np.cos(a), np.full(n, cy+h), rt*np.sin(a)], 1)
    tris = []
    for i in range(n):
        j = (i+1) % n
        tris += [bot[i], top[j], bot[j], bot[i], top[i], top[j]]
        if cap:
            tris += [[0,cy+h,0], top[j], top[i], [0,cy,0], bot[i], bot[j]]
    return _soup(tris, color)

def sphere(r, color, n=8, squash=(1,1,1)):
    tris = []
    for i in range(n):
        for j in range(n):
            t0, t1 = np.pi*i/n, np.pi*(i+1)/n
            p0, p1 = 2*np.pi*j/n, 2*np.pi*(j+1)/n
            def pt(t, p):
                return [r*np.sin(t)*np.cos(p)*squash[0], r*np.cos(t)*squash[1], r*np.sin(t)*np.sin(p)*squash[2]]
            a_, b_, c_, d_ = pt(t0,p0), pt(t0,p1), pt(t1,p1), pt(t1,p0)
            tris += [a_, b_, c_, a_, c_, d_]
    return _soup(tris, color)

def quad(p0, p1, p2, p3, color):
    return _soup([p0, p1, p2, p0, p2, p3], color)

def combine(parts):
    v = np.concatenate([p['v'] for p in parts])
    c = np.concatenate([p['c'] for p in parts])
    return {'v': v, 'c': c}

# ---- transforms -------------------------------------------------------------
def rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1,0,0],[0,c,-s],[0,s,c]], 'f4')

def rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c,0,s],[0,1,0],[-s,0,c]], 'f4')

def rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c,-s,0],[s,c,0],[0,0,1]], 'f4')

def transform(part, R=None, t=(0,0,0), s=1.0):
    v = part['v'] * s
    if R is not None:
        v = v @ R.T
    v = v + np.asarray(t, 'f4')
    return {'v': v.astype('f4'), 'c': part['c']}

def mat4(R=None, t=(0,0,0)):
    M = np.eye(4, dtype='f4')
    if R is not None:
        M[:3, :3] = R
    M[:3, 3] = t
    return M

def look_at(eye, target, up=(0,1,0)):
    eye, target = np.asarray(eye,'f4'), np.asarray(target,'f4')
    f = target - eye; f /= np.linalg.norm(f)
    r = np.cross(f, np.asarray(up,'f4')); r /= max(np.linalg.norm(r),1e-9)
    u = np.cross(r, f)
    M = np.eye(4,dtype='f4')
    M[0,:3], M[1,:3], M[2,:3] = r, u, -f
    M[:3,3] = -M[:3,:3] @ eye
    return M

def perspective(fovy_deg, aspect, near, far):
    f = 1.0/np.tan(np.radians(fovy_deg)/2)
    M = np.zeros((4,4),'f4')
    M[0,0] = f/aspect; M[1,1] = f
    M[2,2] = (far+near)/(near-far); M[2,3] = 2*far*near/(near-far)
    M[3,2] = -1
    return M

def ortho(l, r, b, t, near, far):
    M = np.eye(4,dtype='f4')
    M[0,0] = 2/(r-l); M[1,1] = 2/(t-b); M[2,2] = -2/(far-near)
    M[0,3] = -(r+l)/(r-l); M[1,3] = -(t+b)/(t-b); M[2,3] = -(far+near)/(far-near)
    return M
