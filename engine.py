"""Toon render engine on moderngl/llvmpipe.
Passes: shadow map -> sky gradient -> toon geometry -> particles -> outline/grade post.
Palette: sunset orange key vs deep violet shadow, thick dark outlines, hard rim.
"""
import moderngl
import numpy as np
from geo import face_normals, look_at, perspective, ortho

W, H = 3840, 2880
FPS = 24

# palette (linear-ish, tuned by eye in sRGB space)
DEEP_VIOLET = (0.136, 0.085, 0.120)
SUN_DIR = np.array([-0.42, 0.30, -0.86], 'f4')  # low sunset sun in the south-west
SUN_DIR /= np.linalg.norm(SUN_DIR)

VS = """
#version 330
in vec3 in_pos; in vec3 in_nrm; in vec3 in_col; in float in_flag;
uniform mat4 u_vp; uniform mat4 u_light_vp;
out vec3 v_nrm; out vec3 v_col; out float v_flag; out vec3 v_wpos; out vec4 v_lpos;
void main(){
    gl_Position = u_vp * vec4(in_pos,1.0);
    v_nrm = in_nrm; v_col = in_col; v_flag = in_flag; v_wpos = in_pos;
    v_lpos = u_light_vp * vec4(in_pos,1.0);
}
"""

FS = """
#version 330
in vec3 v_nrm; in vec3 v_col; in float v_flag; in vec3 v_wpos; in vec4 v_lpos;
uniform vec3 u_sun; uniform vec3 u_eye; uniform sampler2D u_shadow;
uniform vec3 u_fog_col; uniform float u_fog_d0; uniform float u_fog_d1;
out vec4 f_col;

vec3 shade_bands(vec3 base, float lum){
    // three tones: lit / mid / shadow, violet-pushed shadows
    vec3 lit  = base * vec3(1.18, 1.02, 0.80) + vec3(0.06, 0.01, 0.0);
    vec3 mid  = mix(base * 0.72, vec3(0.42, 0.26, 0.50), 0.28);
    vec3 shad = mix(base * 0.34, vec3(0.235, 0.16, 0.38), 0.52);
    if (lum > 0.56) return lit;
    if (lum > 0.18) return mid;
    return shad;
}

void main(){
    vec3 N = normalize(v_nrm);
    float ndl = dot(N, u_sun);
    // shadow map (hard)
    vec3 sp = v_lpos.xyz / v_lpos.w * 0.5 + 0.5;
    float sh = 1.0;
    if (sp.x > 0.0 && sp.x < 1.0 && sp.y > 0.0 && sp.y < 1.0 && sp.z < 1.0){
        float d = texture(u_shadow, sp.xy).r;
        if (sp.z - 0.0025 > d) sh = 0.0;
    }
    float lum = ndl * 0.5 + 0.5;
    if (sh < 0.5) lum = min(lum, 0.14);
    vec3 base = v_col;
    if (v_flag > 1.5 && v_flag < 2.5){
        float y = v_wpos.y + 0.35*sin(v_wpos.z*0.05) + 0.2*sin(v_wpos.x*0.11 + v_wpos.z*0.03);
        int b = int(mod(floor(y/2.4), 5.0));
        vec3 S[5] = vec3[5](vec3(0.86,0.47,0.27), vec3(0.72,0.32,0.22), vec3(0.91,0.62,0.36),
                            vec3(0.80,0.40,0.25), vec3(0.64,0.29,0.23));
        base = S[b];
        if (abs(N.y) > 0.88) base = mix(base, vec3(0.92,0.64,0.40), 0.5);
    }
    vec3 col = shade_bands(base, lum);
    if (v_flag > 2.5 && v_flag < 3.5){
        float l2 = ndl * 0.5 + 0.5;
        col = l2 > 0.45 ? base * vec3(1.08, 1.0, 0.92) : mix(base, vec3(0.62, 0.42, 0.62), 0.38);
    }
    if (v_flag < 0.5 && N.y > 0.9 && v_wpos.y < 0.7){
        float rip = sin(v_wpos.x * 1.3 + sin(v_wpos.z * 0.21) * 2.5 + v_wpos.z * 0.08);
        if (rip > 0.93 && sh > 0.5) col *= vec3(0.90, 0.86, 0.84);
        float blot = sin(v_wpos.x*0.17+1.0)*sin(v_wpos.z*0.13) + 0.5*sin(v_wpos.x*0.41 - v_wpos.z*0.29);
        if (blot > 0.95) col = mix(col, col * vec3(0.92, 0.85, 0.82), 0.8);
    }
    // hard rim light: silhouette edge facing roughly sunward
    vec3 V = normalize(u_eye - v_wpos);
    float rim = 1.0 - max(dot(N, V), 0.0);
    float sunface = smoothstep(-0.25, 0.15, ndl);
    if ((v_flag < 0.5 || (v_flag > 1.5 && v_flag < 2.5)) && rim > 0.62 && sunface > 0.5 && sh > 0.5)
        col = mix(col, vec3(1.05, 0.72, 0.42), 0.85);
    if (v_flag > 0.5 && v_flag < 1.5) col = v_col;          // emissive / unshaded
    // distance haze toward violet
    float dist = length(u_eye - v_wpos);
    float fog = clamp((dist - u_fog_d0) / (u_fog_d1 - u_fog_d0), 0.0, 1.0);
    fog = floor(fog * 4.0) / 4.0;                            // stepped haze (2D feel)
    col = mix(col, u_fog_col, fog * 0.72);
    f_col = vec4(col, 1.0);
}
"""

SHADOW_VS = """
#version 330
in vec3 in_pos; uniform mat4 u_light_vp;
void main(){ gl_Position = u_light_vp * vec4(in_pos,1.0); }
"""
SHADOW_FS = "#version 330\nvoid main(){}"

SKY_FS = """
#version 330
in vec2 uv; out vec4 f_col;
uniform vec3 u_sun_screen;   // x,y in uv space, z = visible flag
uniform float u_horizon;     // horizon line uv.y
uniform float u_pan;
void main(){
    float t = clamp((uv.y - u_horizon) / (1.0 - u_horizon), 0.0, 1.0);
    vec3 lo  = vec3(1.00, 0.62, 0.25);   // horizon orange
    vec3 mid = vec3(0.86, 0.32, 0.34);   // salmon
    vec3 hi  = vec3(0.235, 0.14, 0.38);  // deep violet
    vec3 c = t < 0.38 ? mix(lo, mid, t/0.38) : mix(mid, hi, (t-0.38)/0.62);
    c = floor(c * 7.0 + 0.5) / 7.0;      // banded gradient
    // sun disc
    if (u_sun_screen.z > 0.5){
        vec2 d = (uv - u_sun_screen.xy) * vec2(4.0/3.0, 1.0);
        float r = length(d);
        if (r < 0.052) c = vec3(1.0, 0.92, 0.70);
        else if (r < 0.064) c = vec3(1.0, 0.70, 0.38);
    }
    // thin horizontal cloud bands
    for (int i = 0; i < 6; i++){
        float fi = float(i);
        vec2 cc = vec2(fract(fi*0.37 + 0.11 + u_pan*0.15*(1.0+fi*0.1)), u_horizon + 0.10 + fract(fi*0.53)*0.32);
        vec2 dd = (uv - cc) * vec2(1.0/(0.16 + 0.05*sin(fi*3.1)), 1.0/0.013);
        float k = dot(dd, dd);
        if (k < 1.0) c = (uv.y < cc.y - 0.002) ? vec3(0.62, 0.24, 0.40) : vec3(1.0, 0.66, 0.46);
    }
    f_col = vec4(c, 1.0);
}
"""

EDGE_FS = """
#version 330
in vec2 uv; out vec4 f_col;
uniform sampler2D u_dep; uniform sampler2D u_nrm; uniform vec2 u_px;
float iw(vec2 p){                       // inverse linear depth (affine on planes)
    float d = texture(u_dep, p).r * 2.0 - 1.0;
    float n = 0.3, f = 900.0;
    float z = 2.0 * n * f / (f + n - d * (f - n));
    return 1.0 / z;
}
void main(){
    float w0 = iw(uv);
    float wl = iw(uv - vec2(u_px.x, 0)), wr = iw(uv + vec2(u_px.x, 0));
    float wd = iw(uv - vec2(0, u_px.y)), wu = iw(uv + vec2(0, u_px.y));
    float lap = abs(wl + wr - 2.0 * w0) + abs(wd + wu - 2.0 * w0);
    float e = smoothstep(0.015, 0.05, lap / max(w0, 1e-6));
    vec3 n0 = texture(u_nrm, uv).rgb * 2.0 - 1.0;
    vec3 nr = texture(u_nrm, uv + vec2(u_px.x, 0)).rgb * 2.0 - 1.0;
    vec3 nu = texture(u_nrm, uv + vec2(0, u_px.y)).rgb * 2.0 - 1.0;
    float cr = 1.0 - min(dot(n0, nr), dot(n0, nu));
    e = max(e, smoothstep(0.55, 0.8, cr) * 0.85);
    f_col = vec4(e, 0, 0, 1);
}
"""

DIL_FS = """
#version 330
in vec2 uv; out vec4 f_col;
uniform sampler2D u_src; uniform vec2 u_dir; uniform float u_r;
void main(){
    float m = 0.0;
    int R = int(ceil(u_r));
    for (int k = -R; k <= R; k++){
        float wgt = clamp(u_r + 0.5 - abs(float(k)), 0.0, 1.0);
        m = max(m, texture(u_src, uv + u_dir * float(k)).r * wgt);
    }
    f_col = vec4(m, 0, 0, 1);
}
"""

POST_FS = """
#version 330
in vec2 uv; out vec4 f_col;
uniform sampler2D u_tex; uniform sampler2D u_line;
uniform float u_t; uniform vec3 u_flash;
void main(){
    vec3 col = texture(u_tex, uv).rgb;
    float line = texture(u_line, uv).r;
    col = mix(col, vec3(0.085, 0.055, 0.11), clamp(line, 0.0, 1.0));
    vec2 vg = (uv - 0.5) * vec2(1.15, 1.0);
    col *= 1.0 - dot(vg, vg) * 0.42;
    float g = fract(sin(dot(uv * (u_t + 13.0), vec2(12.9898, 78.233))) * 43758.5);
    col += (g - 0.5) * 0.015;
    col = mix(col, vec3(1.0, 0.93, 0.80), u_flash.x);
    col *= (1.0 - u_flash.y);
    f_col = vec4(col, 1.0);
}
"""

QUAD_VS = "#version 330\nin vec2 p; out vec2 uv; void main(){gl_Position=vec4(p,0.,1.);uv=p*0.5+0.5;}"

NRM_FS = """
#version 330
in vec3 v_nrm; in vec3 v_col; in float v_flag; in vec3 v_wpos; in vec4 v_lpos;
out vec4 f_col;
void main(){ f_col = vec4(normalize(v_nrm)*0.5+0.5, 1.0); }
"""


class Renderer:
    def __init__(self, w=W, h=H, shadow_res=2048):
        self.w, self.h = w, h
        self.ctx = moderngl.create_standalone_context(backend='egl')
        c = self.ctx
        self.prog = c.program(vertex_shader=VS, fragment_shader=FS)
        self.prog_n = c.program(vertex_shader=VS, fragment_shader=NRM_FS)
        self.prog_sh = c.program(vertex_shader=SHADOW_VS, fragment_shader=SHADOW_FS)
        self.prog_sky = c.program(vertex_shader=QUAD_VS, fragment_shader=SKY_FS)
        self.prog_post = c.program(vertex_shader=QUAD_VS, fragment_shader=POST_FS)
        self.prog_edge = c.program(vertex_shader=QUAD_VS, fragment_shader=EDGE_FS)
        self.prog_dil = c.program(vertex_shader=QUAD_VS, fragment_shader=DIL_FS)
        self.quad = c.buffer(np.array([-1,-1, 3,-1, -1,3], 'f4').tobytes())
        self.vao_sky = c.vertex_array(self.prog_sky, [(self.quad, '2f', 'p')])
        self.vao_post = c.vertex_array(self.prog_post, [(self.quad, '2f', 'p')])
        self.vao_edge = c.vertex_array(self.prog_edge, [(self.quad, '2f', 'p')])
        self.vao_dil = c.vertex_array(self.prog_dil, [(self.quad, '2f', 'p')])
        # targets
        self.tex_col = c.texture((w, h), 4)
        self.tex_dep = c.depth_texture((w, h))
        self.tex_nrm = c.texture((w, h), 4)
        self.fbo_main = c.framebuffer([self.tex_col], self.tex_dep)
        self.fbo_nrm = c.framebuffer([self.tex_nrm], self.tex_dep)
        self.tex_out = c.texture((w, h), 3)
        self.fbo_out = c.framebuffer([self.tex_out])
        self.sh_dep = c.depth_texture((shadow_res, shadow_res))
        self.fbo_sh = c.framebuffer(depth_attachment=self.sh_dep)
        self.sh_dep.compare_func = ''
        self.sh_dep.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self.prog_edge['u_px'].value = (1.0/w, 1.0/h)
        self.line_r = 1.9 * w / 1440.0
        self.tex_e1 = c.texture((w, h), 1, dtype='f1'); self.fbo_e1 = c.framebuffer([self.tex_e1])
        self.tex_e2 = c.texture((w, h), 1, dtype='f1'); self.fbo_e2 = c.framebuffer([self.tex_e2])
        for t_ in (self.tex_e1, self.tex_e2):
            t_.filter = (moderngl.NEAREST, moderngl.NEAREST); t_.repeat_x = False; t_.repeat_y = False
        self._vbo_cache = {}

    def upload(self, name, mesh, dynamic=False):
        """mesh: dict(v Nx3, c Nx3) triangle soup; computes face normals + flag col alpha."""
        v, col = mesh['v'], mesh['c']
        n = face_normals(v)
        flag = mesh.get('flag')
        if flag is None:
            flag = np.zeros((len(v), 1), 'f4')
        data = np.hstack([v, n, col, flag]).astype('f4').tobytes()
        if name in self._vbo_cache:
            vbo = self._vbo_cache[name][0]
            if vbo.size >= len(data):
                vbo.write(data)
                self._vbo_cache[name] = (vbo, len(v))
                return
            vbo.release()
        vbo = self.ctx.buffer(data, dynamic=dynamic)
        self._vbo_cache[name] = (vbo, len(v))

    def _vaos(self, names, prog):
        out = []
        attrs = [('in_pos', '3f', '3x4'), ('in_nrm', '3f', '3x4'), ('in_col', '3f', '3x4'), ('in_flag', '1f', '1x4')]
        fmt, used = [], []
        for a, f, skip in attrs:
            if a in prog:
                fmt.append(f); used.append(a)
            else:
                fmt.append(skip)
        fmt = ' '.join(fmt)
        for nm in names:
            vbo, n = self._vbo_cache[nm]
            out.append((self.ctx.vertex_array(prog, [(vbo, fmt, *used)]), n))
        return out

    def render(self, names, cam, sun_center=None, frame_t=0.0, light_span=60.0, light_center=(0,0,0), flash=0.0, black=0.0):
        """cam: dict(eye, target, fov, roll). Returns RGB bytes."""
        c = self.ctx
        eye = np.asarray(cam['eye'], 'f4'); tgt = np.asarray(cam['target'], 'f4')
        up = (0, 1, 0)
        roll = cam.get('roll', 0.0)
        if roll:
            fwd = tgt - eye; fwd = fwd/np.linalg.norm(fwd)
            u0 = np.array([0,1,0],'f4')
            r0 = np.cross(fwd, u0); r0/=max(np.linalg.norm(r0),1e-9)
            u1 = np.cross(r0, fwd)
            up = tuple((np.cos(roll)*u1 + np.sin(roll)*r0).tolist())
        view = look_at(eye, tgt, up)
        proj = perspective(cam.get('fov', 45), self.w/self.h, 0.3, 900.0)
        vp = (proj @ view).T.astype('f4')
        # light matrices (ortho around light_center)
        lc = np.asarray(light_center, 'f4')
        lview = look_at(lc + SUN_DIR*140.0, lc)
        s = light_span
        lproj = ortho(-s, s, -s, s, 10.0, 320.0)
        lvp = (lproj @ lview).T.astype('f4')

        # shadow pass
        self.fbo_sh.use(); c.clear(depth=1.0)
        c.enable(moderngl.DEPTH_TEST)
        self.prog_sh['u_light_vp'].write(lvp.tobytes())
        for vao, n in self._vaos(names, self.prog_sh):
            vao.render(vertices=n); vao.release()

        # sky + main
        self.fbo_main.use()
        c.disable(moderngl.DEPTH_TEST)
        c.clear(*DEEP_VIOLET, depth=1.0)
        # sun in screen space
        if sun_center is None:
            sun_world = eye + SUN_DIR * 500.0
        else:
            sun_world = np.asarray(sun_center, 'f4')
        sw = vp.T @ np.array([*sun_world, 1.0], 'f4')
        if sw[3] > 0:
            ss = (sw[0]/sw[3]*0.5+0.5, sw[1]/sw[3]*0.5+0.5, 1.0)
        else:
            ss = (0, 0, 0.0)
        # horizon: project far point ahead at y=0
        fwd = tgt - eye; fwd = fwd/np.linalg.norm(fwd)
        hp = eye + np.array([fwd[0],0,fwd[2]],'f4')/max(np.linalg.norm([fwd[0],fwd[2]]),1e-6)*600.0
        hp[1] = 0
        hw = vp.T @ np.array([*hp, 1.0], 'f4')
        hy = float(np.clip(hw[1]/max(hw[3],1e-6)*0.5+0.5, -0.5, 0.9))
        self.prog_sky['u_sun_screen'].value = ss
        self.prog_sky['u_horizon'].value = hy
        yaw = float(np.arctan2(fwd[0], -fwd[2]))
        self.prog_sky['u_pan'].value = -yaw / (np.radians(cam.get('fov',45))*1.33) * 0.5
        self.vao_sky.render()
        c.enable(moderngl.DEPTH_TEST)
        self.prog['u_vp'].write(vp.tobytes())
        self.prog['u_light_vp'].write(lvp.tobytes())
        self.prog['u_sun'].value = tuple(SUN_DIR.tolist())
        self.prog['u_eye'].value = tuple(eye.tolist())
        self.prog['u_fog_col'].value = (0.62, 0.33, 0.47)
        self.prog['u_fog_d0'].value = 90.0
        self.prog['u_fog_d1'].value = 420.0
        self.sh_dep.use(0); self.prog['u_shadow'] = 0
        for vao, n in self._vaos(names, self.prog):
            vao.render(vertices=n); vao.release()

        # normal buffer (reuses depth: EQUAL test keeps visible surfaces only)
        self.fbo_nrm.use()
        c.clear(0.5, 0.5, 1.0)
        self.prog_n['u_vp'].write(vp.tobytes())
        if 'u_light_vp' in self.prog_n: self.prog_n['u_light_vp'].write(lvp.tobytes())
        c.depth_func = '<='
        for vao, n in self._vaos(names, self.prog_n):
            vao.render(vertices=n); vao.release()
        c.depth_func = '<'

        # post
        c.disable(moderngl.DEPTH_TEST)
        # edges -> e1
        self.fbo_e1.use()
        self.tex_dep.use(1); self.tex_nrm.use(2)
        self.prog_edge['u_dep'] = 1; self.prog_edge['u_nrm'] = 2
        self.vao_edge.render()
        # dilate horizontally e1 -> e2, vertically e2 -> e1
        self.prog_dil['u_r'].value = self.line_r
        self.fbo_e2.use(); self.tex_e1.use(3); self.prog_dil['u_src'] = 3
        self.prog_dil['u_dir'].value = (1.0 / self.w, 0.0); self.vao_dil.render()
        self.fbo_e1.use(); self.tex_e2.use(3)
        self.prog_dil['u_dir'].value = (0.0, 1.0 / self.h); self.vao_dil.render()
        # composite
        self.fbo_out.use()
        self.tex_col.use(0); self.tex_e1.use(4)
        self.prog_post['u_tex'] = 0; self.prog_post['u_line'] = 4
        self.prog_post['u_t'].value = frame_t % 7.0
        self.prog_post['u_flash'].value = (float(flash), float(black), 0.0)
        self.vao_post.render()
        return self.fbo_out.read(components=3)
