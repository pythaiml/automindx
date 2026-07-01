'use client';
import { useEffect, useRef } from 'react';

// A tasteful WebGL substrate — a slow, low-opacity neural/plasma field that
// shifts with the active persona. An homage to the automindX WebGL substrate at
// https://mindx.pythai.net/automindx. Degrades to nothing without WebGL and holds
// a single static frame under prefers-reduced-motion. Colours/seed lerp smoothly
// when the persona changes, so the substrate "becomes" each persona.
const FRAG = `
precision highp float;
uniform vec2 u_res; uniform float u_time; uniform float u_seed;
uniform vec3 u_a; uniform vec3 u_b;
float hash(vec2 p){ return fract(sin(dot(p, vec2(127.1,311.7)))*43758.5453); }
float noise(vec2 p){ vec2 i=floor(p), f=fract(p); vec2 u=f*f*(3.0-2.0*f);
  return mix(mix(hash(i),hash(i+vec2(1,0)),u.x), mix(hash(i+vec2(0,1)),hash(i+vec2(1,1)),u.x), u.y); }
float fbm(vec2 p){ float v=0.0, a=0.5; for(int i=0;i<5;i++){ v+=a*noise(p); p*=2.0; a*=0.5; } return v; }
void main(){
  vec2 uv = gl_FragCoord.xy / u_res.xy;
  vec2 p = uv * 3.0 + u_seed; p.x *= u_res.x/u_res.y;
  float t = u_time * 0.03;
  vec2 q = vec2(fbm(p + t), fbm(p - t + 5.2 + u_seed));
  float f = fbm(p + 1.6*q + vec2(t*1.3, -t));
  float fil = smoothstep(0.35, 0.9, f);
  vec3 col = mix(u_a, u_b, clamp(q.x*1.3, 0.0, 1.0)) * (0.10 + 0.28*fil);
  float vig = smoothstep(1.25, 0.2, length(uv-0.5));
  gl_FragColor = vec4(col * vig, 1.0);
}`;
const VERT = `attribute vec2 p; void main(){ gl_Position = vec4(p,0.0,1.0); }`;

function hexToRgb(hex: string): [number, number, number] {
  const m = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec((hex || '').trim());
  return m ? [parseInt(m[1], 16) / 255, parseInt(m[2], 16) / 255, parseInt(m[3], 16) / 255] : [0.18, 0.9, 0.65];
}
const lerp = (a: number, b: number, t: number) => a + (b - a) * t;

export default function Substrate({ a = '#2ee6a6', b = '#37b6ff', seed = 0 }: { a?: string; b?: string; seed?: number }) {
  const ref = useRef<HTMLCanvasElement>(null);
  const target = useRef({ a: hexToRgb(a), b: hexToRgb(b), seed });
  target.current = { a: hexToRgb(a), b: hexToRgb(b), seed }; // latest, read by the loop

  useEffect(() => {
    const cv = ref.current; if (!cv) return;
    const gl = cv.getContext('webgl', { antialias: false, alpha: true, premultipliedAlpha: false });
    if (!gl) return;
    const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

    const compile = (type: number, src: string) => { const s = gl.createShader(type)!; gl.shaderSource(s, src); gl.compileShader(s); return s; };
    const prog = gl.createProgram()!;
    gl.attachShader(prog, compile(gl.VERTEX_SHADER, VERT));
    gl.attachShader(prog, compile(gl.FRAGMENT_SHADER, FRAG));
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) return;
    gl.useProgram(prog);

    const buf = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 3, -1, -1, 3]), gl.STATIC_DRAW);
    const loc = gl.getAttribLocation(prog, 'p'); gl.enableVertexAttribArray(loc); gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);
    const uRes = gl.getUniformLocation(prog, 'u_res'), uTime = gl.getUniformLocation(prog, 'u_time');
    const uA = gl.getUniformLocation(prog, 'u_a'), uB = gl.getUniformLocation(prog, 'u_b'), uSeed = gl.getUniformLocation(prog, 'u_seed');

    const resize = () => {
      const dpr = Math.min(window.devicePixelRatio || 1, 1.5);
      cv.width = Math.floor(innerWidth * dpr); cv.height = Math.floor(innerHeight * dpr);
      gl.viewport(0, 0, cv.width, cv.height); gl.uniform2f(uRes, cv.width, cv.height);
    };
    resize(); window.addEventListener('resize', resize);

    // current colours lerp toward target for a smooth persona transition
    const cur = { a: [...target.current.a] as number[], b: [...target.current.b] as number[], seed: target.current.seed };
    let raf = 0; const t0 = performance.now();
    const draw = (now: number) => {
      const k = 0.05; const tg = target.current;
      for (let i = 0; i < 3; i++) { cur.a[i] = lerp(cur.a[i], tg.a[i], k); cur.b[i] = lerp(cur.b[i], tg.b[i], k); }
      cur.seed = lerp(cur.seed, tg.seed, k);
      gl.uniform1f(uTime, (now - t0) / 1000); gl.uniform1f(uSeed, cur.seed);
      gl.uniform3fv(uA, cur.a); gl.uniform3fv(uB, cur.b);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
      raf = requestAnimationFrame(draw);
    };
    if (reduce) { gl.uniform1f(uTime, 8.0); gl.uniform1f(uSeed, cur.seed); gl.uniform3fv(uA, cur.a); gl.uniform3fv(uB, cur.b); gl.drawArrays(gl.TRIANGLES, 0, 3); }
    else raf = requestAnimationFrame(draw);

    return () => { cancelAnimationFrame(raf); window.removeEventListener('resize', resize); };
  }, []);

  return <canvas ref={ref} aria-hidden className="substrate" />;
}
