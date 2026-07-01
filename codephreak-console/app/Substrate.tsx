'use client';
import { useEffect, useRef } from 'react';

// A living WebGL substrate — a neural/plasma field that shifts with the active
// persona and surges into flux while the model is thinking. An homage to the
// automindX substrate at https://mindx.pythai.net/automindx. Always animates
// (the flux surge is the event cue); degrades to nothing when WebGL is absent.
const FRAG = `
precision highp float;
uniform vec2 u_res; uniform float u_time; uniform float u_seed; uniform float u_flux; uniform float u_calm;
uniform vec3 u_a; uniform vec3 u_b;
float hash(vec2 p){ return fract(sin(dot(p, vec2(127.1,311.7)))*43758.5453); }
float noise(vec2 p){ vec2 i=floor(p), f=fract(p); vec2 u=f*f*(3.0-2.0*f);
  return mix(mix(hash(i),hash(i+vec2(1,0)),u.x), mix(hash(i+vec2(0,1)),hash(i+vec2(1,1)),u.x), u.y); }
float fbm(vec2 p){ float v=0.0, a=0.5; for(int i=0;i<6;i++){ v+=a*noise(p); p=p*2.0+vec2(1.7,9.2); a*=0.5; } return v; }
void main(){
  vec2 uv = gl_FragCoord.xy / u_res.xy;
  vec2 p = uv * 3.0 + u_seed; p.x *= u_res.x/u_res.y;
  // Slow per-persona evolution: a very gentle wander of the noise domain, its phase
  // seeded by the persona, so each substrate is unique AND slowly evolving over time.
  p += (0.45 - 0.3 * u_calm) * vec2(sin(u_time * 0.018 + u_seed), cos(u_time * 0.013 + u_seed * 1.3));
  float t = u_time * (0.06 + 0.08 * u_flux) * (1.0 - 0.7 * u_calm);   // steady; mellow (slow) in the persona picker
  // domain-warped flow — livelier warp under flux
  vec2 q = vec2(fbm(p + t), fbm(p - t + 5.2 + u_seed));
  vec2 r = vec2(fbm(p + (1.6 + 0.8*u_flux)*q + vec2(1.7, 9.2) + t),
                fbm(p + (1.6 + 0.8*u_flux)*q + vec2(8.3, 2.8) - t));
  float f = fbm(p + 1.8*r + vec2(t*1.4, -t*0.9));
  float fil = smoothstep(0.34 - 0.12*u_flux, 0.9, f);
  float pulse = 0.92 + 0.08*sin(u_time*0.9);          // slow breathing
  vec3 col = mix(u_a, u_b, clamp(r.x*1.4, 0.0, 1.0)) * ((0.10 + 0.28*fil) * pulse) * (1.0 + 0.32*u_flux);
  // soft synaptic filaments that brighten while thinking (steady, not flaring)
  col += u_b * (0.10 + 0.28*u_flux) * smoothstep(0.78, 0.97, f);
  col += u_a * 0.14 * u_flux * smoothstep(0.6, 0.75, f);
  // Living thoughts: percolating cells that gently rise and pulse while thinking.
  vec2 bp = p * 2.3 + vec2(u_seed, 0.0);
  bp.y -= u_time * (0.12 + 0.28 * u_flux);            // gentle rise
  float cells = fbm(bp + 0.8 * r);
  float bub = smoothstep(0.52, 0.70, cells) * (1.0 - smoothstep(0.80, 0.95, cells)); // blob rings
  bub *= 0.72 + 0.28 * sin(u_time * 1.7 + cells * 8.0);   // soft shimmer
  col += mix(u_a, u_b, 0.35 + 0.65 * cells) * max(bub, 0.0) * (0.03 + 0.34 * u_flux);
  float vig = smoothstep(1.3, 0.15, length(uv-0.5));
  gl_FragColor = vec4(col * vig, 1.0);
}`;
const VERT = `attribute vec2 p; void main(){ gl_Position = vec4(p,0.0,1.0); }`;

function hexToRgb(hex: string): [number, number, number] {
  const m = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec((hex || '').trim());
  return m ? [parseInt(m[1], 16) / 255, parseInt(m[2], 16) / 255, parseInt(m[3], 16) / 255] : [0.18, 0.9, 0.65];
}
const lerp = (a: number, b: number, t: number) => a + (b - a) * t;

export default function Substrate({ a = '#2ee6a6', b = '#37b6ff', seed = 0, flux = 0, mellow = false }:
  { a?: string; b?: string; seed?: number; flux?: number; mellow?: boolean }) {
  const ref = useRef<HTMLCanvasElement>(null);
  const target = useRef({ a: hexToRgb(a), b: hexToRgb(b), seed, flux, calm: mellow ? 1 : 0 });
  target.current = { a: hexToRgb(a), b: hexToRgb(b), seed, flux, calm: mellow ? 1 : 0 };

  useEffect(() => {
    const cv = ref.current; if (!cv) return;
    const gl = cv.getContext('webgl', { antialias: false, alpha: true, premultipliedAlpha: false });
    if (!gl) return;

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
    const U = (n: string) => gl.getUniformLocation(prog, n);
    const uRes = U('u_res'), uTime = U('u_time'), uA = U('u_a'), uB = U('u_b'), uSeed = U('u_seed'), uFlux = U('u_flux'), uCalm = U('u_calm');

    const resize = () => {
      const dpr = Math.min(window.devicePixelRatio || 1, 1.5);
      cv.width = Math.floor(innerWidth * dpr); cv.height = Math.floor(innerHeight * dpr);
      gl.viewport(0, 0, cv.width, cv.height); gl.uniform2f(uRes, cv.width, cv.height);
    };
    resize(); window.addEventListener('resize', resize);

    const cur = { a: [...target.current.a] as number[], b: [...target.current.b] as number[], seed: target.current.seed, flux: target.current.flux, calm: target.current.calm };
    let raf = 0; const t0 = performance.now();
    const draw = (now: number) => {
      // Gentle, eased crossfade so persona transitions flow seamlessly:
      // colours ease over ~1s, the noise-domain seed drifts very slowly (no
      // abrupt scroll), flux stays responsive to thinking.
      // Subtle persona transitions: colours ease slowly (~2s) and the noise-domain
      // seed drifts very gently, so switching persona never jolts the background.
      const tg = target.current;
      for (let i = 0; i < 3; i++) { cur.a[i] = lerp(cur.a[i], tg.a[i], 0.018); cur.b[i] = lerp(cur.b[i], tg.b[i], 0.018); }
      cur.seed = lerp(cur.seed, tg.seed, 0.005);
      cur.flux = lerp(cur.flux, tg.flux, 0.06);
      cur.calm = lerp(cur.calm, tg.calm, 0.05);           // ease into/out of the mellow picker state
      gl.uniform1f(uTime, (now - t0) / 1000); gl.uniform1f(uSeed, cur.seed); gl.uniform1f(uFlux, cur.flux); gl.uniform1f(uCalm, cur.calm);
      gl.uniform3fv(uA, cur.a); gl.uniform3fv(uB, cur.b);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
      cv.style.opacity = String((0.6 - 0.22 * cur.calm) + 0.16 * cur.flux); // dimmer + calmer in the picker
      raf = requestAnimationFrame(draw);
    };
    raf = requestAnimationFrame(draw);  // always animate — the substrate is alive

    return () => { cancelAnimationFrame(raf); window.removeEventListener('resize', resize); };
  }, []);

  return <canvas ref={ref} aria-hidden className="substrate" />;
}
