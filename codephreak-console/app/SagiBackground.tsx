'use client';
// A generative WebGL background for the sAGI tab whose geometry EVOLVES with the
// build — a unique mathematical expression:
//   • a golden-ratio phyllotaxis spiral (Fibonacci → golden angle) whose points
//     light up as modules are built;
//   • a rotating polygon that morphs 3→∞ (triangle spins up into a circle), then
//     slows and settles to a square (4), and RANDOM from there.
// Each built module advances the evolution; time keeps it alive.
import { useEffect, useRef } from 'react';

const FRAG = `
precision highp float;
uniform vec2 u_res; uniform float u_time; uniform float u_sides; uniform float u_rot; uniform float u_energy; uniform float u_count;
#define PI 3.14159265
float hash(float n){ return fract(sin(n)*43758.5453); }
// distance to a regular N-gon (float N, so it morphs continuously; large N → circle)
float ngon(vec2 p, float r, float N){
  float an = PI / N;
  float a = atan(p.y, p.x) + u_rot;
  float d = cos(floor(0.5 + a/(2.0*an)) * 2.0*an - a) * length(p);
  return d - r*cos(an);
}
void main(){
  vec2 uv = (gl_FragCoord.xy - 0.5*u_res.xy) / u_res.y;   // centered, aspect-correct
  vec3 col = vec3(0.02,0.03,0.05);
  // Fibonacci / golden-ratio phyllotaxis — points light up as sAGI grows.
  float ga = 2.399963;
  float lit = clamp(u_count, 0.0, 89.0);
  for(int i=0;i<89;i++){
    float fi=float(i);
    float rr = sqrt(fi/89.0)*0.66;
    float a = fi*ga + u_time*0.05;
    vec2 c = vec2(cos(a), sin(a))*rr;
    float dd = length(uv - c);
    float on = 1.0 - smoothstep(lit-1.0, lit+1.0, fi);    // built points glow (fi ≤ count)
    col += (0.0008 + 0.0012*u_energy) / (dd+0.004) * on * vec3(0.16,0.92,0.70);
  }
  // Morphing rotating polygon: triangle → circle → square → random.
  float d = ngon(uv, 0.36, max(3.0, u_sides));
  float edge = smoothstep(0.014, 0.0, abs(d));
  float fill = smoothstep(0.0, -0.2, d) * 0.10;
  col += edge * (0.55 + 0.9*u_energy) * vec3(0.22,1.0,0.78);
  col += fill * vec3(0.12,0.55,0.46);
  col *= smoothstep(1.15, 0.15, length(uv));              // vignette
  gl_FragColor = vec4(col, 1.0);
}`;

const VERT = 'attribute vec2 p; void main(){ gl_Position = vec4(p, 0.0, 1.0); }';
const smooth = (x: number) => { const t = Math.max(0, Math.min(1, x)); return t * t * (3 - 2 * t); };
const mix = (a: number, b: number, t: number) => a + (b - a) * t;
const hash = (n: number) => { const s = Math.sin(n) * 43758.5453; return s - Math.floor(s); };

export default function SagiBackground({ count, building }: { count: number; building: boolean }) {
  const ref = useRef<HTMLCanvasElement>(null);
  const countRef = useRef(count); countRef.current = count;
  const buildRef = useRef(building); buildRef.current = building;

  useEffect(() => {
    const cv = ref.current;
    if (!cv) return;
    const gl = cv.getContext('webgl', { antialias: false, alpha: true });
    if (!gl) return;
    const sh = (t: number, src: string) => { const s = gl.createShader(t)!; gl.shaderSource(s, src); gl.compileShader(s); return s; };
    const prog = gl.createProgram()!;
    gl.attachShader(prog, sh(gl.VERTEX_SHADER, VERT));
    gl.attachShader(prog, sh(gl.FRAGMENT_SHADER, FRAG));
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) return;
    gl.useProgram(prog);
    const buf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 3, -1, -1, 3]), gl.STATIC_DRAW);
    const loc = gl.getAttribLocation(prog, 'p');
    gl.enableVertexAttribArray(loc);
    gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);
    const U = (n: string) => gl.getUniformLocation(prog, n);
    const uRes = U('u_res'), uTime = U('u_time'), uSides = U('u_sides'), uRot = U('u_rot'), uEnergy = U('u_energy'), uCount = U('u_count');

    const resize = () => { const r = cv.getBoundingClientRect(); const dpr = Math.min(2, window.devicePixelRatio || 1); cv.width = Math.max(2, r.width * dpr); cv.height = Math.max(2, r.height * dpr); gl.viewport(0, 0, cv.width, cv.height); };
    resize();
    const ro = new ResizeObserver(resize); ro.observe(cv);

    let raf = 0, rot = 0, energy = 0.3, last = 0;
    const t0 = performance.now();
    const draw = (now: number) => {
      const dt = last ? Math.min(0.05, (now - last) / 1000) : 0.016; last = now;
      const elapsed = (now - t0) / 1000;
      const phase = countRef.current * 0.5 + elapsed * 0.03;   // builds advance it; time drifts
      let sides: number, spin: number;
      if (phase < 1) { sides = mix(3, 44, smooth(phase)); spin = mix(0.15, 1.5, phase); }          // triangle → circle (spinning up)
      else if (phase < 2) { const k = phase - 1; sides = mix(44, 4, smooth(k)); spin = mix(1.5, 0.04, smooth(k)); } // circle → square (slowing)
      else { const k = Math.floor(phase); sides = 3 + Math.floor(hash(k) * 6); spin = (hash(k * 1.7 + 0.3) - 0.5) * 1.8; } // random from there
      rot += spin * dt;
      energy += ((buildRef.current ? 1 : 0.3) - energy) * 0.05;
      gl.uniform2f(uRes, cv.width, cv.height);
      gl.uniform1f(uTime, elapsed);
      gl.uniform1f(uSides, sides);
      gl.uniform1f(uRot, rot);
      gl.uniform1f(uEnergy, energy);
      gl.uniform1f(uCount, countRef.current);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
      raf = requestAnimationFrame(draw);
    };
    raf = requestAnimationFrame(draw);
    return () => { cancelAnimationFrame(raf); ro.disconnect(); };
  }, []);

  return <canvas ref={ref} className="sagi-bg" />;
}
