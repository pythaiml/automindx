'use client';
// A living visual expression of sAGI self-building: each module on disk is a node
// placed on an outward golden-angle spiral around a pulsing core, connected back to
// it. The newest node glows; the whole constellation drifts, and surges while a
// build is active. Polls sagi/ on disk, so it watches both the in-console loop and
// the terminal's green-light `sagi_build.py` — one grows before your eyes.
export type Mod = { step: number; title: string; ts: number };

const GA = 2.399963; // golden angle (radians)
const CX = 150;
const CY = 130;

export default function SagiVisual({ modules, building }: { modules: Mod[]; building: boolean }) {
  const nodes = modules.map((m, i) => {
    const r = 20 + i * 6.2;
    const a = i * GA;
    return { ...m, i, x: CX + r * Math.cos(a), y: CY + r * Math.sin(a) };
  });
  const last = nodes[nodes.length - 1];
  const n = nodes.length;

  return (
    <div className={'sagiviz' + (building ? ' building' : '')}>
      <svg viewBox="0 0 300 260" className="sagiviz-svg" role="img" aria-label={`sAGI: ${n} modules`}>
        <g className="sv-spin">
          {nodes.map((nd) => (
            <line key={'l' + nd.step} x1={CX} y1={CY} x2={nd.x} y2={nd.y} className="sv-link"
              style={{ animationDelay: `${nd.i * 40}ms` }} />
          ))}
          {nodes.map((nd) => (
            <circle key={nd.step} cx={nd.x} cy={nd.y} r={nd.i === n - 1 ? 6 : 4.5}
              className={'sv-node' + (nd.i === n - 1 ? ' latest' : '')}
              style={{ animationDelay: `${nd.i * 40}ms` }}>
              <title>{nd.step}. {nd.title}</title>
            </circle>
          ))}
        </g>
        <circle cx={CX} cy={CY} r={13} className="sv-core" />
        <text x={CX} y={CY + 4} className="sv-core-n" textAnchor="middle">{n}</text>
      </svg>
      <div className="sv-meta">
        {building ? <span className="sv-status"><span className="spin" /> growing</span>
          : <span className="sv-status idle">◈ stable</span>}
        <span className="sv-label">{n ? <>latest → <b>{last.title}</b></> : 'awaiting first module…'}</span>
      </div>
    </div>
  );
}
