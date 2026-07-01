'use client';
import { useEffect, useRef, useState } from 'react';

export type SagiModule = { step: number; title: string; body: string };

const ATARAXIA_PATIENCE = 3;   // consecutive no-new steps → converged (perfection = Ataraxia)

type Opts = {
  collectChat: (system: string, userText: string) => Promise<string>;
  savante: string;       // Savante persona prompt
  directive: string;     // sAGI directive appended to it
  autonomous: boolean;
  sagi: boolean;
  maxSteps?: number;
  target?: number;       // how many iterations this run should build (≤ maxSteps)
};

// The sAGI self-building loop + on-disk state. Orchestrator-agnostic contract:
// propose → specify → persist (to sagi/ via /api/sagi).
export function useSagi({ collectChat, savante, directive, autonomous, sagi, maxSteps = 16, target }: Opts) {
  const targetRef = useRef(target ?? maxSteps); targetRef.current = target ?? maxSteps;
  const [running, setRunning] = useState(false);
  const [ataraxia, setAtaraxia] = useState<string | null>(null);   // set when the breaker trips
  const [log, setLog] = useState<SagiModule[]>([]);
  const [disk, setDisk] = useState<{ count: number; last?: string; modules?: { step: number; title: string; ts: number }[] }>({ count: 0 });
  const stop = useRef(false);
  const autoRef = useRef(autonomous); autoRef.current = autonomous;
  const sagiRef = useRef(sagi); sagiRef.current = sagi;

  // Turning Autonomous off halts a running loop (sAGI stops building when
  // autonomous is off — as designed).
  useEffect(() => { if (!autonomous) stop.current = true; }, [autonomous]);
  // Turning the sAGI toggle on starts the creation loop (when autonomous is on).
  useEffect(() => { if (sagi && autonomous) runLoop(); /* eslint-disable-next-line */ }, [sagi]);

  async function loadDisk() {
    try {
      const j = await (await fetch('/api/sagi')).json();
      if (j.ok) setDisk({ count: j.count, modules: j.modules });
    } catch {}
  }

  async function buildStep(operatorInput?: string, priorTitles?: string): Promise<string> {
    const built = priorTitles ?? log.map((s) => s.title).join('; ');
    const base = `You are building sAGI — a self-building, agnostic, modular scientific general intelligence — one module per step. Modules built so far: ${built || '(none)'}.`;
    const ask = operatorInput
      ? `${base}\nThe operator directs this step: "${operatorInput}". Specify this module: a short Title line, then a concise spec (purpose · interface · how it plugs into an agnostic core · how it advances self-building).`
      : `${base}\nPropose and specify the NEXT single module (do not repeat one already built): a short Title line, then a concise spec (purpose · interface · how it plugs into an agnostic core · how it advances self-building). Keep it modular and includable in any project (including as a Tauri app).`;
    const text = await collectChat(savante + directive, ask);
    const priorCount = priorTitles !== undefined ? (priorTitles ? priorTitles.split(';').filter((s) => s.trim()).length : 0) : log.length;
    const step = priorCount + 1;
    const title = ((text.split('\n').find((l) => l.trim()) || 'Module')
      .replace(/\*\*/g, '').replace(/^#+\s*|^title:\s*|^module\s*\d*\s*[:\-–]\s*/i, '').trim().slice(0, 90)) || 'Module';
    setLog((l) => [...l, { step: l.length + 1, title, body: text }]);
    try {
      const j = await (await fetch('/api/sagi', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ step, title, body: text }) })).json();
      if (j.ok) setDisk({ count: j.count, last: j.file });
    } catch {}
    return title;
  }

  const norm = (t: string) => t.toLowerCase().replace(/[^a-z0-9]+/g, ' ').trim();
  async function runLoop() {
    if (running) return;
    setRunning(true); stop.current = false; setAtaraxia(null);
    const titles = log.map((s) => s.title);
    const seen = new Set(titles.map(norm));
    let calm = 0;                               // consecutive no-new (redundant) steps
    while (!stop.current && autoRef.current && titles.length < Math.min(maxSteps, targetRef.current)) {
      const title = await buildStep(undefined, titles.join('; '));
      titles.push(title);
      const n = norm(title);
      // Ataraxia circuit breaker: perfection is Ataraxia — stop once converged.
      if (seen.has(n)) { calm += 1; if (calm >= ATARAXIA_PATIENCE) { setAtaraxia('done'); break; } }
      else { calm = 0; seen.add(n); }
      await new Promise((r) => setTimeout(r, 600));
    }
    setRunning(false);
  }
  function stopLoop() { stop.current = true; setRunning(false); }

  return { running, log, setLog, disk, loadDisk, buildStep, runLoop, stopLoop, maxSteps, ataraxia };
}
