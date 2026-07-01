'use client';
// A real interactive terminal embedded in the dapp (xterm.js ⇄ pty-server.js),
// in a draggable / resizable / minimizable / maximizable window. Type `claude` to
// sign in with your Claude subscription (browser OAuth) and drive Claude Code
// against this project. Everything runs on the local host.
import { useEffect, useRef, useState } from 'react';
import '@xterm/xterm/css/xterm.css';

const PTY_PORT = 3101;
type Mode = 'normal' | 'min' | 'max';

// Controller the sAGI chooser uses to drive the terminal — including starting
// interactive `claude`, waiting until it is ready, and injecting a prompt.
export type TermCtl = {
  send: (s: string) => void;                       // raw byte write to the PTY
  restore: () => void;                             // un-minimize so a live TUI is visible
  note: (line: string) => void;                    // write a status line into xterm
  claudeInteractive: (
    planPrompt: string,
    opts?: {
      proceed?: string;              // ultracode + /goal message sent AFTER the plan output finishes
      quietMs?: number;              // launch-readiness quiescence window
      timeoutMs?: number;            // launch-readiness overall timeout
      respondQuietMs?: number;       // plan-output "gone quiet" window (response complete)
      respondTimeoutMs?: number;     // plan-output overall fallback timeout
    },
  ) => Promise<'ready' | 'timeout' | 'login-required'>;
};

export default function ClaudeTerminal({ onClose, onStartSagi, onReady, onClaude }: { onClose: () => void; onStartSagi?: () => void; onReady?: (ctl: TermCtl) => void; onClaude?: () => void }) {
  const hostRef = useRef<HTMLDivElement>(null);
  const refitRef = useRef<() => void>(() => {});
  const sendRef = useRef<(s: string) => void>(() => {});
  const outBufRef = useRef('');        // rolling ANSI-inclusive tail of PTY output
  const lastByteAtRef = useRef(0);     // ms timestamp of last received byte (quiescence)
  const [mode, setMode] = useState<Mode>('normal');
  const [pos, setPos] = useState(() => ({ x: Math.max(16, (window.innerWidth - 900) / 2), y: 72 }));
  const [size, setSize] = useState(() => ({ w: Math.min(900, window.innerWidth * 0.92), h: Math.min(540, window.innerHeight * 0.74) }));
  const drag = useRef<{ sx: number; sy: number; ox: number; oy: number } | null>(null);
  const rz = useRef<{ sx: number; sy: number; ow: number; oh: number } | null>(null);

  useEffect(() => {
    let disposed = false;
    let cleanup = () => {};
    (async () => {
      const [{ Terminal }, { FitAddon }] = await Promise.all([import('@xterm/xterm'), import('@xterm/addon-fit')]);
      if (disposed || !hostRef.current) return;
      const term = new Terminal({
        fontFamily: 'JetBrains Mono, ui-monospace, monospace', fontSize: 13, cursorBlink: true, convertEol: true,
        theme: { background: '#080d15', foreground: '#d7e2f0', cursor: '#2ee6a6', selectionBackground: 'rgba(46,230,166,.25)',
          green: '#2ee6a6', blue: '#37b6ff', cyan: '#5be0d0', red: '#ff5c7a', yellow: '#f5c451', magenta: '#c792ea', white: '#d7e2f0' },
      });
      const fit = new FitAddon();
      term.loadAddon(fit);
      term.open(hostRef.current);
      const refit = () => {
        const el = hostRef.current;
        if (!el || el.clientWidth < 40 || el.clientHeight < 40) return;   // skip when minimized/hidden
        try { fit.fit(); if (ws.readyState === 1) ws.send(`\x01${term.cols}x${term.rows}`); } catch { /* mid-teardown */ }
      };
      refitRef.current = refit;
      refit();

      const ws = new WebSocket(`ws://127.0.0.1:${PTY_PORT}`);
      ws.onopen = () => {
        term.writeln('\x1b[38;5;42m▸ interactive terminal — cwd is the automindX project\x1b[0m');
        term.writeln('\x1b[38;5;245m  type \x1b[1;38;5;42mclaude\x1b[0;38;5;245m to sign in with your subscription (opens the browser), then work on the dapp.\x1b[0m');
        term.writeln('\x1b[38;5;245m  or click the \x1b[1;38;5;42m●\x1b[0;38;5;245m green light to start Claude + let sAGI drive the build.\x1b[0m\r\n');
        refit();
      };
      ws.onmessage = (e) => {
        const s = typeof e.data === 'string' ? e.data : new TextDecoder().decode(new Uint8Array(e.data as ArrayBuffer));
        term.write(s);                                    // display fidelity (unchanged)
        outBufRef.current = (outBufRef.current + s).slice(-8192);   // last ~8KB, scannable
        lastByteAtRef.current = Date.now();
      };
      ws.onclose = () => term.writeln('\r\n\x1b[38;5;203m[terminal server not connected — start it: node codephreak-console/pty-server.js]\x1b[0m');
      term.onData((d) => ws.readyState === 1 && ws.send(d));
      sendRef.current = (s: string) => { if (ws.readyState === 1) { ws.send(s); term.focus(); } };

      // ── interactive-claude driver: start claude → wait for ready → inject prompt ──
      const stripAnsi = (x: string) => x.replace(/\x1b\[[0-9;?]*[ -\/]*[@-~]/g, '').replace(/\x1b\][^\x07]*\x07/g, '');
      const READY_RE = /(Type \/ for commands|for shortcuts|╭─{3,}|┌─{3,}|╰─{3,}|│\s*>|❯|\bclaude \(claude-)/i;
      const LOGIN_RE = /(\/login\b|Select login method|Sign in|Log in with|OAuth|console\.anthropic\.com|https?:\/\/[^\s]*(claude\.ai|anthropic\.com)\/(oauth|login)|Opening browser|Browser did ?n'?t open|Paste (the )?code|Invalid API key|not (?:logged in|authenticated))/i;
      const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));
      const typeLine = async (text: string) => {
        if (ws.readyState !== 1) return;
        ws.send(text); await sleep(150);                  // raw UTF-8 paste, let the box register
        if (ws.readyState === 1) ws.send('\r');           // submit
      };
      // Wait for Claude's response to a just-sent prompt: (a) STARTED — a byte arrived
      // after we submitted (sinceTs); the redraw + animated spinner emit through the
      // whole turn. (b) QUIET — no byte for ≥ quietMs (the plan stream is done).
      const waitForResponse = async (sinceTs: number, wopts: { quietMs?: number; timeoutMs?: number } = {}): Promise<'complete' | 'timeout' | 'closed'> => {
        const quietMs = wopts.quietMs ?? 1500;            // > mid-stream pauses
        const timeoutMs = wopts.timeoutMs ?? 180000;      // hard cap
        const start = Date.now();
        let started = false;
        for (;;) {
          if (ws.readyState !== 1) return 'closed';
          if (!started && lastByteAtRef.current > sinceTs) started = true;   // (a) output began
          const quiet = Date.now() - lastByteAtRef.current >= quietMs;       // (b) output finished
          if (started && quiet) return 'complete';
          if (Date.now() - start >= timeoutMs) return started ? 'complete' : 'timeout';
          await sleep(120);
        }
      };
      const claudeInteractive: TermCtl['claudeInteractive'] = (planPrompt, opts = {}) =>
        new Promise(async (resolve) => {
          const quietMs = opts.quietMs ?? 700, timeoutMs = opts.timeoutMs ?? 15000;
          if (ws.readyState !== 1) { resolve('timeout'); return; }
          outBufRef.current = '';
          const start = Date.now();
          ws.send('claude\r');                            // (1) launch interactive claude
          lastByteAtRef.current = Date.now();
          const outcome: 'ready' | 'timeout' | 'login-required' = await new Promise((res) => {
            const iv = setInterval(() => {                // (2) readiness watch
              const clean = stripAnsi(outBufRef.current);
              const quiet = Date.now() - lastByteAtRef.current >= quietMs;
              const ready = READY_RE.test(clean), login = LOGIN_RE.test(clean);
              const elapsed = Date.now() - start;
              if (ready && quiet) { clearInterval(iv); res('ready'); }
              else if (login && !ready && elapsed > 3000) { clearInterval(iv); res('login-required'); }
              else if (elapsed > timeoutMs) { clearInterval(iv); res(login ? 'login-required' : 'timeout'); }
            }, 120);
          });
          if (outcome === 'login-required') { resolve('login-required'); return; }  // never inject blind
          await sleep(200);
          // ── STAGE 1: send the /plan message ALONE ──
          await typeLine(planPrompt);
          const planSentAt = Date.now();
          // ── STAGE 2: WAIT for the plan output to START then go QUIET (plan finished) ──
          await waitForResponse(planSentAt, { quietMs: opts.respondQuietMs ?? 1500, timeoutMs: opts.respondTimeoutMs ?? 180000 });
          // ── STAGE 3: send the ultracode + /goal proceed message ──
          if (opts.proceed) { await sleep(300); await typeLine(opts.proceed); }
          resolve(outcome);
        });

      onReady?.({
        send: sendRef.current,
        restore: () => setMode('normal'),
        note: (line: string) => term.writeln('\r\n' + line),
        claudeInteractive,
      });

      const ro = new ResizeObserver(() => refit());
      ro.observe(hostRef.current);
      term.focus();
      cleanup = () => { ro.disconnect(); try { ws.close(); } catch { /* noop */ } term.dispose(); };
    })();
    return () => { disposed = true; cleanup(); };
  }, []);

  // Refit whenever the window geometry changes (resize / maximize / restore).
  useEffect(() => { const t = setTimeout(() => refitRef.current(), 60); return () => clearTimeout(t); }, [size, mode]);

  const onBarDown = (e: React.PointerEvent) => {
    if (mode === 'max') return;
    drag.current = { sx: e.clientX, sy: e.clientY, ox: pos.x, oy: pos.y };
    (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
  };
  const onBarMove = (e: React.PointerEvent) => {
    if (!drag.current) return;
    const nx = drag.current.ox + (e.clientX - drag.current.sx);
    const ny = drag.current.oy + (e.clientY - drag.current.sy);
    setPos({ x: Math.max(-size.w + 120, Math.min(window.innerWidth - 80, nx)), y: Math.max(0, Math.min(window.innerHeight - 40, ny)) });
  };
  const onBarUp = () => { drag.current = null; };

  const onRzDown = (e: React.PointerEvent) => {
    e.stopPropagation();
    rz.current = { sx: e.clientX, sy: e.clientY, ow: size.w, oh: size.h };
    (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
  };
  const onRzMove = (e: React.PointerEvent) => {
    if (!rz.current) return;
    setSize({ w: Math.max(360, rz.current.ow + (e.clientX - rz.current.sx)), h: Math.max(200, rz.current.oh + (e.clientY - rz.current.sy)) });
  };
  const onRzUp = () => { rz.current = null; };

  // Green light: activate the sAGI card, which asks how many interactions (with an
  // autonomous option) and then sends the Claude-driven build command back to this
  // PTY — "claude starts, then sAGI passes commands".
  const startClaudeSagi = () => {
    setMode('min');    // shrink so the sAGI card + chooser are visible
    onStartSagi?.();
  };
  const stop = (e: React.PointerEvent) => e.stopPropagation();   // don't drag when clicking a light

  const style: React.CSSProperties =
    mode === 'max' ? { left: 8, top: 8, width: 'calc(100vw - 16px)', height: 'calc(100vh - 16px)' }
      : mode === 'min' ? { left: pos.x, top: pos.y, width: 340, height: 'auto' }
        : { left: pos.x, top: pos.y, width: size.w, height: size.h };

  return (
    <div className={'term-window' + (mode === 'min' ? ' min' : '')} style={style}>
      <div className="term-bar" onPointerDown={onBarDown} onPointerMove={onBarMove} onPointerUp={onBarUp}
           onDoubleClick={() => setMode((m) => (m === 'max' ? 'normal' : 'max'))}>
        <span className="term-dots">
          <i className="tl-red" title="close" onPointerDown={stop} onClick={onClose} />
          <i className="tl-amber" title="minimize" onPointerDown={stop} onClick={() => setMode((m) => (m === 'min' ? 'normal' : 'min'))} />
          <i className="tl-green" title="start Claude + sAGI build — opens the sAGI card to choose interactions" onPointerDown={stop} onClick={startClaudeSagi} />
        </span>
        {onClaude && <button className="term-claude" onPointerDown={(e) => e.stopPropagation()} onClick={onClaude} title="activate Claude — start interactive claude and inject the autonomous sAGI /plan ultracode prompt">◆ Claude</button>}
        <span className="term-title">⌘ Claude terminal — subscription CLI · project: automindX</span>
        <span className="term-win-btns" onPointerDown={(e) => e.stopPropagation()}>
          <button className="btn ghost sm icon" title="minimize" onClick={() => setMode((m) => (m === 'min' ? 'normal' : 'min'))}>—</button>
          <button className="btn ghost sm icon" title={mode === 'max' ? 'restore' : 'maximize'} onClick={() => setMode((m) => (m === 'max' ? 'normal' : 'max'))}>{mode === 'max' ? '❐' : '□'}</button>
          <button className="btn ghost sm icon" title="close" onClick={onClose}>✕</button>
        </span>
      </div>
      <div className="term-host" ref={hostRef} />
      {mode === 'normal' && <div className="term-resize" title="drag to resize" onPointerDown={onRzDown} onPointerMove={onRzMove} onPointerUp={onRzUp} />}
    </div>
  );
}
