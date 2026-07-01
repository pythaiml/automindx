'use client';
// A real interactive terminal embedded in the dapp (xterm.js ⇄ pty-server.js).
// Type `claude` to sign in with your Claude subscription (opens the browser) and
// drive Claude Code against this very project. Everything runs on the local host.
import { useEffect, useRef } from 'react';
import '@xterm/xterm/css/xterm.css';

const PTY_PORT = 3101;

export default function ClaudeTerminal({ onClose }: { onClose: () => void }) {
  const hostRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    let disposed = false;
    let cleanup = () => {};
    (async () => {
      const [{ Terminal }, { FitAddon }] = await Promise.all([
        import('@xterm/xterm'),
        import('@xterm/addon-fit'),
      ]);
      if (disposed || !hostRef.current) return;

      const term = new Terminal({
        fontFamily: 'JetBrains Mono, ui-monospace, monospace',
        fontSize: 13, cursorBlink: true, convertEol: true,
        theme: {
          background: '#080d15', foreground: '#d7e2f0', cursor: '#2ee6a6',
          selectionBackground: 'rgba(46,230,166,.25)',
          black: '#0b1220', green: '#2ee6a6', blue: '#37b6ff', cyan: '#5be0d0',
          red: '#ff5c7a', yellow: '#f5c451', magenta: '#c792ea', white: '#d7e2f0',
        },
      });
      const fit = new FitAddon();
      term.loadAddon(fit);
      term.open(hostRef.current);
      fit.fit();

      const ws = new WebSocket(`ws://127.0.0.1:${PTY_PORT}`);
      const sendResize = () => ws.readyState === 1 && ws.send(`\x01${term.cols}x${term.rows}`);

      ws.onopen = () => {
        term.writeln('\x1b[38;5;42m▸ interactive terminal — cwd is the automindX project\x1b[0m');
        term.writeln('\x1b[38;5;245m  type \x1b[1;38;5;42mclaude\x1b[0;38;5;245m to sign in with your subscription (opens the browser), then work on the dapp.\x1b[0m\r\n');
        sendResize();
      };
      ws.onmessage = (e) => term.write(typeof e.data === 'string' ? e.data : new Uint8Array(e.data as ArrayBuffer));
      ws.onclose = () => term.writeln('\r\n\x1b[38;5;203m[terminal server not connected — start it: node codephreak-console/pty-server.js]\x1b[0m');
      term.onData((d) => ws.readyState === 1 && ws.send(d));

      const ro = new ResizeObserver(() => { try { fit.fit(); sendResize(); } catch { /* mid-teardown */ } });
      ro.observe(hostRef.current);
      term.focus();

      cleanup = () => { ro.disconnect(); try { ws.close(); } catch { /* noop */ } term.dispose(); };
    })();
    return () => { disposed = true; cleanup(); };
  }, []);

  return (
    <div className="term-overlay" onMouseDown={onClose}>
      <div className="term-window" onMouseDown={(e) => e.stopPropagation()}>
        <div className="term-bar">
          <span className="term-dots"><i /><i /><i /></span>
          <span className="term-title">⌘ Claude terminal — subscription CLI · project: automindX</span>
          <button className="btn ghost sm icon" onClick={onClose} title="close">✕</button>
        </div>
        <div className="term-host" ref={hostRef} />
      </div>
    </div>
  );
}
