// pty-server.js — a local-only PTY bridge so the dapp has a REAL interactive
// terminal. This is what lets you type `claude` inside the console and complete
// the browser sign-in with your Claude subscription, then keep working — from the
// dapp, on the dapp (the shell opens in the project root).
//
// It is bound to 127.0.0.1 only (never exposed off-box). Start it alongside the
// console:  node pty-server.js   (the run script does this for you).
const os = require('node:os');
const path = require('node:path');
const { WebSocketServer } = require('ws');
const pty = require('node-pty');

const PORT = Number(process.env.PTY_PORT || 3101);
const ROOT = process.env.AUTOMINDX_ROOT || path.resolve(__dirname, '..'); // repo root
const SHELL = process.env.SHELL || (os.platform() === 'win32' ? 'powershell.exe' : 'bash');

const wss = new WebSocketServer({ port: PORT, host: '127.0.0.1' });

wss.on('connection', (ws) => {
  const term = pty.spawn(SHELL, [], {
    name: 'xterm-color', cols: 80, rows: 24, cwd: ROOT, env: process.env,
  });
  term.onData((d) => { try { ws.send(d); } catch { /* closed */ } });
  term.onExit(() => { try { ws.close(); } catch { /* closed */ } });

  ws.on('message', (msg) => {
    const s = msg.toString();
    if (s[0] === '\x01') {                       // control: \x01<cols>x<rows>
      const [c, r] = s.slice(1).split('x');
      try { term.resize(parseInt(c, 10) || 80, parseInt(r, 10) || 24); } catch { /* ignore */ }
      return;
    }
    term.write(s);                               // raw keystrokes → shell
  });
  ws.on('close', () => { try { term.kill(); } catch { /* already dead */ } });
});

console.log(`[pty] interactive terminal on ws://127.0.0.1:${PORT}  (cwd ${ROOT}, shell ${SHELL})`);
