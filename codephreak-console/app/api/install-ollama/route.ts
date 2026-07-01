// One-click Ollama install on the host automindX runs on. Runs the OFFICIAL
// installer only (https://ollama.com/install.sh). Requires an explicit confirm
// token so it can't be triggered incidentally. Linux/macOS.
import { spawn } from 'node:child_process';

export const runtime = 'nodejs';
const OLLAMA = process.env.OLLAMA_HOST || 'http://localhost:11434';

async function ollamaUp(): Promise<boolean> {
  try {
    const r = await fetch(`${OLLAMA}/api/tags`, { signal: AbortSignal.timeout(2000) });
    return r.ok;
  } catch {
    return false;
  }
}

const INSTALL_TIMEOUT_MS = 5 * 60_000; // hard cap; kill a hung installer

export async function POST(req: Request) {
  // Opt-out for shared/hardened deployments — this endpoint runs a shell command.
  if (process.env.CODEPHREAK_DISABLE_INSTALL === '1') {
    return Response.json({ error: 'install endpoint disabled (CODEPHREAK_DISABLE_INSTALL=1)' }, { status: 403 });
  }
  const { confirm } = await req.json().catch(() => ({}));
  if (confirm !== 'install-ollama') {
    return Response.json({ error: 'confirmation required' }, { status: 400 });
  }
  if (await ollamaUp()) {
    return Response.json({ ok: true, already: true, note: 'Ollama is already running.' });
  }
  if (process.platform === 'win32') {
    return Response.json({
      ok: false,
      manual: true,
      note: 'On Windows, download the installer from https://ollama.com/download',
    });
  }
  // Run the OFFICIAL install script only, capped output, killed on timeout.
  return await new Promise<Response>((resolve) => {
    const child = spawn('sh', ['-c', 'curl -fsSL https://ollama.com/install.sh | sh'], {
      stdio: ['ignore', 'pipe', 'pipe'],
    });
    let out = '';
    const cap = (d: Buffer) => { if (out.length < 64_000) out += d.toString(); };
    child.stdout.on('data', cap);
    child.stderr.on('data', cap);
    const timer = setTimeout(() => { try { child.kill('SIGKILL'); } catch {} }, INSTALL_TIMEOUT_MS);
    child.on('close', (code) => {
      clearTimeout(timer);
      resolve(Response.json({
        ok: code === 0,
        code,
        log: out.slice(-2000),
        note: code === 0
          ? 'Ollama installed. Start it with `ollama serve`, then reload.'
          : 'Install failed or timed out — see log, or install manually from https://ollama.com/download',
      }));
    });
    child.on('error', (e) => { clearTimeout(timer); resolve(Response.json({ ok: false, error: String(e) }, { status: 500 })); });
  });
}
