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

export async function POST(req: Request) {
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
  // Run the official install script and capture output.
  return await new Promise<Response>((resolve) => {
    const child = spawn('sh', ['-c', 'curl -fsSL https://ollama.com/install.sh | sh'], {
      stdio: ['ignore', 'pipe', 'pipe'],
    });
    let out = '';
    child.stdout.on('data', (d) => (out += d.toString()));
    child.stderr.on('data', (d) => (out += d.toString()));
    child.on('close', (code) => {
      resolve(Response.json({
        ok: code === 0,
        code,
        log: out.slice(-2000),
        note: code === 0
          ? 'Ollama installed. Start it with `ollama serve`, then reload.'
          : 'Install failed — see log, or install manually from https://ollama.com/download',
      }));
    });
    child.on('error', (e) => resolve(Response.json({ ok: false, error: String(e) }, { status: 500 })));
  });
}
