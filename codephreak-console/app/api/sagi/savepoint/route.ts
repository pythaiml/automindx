// POST → take a sAGI "point of save": snapshots the current modules into a gitmind
// GLOBAL commit, logs a point_of_save to .history, embeds into RAGE, and writes a
// shareable Markdown export. Returns the commit + the shareable text (for copy).
import { promises as fs } from 'node:fs';
import path from 'node:path';
import { execFile } from 'node:child_process';

export const runtime = 'nodejs';
export const maxDuration = 120;

const REPO = path.resolve(process.cwd(), '..');          // console runs in codephreak-console/
const SAGI_DIR = process.env.SAGI_DIR || path.join(REPO, 'sagi');

export async function POST(req: Request) {
  const { label = '' } = await req.json().catch(() => ({} as any));
  const res: any = await new Promise((resolve) => {
    execFile('python3', ['-m', 'sagi.runtime.savepoint', '--dir', SAGI_DIR, '--label', String(label).slice(0, 120), '--json'],
      { cwd: REPO, timeout: 100_000 },
      (err, stdout, stderr) => {
        if (err) return resolve({ error: (stderr || err.message).slice(0, 300) });
        try { resolve(JSON.parse(stdout.trim().split('\n').pop() || '{}')); }
        catch { resolve({ error: 'unparseable output: ' + stdout.slice(0, 200) }); }
      });
  });
  if (res.error) return Response.json({ ok: false, ...res }, { status: 500 });
  let text = '';
  try { text = await fs.readFile(res.export, 'utf8'); } catch { /* export unreadable */ }
  return Response.json({ ok: true, ...res, text });
}
