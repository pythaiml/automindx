// GET → recent entries from sAGI's read-write .history log (sagi/.history/build.jsonl).
// Every module the sandbox grows and every build run is recorded here, so the
// self-build can be watched live and replayed. Returns newest-last.
import { promises as fs } from 'node:fs';
import path from 'node:path';

export const runtime = 'nodejs';

const SAGI_DIR = process.env.SAGI_DIR || path.resolve(process.cwd(), '..', 'sagi');
const HISTORY_LOG = path.join(SAGI_DIR, '.history', 'build.jsonl');

export async function GET(req: Request) {
  const limit = Math.min(500, Number(new URL(req.url).searchParams.get('limit')) || 120);
  try {
    const raw = await fs.readFile(HISTORY_LOG, 'utf8');
    const entries = raw.split('\n').filter(Boolean).slice(-limit).map((l) => {
      try { return JSON.parse(l); } catch { return null; }
    }).filter(Boolean);
    return Response.json({ ok: true, entries });
  } catch {
    return Response.json({ ok: true, entries: [] });   // no history yet
  }
}
