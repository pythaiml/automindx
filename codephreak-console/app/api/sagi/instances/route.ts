// GET → the spawned sAGI hierarchy on disk (lineage: sovereign + subordinate
// children, recursively), so the console can show each as a persistent sub-tab.
// These live on disk (sagi/sovereign/… · sagi/children/…) — runtime state that
// persists across sessions even though it's git-ignored.
import { promises as fs } from 'node:fs';
import path from 'node:path';

export const runtime = 'nodejs';

const SAGI_DIR = process.env.SAGI_DIR || path.resolve(process.cwd(), '..', 'sagi');

async function readIdent(dir: string): Promise<any | null> {
  try { return JSON.parse(await fs.readFile(path.join(dir, 'identity.json'), 'utf8')); } catch { return null; }
}
async function countMods(dir: string): Promise<number> {
  try { return (await fs.readdir(path.join(dir, 'modules'))).filter((f) => f.endsWith('.md')).length; } catch { return 0; }
}

async function walk(dir: string, depth: number, parentId: string, out: any[]): Promise<void> {
  for (const kind of ['sovereign', 'children']) {
    const base = path.join(dir, kind);
    let names: string[] = [];
    try { names = await fs.readdir(base); } catch { continue; }
    for (const name of names) {
      const cdir = path.join(base, name);
      const ident = await readIdent(cdir);
      if (!ident) continue;
      out.push({
        id: ident.id, name: ident.name || ident.id,
        mode: ident.mode, sovereign: !!ident.sovereign,
        dir: path.relative(SAGI_DIR, cdir), depth, parent: parentId,
        modules: await countMods(cdir), individual: ident.individual || '',
        grown_from: ident.grown_from || parentId, ts: ident.created_ts || 0,
      });
      await walk(cdir, depth + 1, ident.id, out);
    }
  }
}

export async function GET() {
  const out: any[] = [];
  try { await walk(SAGI_DIR, 0, 'sagi', out); } catch { /* none */ }
  return Response.json({ ok: true, instances: out });
}
