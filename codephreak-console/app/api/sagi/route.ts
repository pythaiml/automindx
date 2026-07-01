// The sAGI self-build persists to the agnostic sagi/ package on disk:
//   GET  → current manifest (modules built so far)
//   POST → write one module to sagi/modules/NN-slug.md and update manifest.json
// Path-safe: filenames are slugified and confined to sagi/modules.
import { promises as fs } from 'node:fs';
import path from 'node:path';

export const runtime = 'nodejs';

// sagi/ sits at the automindx repo root, one level above the Next app cwd.
const SAGI_DIR = process.env.SAGI_DIR || path.resolve(process.cwd(), '..', 'sagi');
const MODULES_DIR = path.join(SAGI_DIR, 'modules');
const MANIFEST = path.join(SAGI_DIR, 'manifest.json');
const HISTORY_DIR = path.join(SAGI_DIR, '.history');
const HISTORY_LOG = path.join(HISTORY_DIR, 'build.jsonl');

// sAGI's read-write history: append one JSONL line per event (module/run).
async function logHistory(event: Record<string, unknown>): Promise<void> {
  try {
    await fs.mkdir(HISTORY_DIR, { recursive: true });
    await fs.appendFile(HISTORY_LOG, JSON.stringify({ ts: Math.floor(Date.now() / 1000), ...event }) + '\n', 'utf8');
  } catch { /* history is best-effort */ }
}

async function readManifest(): Promise<any> {
  try {
    return JSON.parse(await fs.readFile(MANIFEST, 'utf8'));
  } catch {
    return { name: 'sagi', modules: [] };
  }
}

function slug(title: string): string {
  return (title || 'module')
    .toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-+|-+$/g, '').slice(0, 48) || 'module';
}

// Read a specific instance's package when ?dir= is given (confined under SAGI_DIR),
// else the root sagi/ package. Lets the console show a spawned sAGI's modules.
function resolveDir(req: Request): string {
  const rel = new URL(req.url).searchParams.get('dir') || '';
  if (!rel) return SAGI_DIR;
  const p = path.resolve(SAGI_DIR, rel);
  return (p === SAGI_DIR || p.startsWith(SAGI_DIR + path.sep)) ? p : SAGI_DIR;
}

export async function GET(req: Request) {
  const dir = resolveDir(req);
  let m: any = { modules: [] };
  try { m = JSON.parse(await fs.readFile(path.join(dir, 'manifest.json'), 'utf8')); } catch { /* empty */ }
  // include each module's content so an instance sub-tab can render it
  const mods = await Promise.all((m.modules || []).map(async (x: any) => {
    let body = '';
    try { body = await fs.readFile(path.join(dir, 'modules', x.file), 'utf8'); } catch { /* missing */ }
    return { ...x, body };
  }));
  return Response.json({ ok: true, dir, name: m.name, version: m.version, modules: mods, count: mods.length });
}

export async function POST(req: Request) {
  const { step, title, body } = await req.json().catch(() => ({}));
  if (!body) return Response.json({ error: 'no body' }, { status: 400 });

  const n = String(Number(step) || 1).padStart(2, '0');
  const file = `${n}-${slug(title)}.md`;
  const dest = path.join(MODULES_DIR, file);
  // Confine strictly to sagi/modules (defence against traversal via title).
  if (path.dirname(path.resolve(dest)) !== path.resolve(MODULES_DIR)) {
    return Response.json({ error: 'invalid path' }, { status: 400 });
  }

  try {
    await fs.mkdir(MODULES_DIR, { recursive: true });
    await fs.writeFile(dest, `# ${title}\n\n${body}\n`, 'utf8');

    const manifest = await readManifest();
    manifest.modules = (manifest.modules || []).filter((x: any) => x.file !== file);
    manifest.modules.push({ step: Number(step) || manifest.modules.length + 1, title, file, ts: Date.now() });
    manifest.version = `0.0.${manifest.modules.length}`;
    await fs.writeFile(MANIFEST, JSON.stringify(manifest, null, 2) + '\n', 'utf8');
    await logHistory({ event: 'module', step: Number(step) || manifest.modules.length, title, file, backend: 'console' });

    return Response.json({ ok: true, file: `sagi/modules/${file}`, count: manifest.modules.length });
  } catch (e: any) {
    return Response.json({ ok: false, error: String(e?.message || e) }, { status: 500 });
  }
}
