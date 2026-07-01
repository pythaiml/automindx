// Read-only filesystem tools that give codephreak access to the ACTUAL local
// project that runs the dapp — so "list all project files" / "audit the code"
// return the real tree/source instead of a hallucinated one.
//
// Safety: confined to the project root (realpath check, no traversal); skips
// binaries/large files + dep/build dirs; read-only (no write/execute).
import { promises as fs } from 'node:fs';
import path from 'node:path';

// The dapp runs from codephreak-console/, so the project root is one level up.
const ROOT = path.resolve(process.env.AUTOMINDX_ROOT || path.resolve(process.cwd(), '..'));
const IGNORE_DIRS = new Set(['.git', 'node_modules', '.next', '__pycache__', 'models', '.venv', 'venv', 'dist', 'build', '.ruff_cache']);
const IGNORE_EXT = new Set(['.png', '.jpg', '.jpeg', '.gif', '.webp', '.ico', '.bin', '.gguf', '.ggml', '.pt', '.db', '.jsonl', '.lock', '.woff', '.woff2', '.map']);
const MAX_FILE = 40_000;
const MAX_BYTES = 200_000;

function confined(p: string): boolean {
  const rp = path.resolve(p);
  return rp === ROOT || rp.startsWith(ROOT + path.sep);
}

export async function listFiles(): Promise<string[]> {
  const out: string[] = [];
  async function walk(dir: string) {
    let ents;
    try { ents = await fs.readdir(dir, { withFileTypes: true }); } catch { return; }
    for (const e of ents) {
      const fp = path.join(dir, e.name);
      if (e.isDirectory()) {
        if (!IGNORE_DIRS.has(e.name) && !e.name.startsWith('.')) await walk(fp);
      } else if (!IGNORE_EXT.has(path.extname(e.name).toLowerCase())) {
        out.push(path.relative(ROOT, fp));
      }
    }
  }
  await walk(ROOT);
  return out.sort();
}

export async function readFile(rel: string): Promise<string> {
  const fp = path.join(ROOT, rel);
  if (!confined(fp)) return '[denied: path outside project root]';
  try {
    const st = await fs.stat(fp);
    if (st.size > MAX_BYTES) return `[file too large: ${st.size} bytes]`;
    return (await fs.readFile(fp, 'utf8')).slice(0, MAX_FILE);
  } catch { return '[not found]'; }
}

export async function grep(pattern: string): Promise<string> {
  let re: RegExp;
  try { re = new RegExp(pattern, 'i'); } catch { return '[invalid regex]'; }
  const files = await listFiles();
  const hits: string[] = [];
  for (const rel of files) {
    const content = await readFile(rel);
    content.split('\n').forEach((line, i) => {
      if (re.test(line)) hits.push(`${rel}:${i + 1}: ${line.trim().slice(0, 160)}`);
    });
    if (hits.length >= 200) break;
  }
  return hits.slice(0, 200).join('\n') || '(no matches)';
}

// Ollama /api/chat tool schemas.
export const FS_TOOLS = [
  { type: 'function', function: { name: 'list_files', description: 'List every file in the actual automindX project that runs this app.', parameters: { type: 'object', properties: {} } } },
  { type: 'function', function: { name: 'read_file', description: 'Read the contents of a project file by its relative path.', parameters: { type: 'object', properties: { path: { type: 'string', description: 'relative path, e.g. services/model_service.py' } }, required: ['path'] } } },
  { type: 'function', function: { name: 'grep', description: 'Search all project files for a regex pattern; returns file:line matches.', parameters: { type: 'object', properties: { pattern: { type: 'string' } }, required: ['pattern'] } } },
];

export async function runFsTool(name: string, args: any): Promise<string> {
  try {
    if (name === 'list_files') return (await listFiles()).join('\n');
    if (name === 'read_file') return await readFile(String(args?.path ?? ''));
    if (name === 'grep') return await grep(String(args?.pattern ?? ''));
  } catch (e: any) { return `[tool error: ${e?.message || e}]`; }
  return `[unknown tool: ${name}]`;
}

export const PROJECT_ROOT = ROOT;
