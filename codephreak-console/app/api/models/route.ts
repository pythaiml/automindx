// Live Ollama model list (local + cloud), proxied server-side to avoid CORS.
const OLLAMA = process.env.OLLAMA_HOST || 'http://localhost:11434';

export async function GET() {
  try {
    const r = await fetch(`${OLLAMA}/api/tags`, { cache: 'no-store' });
    if (!r.ok) throw new Error(`ollama ${r.status}`);
    const data = await r.json();
    const local: any[] = [];
    const cloud: any[] = [];
    for (const m of data.models || []) {
      const name: string = m.name || '';
      const isCloud = !!m.remote_host || name.endsWith(':cloud');
      const entry = {
        name,
        param_size: m.details?.parameter_size || '',
        size_gb: m.size && m.size > 1e6 ? Math.round(m.size / 1e8) / 10 : null,
        family: m.details?.family || '',
      };
      (isCloud ? cloud : local).push(entry);
    }
    local.sort((a, b) => (a.size_gb || 0) - (b.size_gb || 0));
    return Response.json({ ok: true, local, cloud, host: OLLAMA });
  } catch (e: any) {
    return Response.json({ ok: false, error: String(e?.message || e), local: [], cloud: [] });
  }
}
