// Live Ollama model list (local + cloud), proxied server-side to avoid CORS.
const OLLAMA = process.env.OLLAMA_HOST || 'http://localhost:11434';

// Ollama cloud models that run on the FREE tier (no subscription). Verified by
// probe; the gpt-oss family is the free set. Override with FREE_CLOUD_MODELS
// (comma-separated) if Ollama's free tier changes.
const FREE_CLOUD = (process.env.FREE_CLOUD_MODELS ||
  'gpt-oss:120b-cloud,gpt-oss:20b-cloud')
  .split(',').map((s) => s.trim()).filter(Boolean);

export async function GET() {
  try {
    const r = await fetch(`${OLLAMA}/api/tags`, { cache: 'no-store' });
    if (!r.ok) throw new Error(`ollama ${r.status}`);
    const data = await r.json();
    const local: any[] = [];
    const cloud: any[] = [];
    const seen = new Set<string>();
    for (const m of data.models || []) {
      const name: string = m.name || '';
      seen.add(name);
      const isCloud = !!m.remote_host || name.endsWith(':cloud');
      const entry = {
        name,
        param_size: m.details?.parameter_size || '',
        size_gb: m.size && m.size > 1e6 ? Math.round(m.size / 1e8) / 10 : null,
        family: m.details?.family || '',
        free: FREE_CLOUD.includes(name),
        pulled: true,
      };
      (isCloud ? cloud : local).push(entry);
    }
    // Always offer the free cloud models as options, even if not pulled yet.
    for (const name of FREE_CLOUD) {
      if (!seen.has(name)) cloud.push({ name, param_size: '', size_gb: null, family: 'gptoss', free: true, pulled: false });
    }
    local.sort((a, b) => (a.size_gb || 0) - (b.size_gb || 0));
    // Free cloud models first.
    cloud.sort((a, b) => Number(b.free) - Number(a.free) || a.name.localeCompare(b.name));
    return Response.json({ ok: true, local, cloud, free: FREE_CLOUD, host: OLLAMA });
  } catch (e: any) {
    return Response.json({ ok: false, error: String(e?.message || e), local: [], cloud: [] });
  }
}
