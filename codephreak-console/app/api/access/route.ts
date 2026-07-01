// Account-aware cloud access. Probes whether the *current Ollama account* can
// actually run a model (a 1-token generation) and caches the verdict, so the
// free/gated split reflects the signed-in account at runtime — not a static list.
const OLLAMA = process.env.OLLAMA_HOST || 'http://localhost:11434';
const TTL_MS = 5 * 60_000;

type Verdict = { accessible: boolean; reason: 'ok' | 'subscription' | 'not-pulled' | 'error'; detail?: string };
const cache = new Map<string, { at: number; v: Verdict }>();

async function probe(model: string): Promise<Verdict> {
  const hit = cache.get(model);
  if (hit && Date.now() - hit.at < TTL_MS) return hit.v;
  let v: Verdict;
  try {
    const r = await fetch(`${OLLAMA}/v1/chat/completions`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ model, messages: [{ role: 'user', content: 'hi' }], max_tokens: 1 }),
      signal: AbortSignal.timeout(30_000),
    });
    if (r.ok) v = { accessible: true, reason: 'ok' };
    else {
      const t = await r.text().catch(() => '');
      if (r.status === 403 || /subscription/i.test(t)) v = { accessible: false, reason: 'subscription', detail: t.slice(0, 160) };
      else if (r.status === 404 || /not found/i.test(t)) v = { accessible: false, reason: 'not-pulled', detail: t.slice(0, 160) };
      else v = { accessible: false, reason: 'error', detail: `HTTP ${r.status}` };
    }
  } catch (e: any) {
    v = { accessible: false, reason: 'error', detail: String(e?.message || e) };
  }
  cache.set(model, { at: Date.now(), v });
  return v;
}

export async function POST(req: Request) {
  const { models } = await req.json().catch(() => ({ models: [] }));
  const list: string[] = Array.isArray(models) ? models.slice(0, 24) : [];
  const results: Record<string, Verdict> = {};
  await Promise.all(list.map(async (m) => { results[m] = await probe(m); }));
  return Response.json({ ok: true, results });
}

export async function GET(req: Request) {
  const model = new URL(req.url).searchParams.get('model');
  if (!model) return Response.json({ error: 'no model' }, { status: 400 });
  return Response.json({ model, ...(await probe(model)) });
}
