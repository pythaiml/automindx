// One-click model pull on the machine automindX runs on. Streams Ollama's
// native /api/pull progress (NDJSON) straight through to the client.
const OLLAMA = process.env.OLLAMA_HOST || 'http://localhost:11434';

export async function POST(req: Request) {
  const { model } = await req.json();
  if (!model) return Response.json({ error: 'no model' }, { status: 400 });
  try {
    const r = await fetch(`${OLLAMA}/api/pull`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ model, stream: true }),
    });
    if (!r.ok || !r.body) {
      const detail = await r.text().catch(() => '');
      return Response.json({ ok: false, error: `Ollama ${r.status}: ${detail.slice(0, 200)}` }, { status: 502 });
    }
    // Pass the progress stream through unchanged.
    return new Response(r.body, {
      headers: { 'content-type': 'application/x-ndjson', 'cache-control': 'no-store' },
    });
  } catch (e: any) {
    return Response.json({ ok: false, error: 'Ollama unreachable: ' + String(e?.message || e) }, { status: 502 });
  }
}
