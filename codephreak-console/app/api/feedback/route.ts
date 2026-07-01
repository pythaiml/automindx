// Forwards realtime 👍/👎 feedback to the self-improving codephreak.py engine.
// Degrades gracefully (200 with delivered:false) if the engine isn't running.
const ENGINE = process.env.CODEPHREAK_ENGINE || 'http://localhost:5001';

export async function POST(req: Request) {
  const body = await req.json();
  try {
    const r = await fetch(`${ENGINE}/feedback`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(3000),
    });
    const data = await r.json().catch(() => ({}));
    return Response.json({ delivered: r.ok, ...data });
  } catch {
    // codephreak.py not running — feedback is still acknowledged client-side.
    return Response.json({ delivered: false, note: 'codephreak.py engine offline' });
  }
}
