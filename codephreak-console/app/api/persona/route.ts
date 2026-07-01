// Pulls a self-improved persona (base + learned directives) from codephreak.py.
const ENGINE = process.env.CODEPHREAK_ENGINE || 'http://localhost:5001';

export async function GET(req: Request) {
  const id = new URL(req.url).searchParams.get('id') || 'codephreak';
  try {
    const r = await fetch(`${ENGINE}/persona?id=${encodeURIComponent(id)}`, {
      signal: AbortSignal.timeout(3000),
      cache: 'no-store',
    });
    if (!r.ok) throw new Error(String(r.status));
    return Response.json({ ok: true, ...(await r.json()) });
  } catch {
    return Response.json({ ok: false, note: 'codephreak.py engine offline', directives: [] });
  }
}
