import {
  createUIMessageStream,
  createUIMessageStreamResponse,
  type UIMessage,
} from 'ai';
import { CODEPHREAK_PERSONA } from '@/lib/persona';

export const maxDuration = 300;

const OLLAMA = process.env.OLLAMA_HOST || 'http://localhost:11434';
const DEFAULT_MODEL = process.env.CODEPHREAK_MODEL || 'gpt-oss:120b-cloud';

// Flatten a UIMessage's text parts.
function textOf(m: UIMessage): string {
  return (m.parts || [])
    .filter((p: any) => p.type === 'text')
    .map((p: any) => p.text)
    .join('');
}

// Only forward parameters the user actually set (skip blanks/NaN).
function cleanOptions(raw: Record<string, unknown> = {}): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  for (const [k, v] of Object.entries(raw)) {
    if (v === '' || v === null || v === undefined) continue;
    if (k === 'stop') {
      const arr = String(v).split(',').map((s) => s.trim()).filter(Boolean);
      if (arr.length) out.stop = arr;
      continue;
    }
    const n = Number(v);
    if (!Number.isNaN(n)) out[k] = n;
  }
  return out;
}

export async function POST(req: Request) {
  const body = (await req.json()) as {
    messages: UIMessage[];
    model?: string;
    system?: string;
    options?: Record<string, unknown>;
    think?: boolean;
  };

  const model = body.model || DEFAULT_MODEL;
  const system = (body.system || CODEPHREAK_PERSONA).trim();
  const options = cleanOptions(body.options);
  const think = body.think !== false;

  const ollamaMessages = [
    { role: 'system', content: system },
    ...body.messages.map((m) => ({ role: m.role, content: textOf(m) })),
  ];

  const startedAt = Date.now();

  const stream = createUIMessageStream({
    // Surface real provider errors (e.g. cloud 403 subscription) to the client.
    onError: (error) => {
      const msg = error instanceof Error ? error.message : String(error);
      if (/subscription/i.test(msg))
        return `Cloud model needs an Ollama subscription — ${msg}`;
      return msg.slice(0, 400);
    },
    execute: async ({ writer }) => {
      const res = await fetch(`${OLLAMA}/api/chat`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({
          model,
          messages: ollamaMessages,
          stream: true,
          think,
          options,
        }),
      });

      if (!res.ok || !res.body) {
        // Model isn't accessible — answer the *accessibility* question directly in
        // the response field (not as a red error), with a login/subscribe link.
        const detail = await res.text().catch(() => '');
        const upgrade = (/https?:\/\/[^\s"')]+/.exec(detail) || [])[0] || 'https://ollama.com/settings';
        let md: string;
        if (res.status === 403 && /subscription|forbidden|access/i.test(detail)) {
          md =
            `☁️ **${model}** is an Ollama **cloud** model and isn't accessible on this account yet.\n\n` +
            `**To use it:**\n` +
            `1. Sign in to Ollama Cloud — run \`ollama signin\`, or log in at [ollama.com/signin](https://ollama.com/signin)\n` +
            `2. If the model needs a plan, subscribe at [${upgrade}](${upgrade})\n\n` +
            `Meanwhile you can pick a **local** model, or the free **\`gpt-oss:120b-cloud\`**, from the model selector above.`;
        } else if (res.status === 404 || /not found|no such model/i.test(detail)) {
          md = `**${model}** isn't available locally. Pull it first: \`ollama pull ${model}\` — or choose another model from the selector.`;
        } else {
          md =
            `⚠️ **${model}** isn't reachable right now (Ollama HTTP ${res.status}).\n\n` +
            `${detail.slice(0, 200)}\n\nCheck that \`ollama serve\` is running, then try again.`;
        }
        writer.write({ type: 'text-start', id: 't0' } as any);
        writer.write({ type: 'text-delta', id: 't0', delta: md } as any);
        writer.write({ type: 'text-end', id: 't0' } as any);
        writer.write({
          type: 'message-metadata',
          messageMetadata: { model, accessible: false, status: res.status },
        } as any);
        return;
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      let reasoningOpen = false;
      let textOpen = false;
      let firstTokenAt = 0;

      const ensureReasoning = () => {
        if (!reasoningOpen) {
          writer.write({ type: 'reasoning-start', id: 'r0' } as any);
          reasoningOpen = true;
        }
      };
      const ensureText = () => {
        if (reasoningOpen) {
          writer.write({ type: 'reasoning-end', id: 'r0' } as any);
          reasoningOpen = false;
        }
        if (!textOpen) {
          writer.write({ type: 'text-start', id: 't0' } as any);
          textOpen = true;
        }
      };

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';
        for (const line of lines) {
          const t = line.trim();
          if (!t) continue;
          let obj: any;
          try {
            obj = JSON.parse(t);
          } catch {
            continue;
          }
          if (obj.error) throw new Error(String(obj.error));
          const msg = obj.message || {};
          if (msg.thinking) {
            ensureReasoning();
            if (!firstTokenAt) firstTokenAt = Date.now();
            writer.write({ type: 'reasoning-delta', id: 'r0', delta: msg.thinking } as any);
          }
          if (msg.content) {
            ensureText();
            if (!firstTokenAt) firstTokenAt = Date.now();
            writer.write({ type: 'text-delta', id: 't0', delta: msg.content } as any);
          }
          if (obj.done) {
            if (reasoningOpen) writer.write({ type: 'reasoning-end', id: 'r0' } as any);
            if (textOpen) writer.write({ type: 'text-end', id: 't0' } as any);
            const evalNs = obj.eval_duration || 0;
            const completion = obj.eval_count || 0;
            const prompt = obj.prompt_eval_count || 0;
            writer.write({
              type: 'message-metadata',
              messageMetadata: {
                model,
                inputTokens: prompt,
                outputTokens: completion,
                totalTokens: prompt + completion,
                tokPerSec: evalNs ? Math.round((completion / (evalNs / 1e9)) * 10) / 10 : null,
                latencyMs: firstTokenAt ? firstTokenAt - startedAt : null,
                totalMs: Date.now() - startedAt,
              },
            } as any);
          }
        }
      }
    },
  });

  return createUIMessageStreamResponse({ stream });
}
