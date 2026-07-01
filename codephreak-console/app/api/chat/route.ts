import {
  createUIMessageStream,
  createUIMessageStreamResponse,
  type UIMessage,
} from 'ai';
import { CODEPHREAK_PERSONA } from '@/lib/persona';
import { FS_TOOLS, runFsTool } from '@/lib/fs-tools';

export const maxDuration = 300;

const OLLAMA = process.env.OLLAMA_HOST || 'http://localhost:11434';
const DEFAULT_MODEL = process.env.CODEPHREAK_MODEL || 'gpt-oss:120b-cloud';
const MAX_TOOL_ROUNDS = 6;

const TOOL_NOTE =
  '\n\n[FILESYSTEM ACCESS] You can read the ACTUAL project that runs this app. ' +
  'Call list_files to list every file, read_file(path) to read one, grep(pattern) to search. ' +
  'When asked about the code, files, architecture, or to audit — USE these tools and answer ' +
  'strictly from the real files. Never invent or guess the project structure.';

function textOf(m: UIMessage): string {
  return (m.parts || []).filter((p: any) => p.type === 'text').map((p: any) => p.text).join('');
}

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
    messages: UIMessage[]; model?: string; system?: string;
    options?: Record<string, unknown>; think?: boolean; fs?: boolean;
  };

  const model = body.model || DEFAULT_MODEL;
  const options = cleanOptions(body.options);
  const think = body.think !== false;
  let useTools = body.fs !== false; // filesystem tools on by default
  const system = (body.system || CODEPHREAK_PERSONA).trim() + (useTools ? TOOL_NOTE : '');

  const msgs: any[] = [
    { role: 'system', content: system },
    ...body.messages.map((m) => ({ role: m.role, content: textOf(m) })),
  ];

  const startedAt = Date.now();

  const stream = createUIMessageStream({
    onError: (error) => {
      const m = error instanceof Error ? error.message : String(error);
      if (/subscription/i.test(m)) return `Cloud model needs an Ollama subscription — ${m}`;
      return m.slice(0, 400);
    },
    execute: async ({ writer }) => {
      let reasoningOpen = false, textOpen = false, firstTokenAt = 0;
      const ensureReasoning = () => { if (!reasoningOpen) { writer.write({ type: 'reasoning-start', id: 'r0' } as any); reasoningOpen = true; } };
      const ensureText = () => {
        if (reasoningOpen) { writer.write({ type: 'reasoning-end', id: 'r0' } as any); reasoningOpen = false; }
        if (!textOpen) { writer.write({ type: 'text-start', id: 't0' } as any); textOpen = true; }
      };
      const note = (s: string) => { ensureReasoning(); writer.write({ type: 'reasoning-delta', id: 'r0', delta: s } as any); };

      for (let round = 0; round < MAX_TOOL_ROUNDS; round++) {
        const res = await fetch(`${OLLAMA}/api/chat`, {
          method: 'POST', headers: { 'content-type': 'application/json' },
          body: JSON.stringify({ model, messages: msgs, stream: true, think, options, ...(useTools ? { tools: FS_TOOLS } : {}) }),
        });

        if (!res.ok || !res.body) {
          const detail = await res.text().catch(() => '');
          // Model doesn't support tools → retry this round without them.
          if (useTools && /tool/i.test(detail)) { useTools = false; round--; continue; }
          const upgrade = (/https?:\/\/[^\s"')]+/.exec(detail) || [])[0] || 'https://ollama.com/settings';
          let md: string;
          if (res.status === 403 && /subscription|forbidden|access/i.test(detail)) {
            md = `☁️ **${model}** is an Ollama **cloud** model and isn't accessible on this account yet.\n\n**To use it:**\n1. Sign in to Ollama Cloud — run \`ollama signin\`, or log in at [ollama.com/signin](https://ollama.com/signin)\n2. If the model needs a plan, subscribe at [${upgrade}](${upgrade})\n\nMeanwhile pick a **local** model, or the free **\`gpt-oss:120b-cloud\`**.`;
          } else if (res.status === 404 || /not found|no such model/i.test(detail)) {
            md = `**${model}** isn't available locally. Pull it first: \`ollama pull ${model}\`.`;
          } else {
            md = `⚠️ **${model}** isn't reachable (Ollama HTTP ${res.status}).\n\n${detail.slice(0, 200)}\n\nIs \`ollama serve\` running?`;
          }
          ensureText();
          writer.write({ type: 'text-delta', id: 't0', delta: md } as any);
          writer.write({ type: 'text-end', id: 't0' } as any);
          writer.write({ type: 'message-metadata', messageMetadata: { model, accessible: false, status: res.status } } as any);
          return;
        }

        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '', content = '', evalNs = 0, completion = 0, prompt = 0;
        const toolCalls: any[] = [];

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split('\n'); buffer = lines.pop() || '';
          for (const line of lines) {
            const t = line.trim(); if (!t) continue;
            let obj: any; try { obj = JSON.parse(t); } catch { continue; }
            if (obj.error) throw new Error(String(obj.error));
            const m = obj.message || {};
            if (m.thinking) { ensureReasoning(); if (!firstTokenAt) firstTokenAt = Date.now(); writer.write({ type: 'reasoning-delta', id: 'r0', delta: m.thinking } as any); }
            if (m.content) { ensureText(); if (!firstTokenAt) firstTokenAt = Date.now(); content += m.content; writer.write({ type: 'text-delta', id: 't0', delta: m.content } as any); }
            if (Array.isArray(m.tool_calls)) toolCalls.push(...m.tool_calls);
            if (obj.done) { evalNs = obj.eval_duration || 0; completion = obj.eval_count || 0; prompt = obj.prompt_eval_count || 0; }
          }
        }

        if (toolCalls.length) {
          // codephreak wants to read the real filesystem — execute the tools.
          msgs.push({ role: 'assistant', content, tool_calls: toolCalls });
          for (const tc of toolCalls) {
            const name = tc.function?.name || tc.name;
            const args = tc.function?.arguments ?? tc.arguments ?? {};
            note(`\n🔧 ${name}(${typeof args === 'string' ? args : JSON.stringify(args)}) …`);
            const out = await runFsTool(name, args);
            note(` ✓ ${out.split('\n').length} line(s)\n`);
            msgs.push({ role: 'tool', content: out.slice(0, 30000) });
          }
          continue; // next round: model answers grounded in the tool results
        }

        // Final answer — close streams + emit exact token usage.
        if (reasoningOpen) writer.write({ type: 'reasoning-end', id: 'r0' } as any);
        if (textOpen) writer.write({ type: 'text-end', id: 't0' } as any);
        writer.write({
          type: 'message-metadata',
          messageMetadata: {
            model, inputTokens: prompt, outputTokens: completion, totalTokens: prompt + completion,
            tokPerSec: evalNs ? Math.round((completion / (evalNs / 1e9)) * 10) / 10 : null,
            latencyMs: firstTokenAt ? firstTokenAt - startedAt : null, totalMs: Date.now() - startedAt,
          },
        } as any);
        return;
      }
      // Exceeded tool rounds.
      if (!textOpen) writer.write({ type: 'text-start', id: 't0' } as any);
      writer.write({ type: 'text-delta', id: 't0', delta: '\n\n[reached the tool-call limit]' } as any);
      writer.write({ type: 'text-end', id: 't0' } as any);
    },
  });

  return createUIMessageStreamResponse({ stream });
}
