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
const MAX_TOOL_ROUNDS = 6;   // tool calls before a turn is deemed "too big"
const MAX_SUBTASKS = 8;      // cap on the decomposition loop

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
    if (k === 'stop') { const a = String(v).split(',').map((s) => s.trim()).filter(Boolean); if (a.length) out.stop = a; continue; }
    const n = Number(v); if (!Number.isNaN(n)) out[k] = n;
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
  let useTools = body.fs !== false;
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
      let reasoningOpen = false, textOpen = false, firstTokenAt = 0, aborted = false;
      let totalPrompt = 0, totalCompletion = 0, lastEvalNs = 0;
      const ensureReasoning = () => { if (!reasoningOpen) { writer.write({ type: 'reasoning-start', id: 'r0' } as any); reasoningOpen = true; } };
      const ensureText = () => {
        if (reasoningOpen) { writer.write({ type: 'reasoning-end', id: 'r0' } as any); reasoningOpen = false; }
        if (!textOpen) { writer.write({ type: 'text-start', id: 't0' } as any); textOpen = true; }
      };
      const note = (s: string) => { ensureReasoning(); writer.write({ type: 'reasoning-delta', id: 'r0', delta: s } as any); };
      const say = (s: string) => { ensureText(); writer.write({ type: 'text-delta', id: 't0', delta: s } as any); };

      // One streamed turn with a bounded tool-call loop.
      // Returns 'answered' (final text produced), 'toollimit' (kept calling tools), or 'error'.
      async function turn(maxRounds: number): Promise<'answered' | 'toollimit' | 'error'> {
        for (let round = 0; round < maxRounds; round++) {
          const res = await fetch(`${OLLAMA}/api/chat`, {
            method: 'POST', headers: { 'content-type': 'application/json' },
            body: JSON.stringify({ model, messages: msgs, stream: true, think, options, ...(useTools ? { tools: FS_TOOLS } : {}) }),
          });
          if (!res.ok || !res.body) {
            const detail = await res.text().catch(() => '');
            if (useTools && /tool/i.test(detail)) { useTools = false; round--; continue; }
            const upgrade = (/https?:\/\/[^\s"')]+/.exec(detail) || [])[0] || 'https://ollama.com/settings';
            let md: string;
            if (res.status === 403 && /subscription|forbidden|access/i.test(detail))
              md = `☁️ **${model}** needs an Ollama subscription — run \`ollama signin\` or subscribe at [${upgrade}](${upgrade}); or pick the free **gpt-oss:120b-cloud**.`;
            else if (res.status === 404 || /not found|no such model/i.test(detail))
              md = `**${model}** isn't available. Pull it: \`ollama pull ${model}\`.`;
            else md = `⚠️ ${model} unreachable (HTTP ${res.status}). ${detail.slice(0, 160)}`;
            say(md);
            return 'error';
          }
          const reader = res.body.getReader(); const dec = new TextDecoder();
          let buffer = '', content = ''; const toolCalls: any[] = [];
          while (true) {
            const { done, value } = await reader.read(); if (done) break;
            buffer += dec.decode(value, { stream: true });
            const lines = buffer.split('\n'); buffer = lines.pop() || '';
            for (const line of lines) {
              const t = line.trim(); if (!t) continue;
              let o: any; try { o = JSON.parse(t); } catch { continue; }
              if (o.error) throw new Error(String(o.error));
              const m = o.message || {};
              if (m.thinking) { ensureReasoning(); if (!firstTokenAt) firstTokenAt = Date.now(); writer.write({ type: 'reasoning-delta', id: 'r0', delta: m.thinking } as any); }
              if (m.content) { ensureText(); if (!firstTokenAt) firstTokenAt = Date.now(); content += m.content; writer.write({ type: 'text-delta', id: 't0', delta: m.content } as any); }
              if (Array.isArray(m.tool_calls)) toolCalls.push(...m.tool_calls);
              if (o.done) { lastEvalNs = o.eval_duration || 0; totalCompletion += o.eval_count || 0; totalPrompt = o.prompt_eval_count || totalPrompt; }
            }
          }
          if (toolCalls.length) {
            msgs.push({ role: 'assistant', content, tool_calls: toolCalls });
            for (const tc of toolCalls) {
              const name = tc.function?.name || tc.name;
              const args = tc.function?.arguments ?? tc.arguments ?? {};
              note(`\n🔧 ${name}(${typeof args === 'string' ? args : JSON.stringify(args)}) …`);
              const out = await runFsTool(name, args);
              note(` ✓ ${out.split('\n').length} line(s)\n`);
              msgs.push({ role: 'tool', content: out.slice(0, 30000) });
            }
            continue;
          }
          msgs.push({ role: 'assistant', content });
          return 'answered';
        }
        return 'toollimit';
      }

      // Non-streamed ask (for planning), no tools.
      async function ask(prompt: string): Promise<string> {
        const r = await fetch(`${OLLAMA}/api/chat`, {
          method: 'POST', headers: { 'content-type': 'application/json' },
          body: JSON.stringify({ model, messages: [...msgs, { role: 'user', content: prompt }], stream: false, think: false, options }),
        });
        if (!r.ok) return '';
        return ((await r.json()).message?.content || '');
      }

      const first = await turn(MAX_TOOL_ROUNDS);

      if (first === 'toollimit') {
        // Too big for one turn → decompose into subtasks and complete one at a time,
        // looping exactly as many times as the model says are required.
        note('\n\n⚙️ task is large — decomposing into subtasks…\n');
        const plan = await ask(
          'This task is too large to answer in one go. Break the ORIGINAL request into a short ordered ' +
          'list of concrete sub-tasks that, completed in order, fully satisfy it. Reply with ONLY a numbered ' +
          'list, one sub-task per line, no preamble.',
        );
        const subtasks = plan.split('\n').map((l) => l.replace(/^\s*\d+[.)]\s*|^[-*]\s*/, '').trim())
          .filter(Boolean).slice(0, MAX_SUBTASKS);
        if (subtasks.length) {
          msgs.push({ role: 'assistant', content: 'Plan:\n' + subtasks.map((s, i) => `${i + 1}. ${s}`).join('\n') });
          say(`\n\nThis needs **${subtasks.length} steps**. Completing them one at a time:\n`);
          for (let i = 0; i < subtasks.length && !aborted; i++) {
            say(`\n\n### Step ${i + 1}/${subtasks.length} — ${subtasks[i]}\n`);
            msgs.push({ role: 'user', content: `Complete ONLY step ${i + 1} of ${subtasks.length}: "${subtasks[i]}". Output just this step's result, concisely. Do not repeat earlier steps.` });
            const r = await turn(2);           // small tool budget per subtask
            if (r === 'error') { aborted = true; break; }
          }
          say('\n\n✅ All steps complete.');
        } else {
          say('\n\n[could not decompose the task — try narrowing the request]');
        }
      }

      if (reasoningOpen) writer.write({ type: 'reasoning-end', id: 'r0' } as any);
      if (textOpen) writer.write({ type: 'text-end', id: 't0' } as any);
      writer.write({
        type: 'message-metadata',
        messageMetadata: {
          model, inputTokens: totalPrompt, outputTokens: totalCompletion, totalTokens: totalPrompt + totalCompletion,
          tokPerSec: lastEvalNs ? Math.round((totalCompletion / (lastEvalNs / 1e9)) * 10) / 10 : null,
          latencyMs: firstTokenAt ? firstTokenAt - startedAt : null, totalMs: Date.now() - startedAt,
        },
      } as any);
    },
  });

  return createUIMessageStreamResponse({ stream });
}
