# Professor Codephreak — AI SDK Console

A cutting-edge front end for automindX built on the **Vercel AI SDK v7**, streaming
from **Ollama** with **gpt-oss:120b-cloud** as the default model. It showcases what
the AI SDK makes effortless, and drives Professor Codephreak (and a family of
sibling personas) with a self-improving feedback loop.

```bash
./scripts/run_codephreak_console.sh      # engine :5001 + console :3100
# open http://localhost:3100   (needs: ollama serve)
```

Or run the parts separately:

```bash
python3 codephreak.py                                   # self-improving engine :5001
cd codephreak-console && npm install && npm run dev     # console :3100
```

## Everything it does

- **Streaming chat** with live token streaming, a collapsible **reasoning** stream
  (gpt-oss / qwen3 / deepseek-r1 thinking), and **markdown + code** rendering.
- **Token counter** per message (prompt + completion), **tok/s**, and **latency**;
  a running session total in the header.
- **Model picker** — live local + `:cloud` models from Ollama; gpt-oss default.
- **Advanced** sampling: temperature, top-p, top-k, repeat/frequency/presence
  penalty, max tokens, seed, plus one-click **presets** (Precise / Balanced /
  Creative / Mirostat).
- **Scientific** sampling (Ollama-native, genuinely applied via `/api/chat`):
  mirostat 0/1/2, τ, η, tail-free z, typical-p, min-p, repeat-last-n, num_ctx,
  and stop sequences.
- **`.persona` creator** — seven built-in personas in the codephreak pattern
  (**Professor Codephreak**, **automindX** the automaton, **jAImla** the ML agent,
  **Savante** sAGI, **Sentinel**, **Architect**, **Mentor**) plus your own. The
  persona you edit **is** the exact system prompt sent — perfect symmetry.
- **`.history`** — local sessions: reopen, export (`.md` / `.json`), copy, delete.
- **Export & copy** anywhere; **regenerate** and **stop** on the last turn.
- **3D-depth UI** — layered elevation shadows, glassmorphism, and beveled controls.

## Self-improvement — `codephreak.py`

`codephreak.py` is a dependency-free self-improving persona engine. Each 👍/👎 in
the chat is posted to it in realtime; it contrasts the responses users liked with
the ones they didn't and synthesizes **learned directives** (e.g. *"prefer shorter
answers"*, *"do not apologize"*, *"always include a code block when code is
requested"*). Those directives are appended to the base persona, so the system
prompt improves over time.

```
Chat ▲/▼  →  /api/feedback  →  codephreak.py  (records + learns)
.persona ⟳ sync learned  ←  /api/persona  ←  base prompt + [LEARNED FROM REALTIME FEEDBACK]
```

HTTP API (stdlib only): `GET /persona?id=<persona>`, `POST /feedback`, `GET /stats`.

## ⌘ Terminal — call Claude from your subscription

The **⌘ Terminal** button opens a real interactive terminal embedded in the dapp
(xterm.js ⇄ `pty-server.js`, a WebSocket PTY bridge bound to `127.0.0.1:3101`). Its
shell opens in the **project root**, so you can:

```
claude          # sign in with your Claude subscription (browser OAuth), then
                # drive Claude Code against automindX — from the dapp, for the dapp
```

No API key — it uses the CLI's subscription login. The launcher starts the bridge
automatically; standalone: `npm run pty` (from `codephreak-console/`). It is
local-only and never exposed off-box. This is the [aiterm](https://github.com/aiterm)
("augmented intelligence terminal") thesis realized inside the console.

**sAGI with Claude.** The headless builder is backend-agnostic too:
`python3 sagi_build.py --backend claude-cli` grows sAGI using your Claude
subscription CLI (or `--backend claude-api` with `ANTHROPIC_API_KEY`).

## Architecture note

The chat route bridges Ollama's **native** `/api/chat` into an AI SDK
`createUIMessageStream`, emitting `reasoning-*`, `text-*`, and `message-metadata`
chunks. Using the native endpoint (rather than the OpenAI-compatible `/v1`) is what
lets the **scientific** parameters actually take effect. The sibling AGLM
augmentation console lives at <https://github.com/GATERAGE/aglm>.
