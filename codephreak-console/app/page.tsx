'use client';

import { useChat } from '@ai-sdk/react';
import { useEffect, useRef, useState, type ReactNode } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { BUILTIN_PERSONAS, CODEPHREAK_PERSONA, type Persona } from '@/lib/persona';

const BUILTIN_IDS = new Set(BUILTIN_PERSONAS.map((p) => p.id));

// ─────────────────────────────────────────────────────────────────────────
// Single source of truth. The persona you edit here IS the system prompt sent
// to the model, and the settings you set here ARE the options sent to Ollama —
// perfect symmetry between the prompt and the code that drives it.
// ─────────────────────────────────────────────────────────────────────────

type Model = { name: string; param_size?: string; size_gb?: number | null };
type Models = { ok: boolean; local: Model[]; cloud: Model[]; host?: string; error?: string };
type Session = { id: string; ts: number; title: string; model: string; messages: any[] };
type Meta = { model?: string; inputTokens?: number; outputTokens?: number; totalTokens?: number; tokPerSec?: number | null; latencyMs?: number | null; totalMs?: number | null };

const DEFAULT_MODEL = 'gpt-oss:120b-cloud';

// Advanced (OpenAI-style) + Scientific (Ollama-native) generation state.
const ADVANCED = {
  temperature: 0.7, top_p: 0.9, top_k: 40, num_predict: 2048,
  repeat_penalty: 1.1, presence_penalty: 0, frequency_penalty: 0, seed: '' as number | '',
};
const SCIENTIFIC = {
  mirostat: 0, mirostat_tau: 5.0, mirostat_eta: 0.1,
  tfs_z: 1.0, typical_p: 1.0, min_p: 0.0, repeat_last_n: 64, num_ctx: 4096, stop: '',
};
type Settings = typeof ADVANCED & typeof SCIENTIFIC;
const DEFAULTS: Settings = { ...ADVANCED, ...SCIENTIFIC };

const PRESETS: Record<string, Partial<Settings>> = {
  Precise: { temperature: 0.2, top_p: 0.7, top_k: 20, mirostat: 0, repeat_penalty: 1.15 },
  Balanced: { temperature: 0.7, top_p: 0.9, top_k: 40, mirostat: 0, repeat_penalty: 1.1 },
  Creative: { temperature: 1.1, top_p: 0.97, top_k: 100, mirostat: 0, repeat_penalty: 1.05 },
  Mirostat: { temperature: 0.8, mirostat: 2, mirostat_tau: 5.0, mirostat_eta: 0.1 },
};

const LS = { personas: 'cpk.personas', active: 'cpk.activePersona', settings: 'cpk.settings', history: 'cpk.history', think: 'cpk.think' };

const uid = () => 'id_' + Date.now().toString(36) + Math.floor(performance.now() % 1e6).toString(36);
const textOf = (m: any) => (m?.parts || []).filter((p: any) => p.type === 'text').map((p: any) => p.text).join('');
const reasonOf = (m: any) => (m?.parts || []).filter((p: any) => p.type === 'reasoning').map((p: any) => p.text || p.delta || '').join('');
const metaOf = (m: any): Meta => (m?.metadata as Meta) || {};

export default function Console() {
  const [tab, setTab] = useState('chat');
  const [models, setModels] = useState<Models | null>(null);
  const [model, setModel] = useState(DEFAULT_MODEL);
  const [think, setThink] = useState(true);
  const [settings, setSettings] = useState<Settings>(DEFAULTS);
  const [input, setInput] = useState('');

  const [personas, setPersonas] = useState<Persona[]>(BUILTIN_PERSONAS);
  const [activePersona, setActivePersona] = useState('codephreak');
  const [fb, setFb] = useState<Record<string, 'up' | 'down'>>({});

  const [sessions, setSessions] = useState<Session[]>([]);
  const activeSession = useRef<string | null>(null);

  const { messages, sendMessage, status, stop, setMessages, error } = useChat();
  const logRef = useRef<HTMLDivElement>(null);
  const busy = status === 'streaming' || status === 'submitted';

  const persona = personas.find((p) => p.id === activePersona) || personas[0];

  // boot: restore persisted state + poll models
  useEffect(() => {
    try {
      const p = localStorage.getItem(LS.personas);
      if (p) {
        const saved: Persona[] = JSON.parse(p);
        // merge: keep saved edits for built-ins, ensure all built-ins exist, keep customs
        const merged = [
          ...BUILTIN_PERSONAS.map((b) => saved.find((s) => s.id === b.id) || b),
          ...saved.filter((s) => !BUILTIN_IDS.has(s.id)),
        ];
        setPersonas(merged);
      }
      const a = localStorage.getItem(LS.active); if (a) setActivePersona(a);
      const s = localStorage.getItem(LS.settings); if (s) setSettings({ ...DEFAULTS, ...JSON.parse(s) });
      const h = localStorage.getItem(LS.history); if (h) setSessions(JSON.parse(h));
      const t = localStorage.getItem(LS.think); if (t) setThink(t === '1');
    } catch {}
    loadModels(); const iv = setInterval(loadModels, 8000); return () => clearInterval(iv);
  }, []);
  useEffect(() => { localStorage.setItem(LS.settings, JSON.stringify(settings)); }, [settings]);
  useEffect(() => { localStorage.setItem(LS.personas, JSON.stringify(personas)); }, [personas]);
  useEffect(() => { localStorage.setItem(LS.active, activePersona); }, [activePersona]);
  useEffect(() => { localStorage.setItem(LS.think, think ? '1' : '0'); }, [think]);
  useEffect(() => { logRef.current?.scrollTo({ top: logRef.current.scrollHeight, behavior: 'smooth' }); }, [messages, status]);
  useEffect(() => { if (status === 'ready' && messages.length) saveSession(messages); /* eslint-disable-next-line */ }, [status]);

  async function loadModels() {
    try { const r = await fetch('/api/models'); const j: Models = await r.json(); setModels(j);
      if (j.ok && ![...j.local, ...j.cloud].some((m) => m.name === model) && [...j.local, ...j.cloud].some((m) => m.name === DEFAULT_MODEL)) setModel(DEFAULT_MODEL);
    } catch { setModels({ ok: false, local: [], cloud: [], error: 'offline' }); }
  }

  // build the Ollama options object from current settings (blanks dropped server-side)
  function buildOptions(): Record<string, unknown> {
    const s: any = settings;
    const o: Record<string, unknown> = {};
    const keys = ['temperature','top_p','top_k','num_predict','repeat_penalty','presence_penalty','frequency_penalty','seed','mirostat','mirostat_tau','mirostat_eta','tfs_z','typical_p','min_p','repeat_last_n','num_ctx','stop'];
    for (const k of keys) if (s[k] !== '' && s[k] != null) o[k] = s[k];
    return o;
  }

  function submit(text: string) {
    if (!text.trim() || busy) return;
    sendMessage({ text }, { body: { model, system: persona.prompt, options: buildOptions(), think } });
  }
  function onSend() { const t = input.trim(); if (!t) return; setInput(''); submit(t); }
  function regenerate() {
    if (busy || !messages.length) return;
    const lastUser = [...messages].reverse().find((m) => m.role === 'user');
    if (!lastUser) return;
    // drop trailing assistant turn(s), then resend the last user text — perfect re-derivation
    let trimmed = [...messages];
    while (trimmed.length && trimmed[trimmed.length - 1].role === 'assistant') trimmed.pop();
    if (trimmed.length && trimmed[trimmed.length - 1].role === 'user') trimmed.pop();
    setMessages(trimmed);
    setTimeout(() => submit(textOf(lastUser)), 20);
  }

  // history
  function persistSessions(next: Session[]) { setSessions(next); localStorage.setItem(LS.history, JSON.stringify(next.slice(0, 100))); }
  function saveSession(msgs: any[]) {
    const firstUser = msgs.find((m) => m.role === 'user');
    const title = firstUser ? textOf(firstUser).slice(0, 80) : 'Conversation';
    setSessions((prev) => {
      let id = activeSession.current; const next = [...prev];
      if (!id) { id = uid(); activeSession.current = id; next.unshift({ id, ts: Date.now(), title, model, messages: msgs }); }
      else { const i = next.findIndex((x) => x.id === id); if (i >= 0) next[i] = { ...next[i], ts: Date.now(), title, model, messages: msgs }; else next.unshift({ id, ts: Date.now(), title, model, messages: msgs }); }
      localStorage.setItem(LS.history, JSON.stringify(next.slice(0, 100))); return next;
    });
  }
  function newChat() { activeSession.current = null; setMessages([]); setTab('chat'); }
  function openSession(s: Session) { activeSession.current = s.id; setMessages(s.messages); setModel(s.model); setTab('chat'); }

  // export / copy
  function toMarkdown(msgs: any[]) {
    return msgs.map((m) => `### ${m.role === 'user' ? 'You' : 'codephreak'}\n\n${textOf(m)}`).join('\n\n---\n\n');
  }
  function download(name: string, text: string, type = 'text/markdown') {
    const url = URL.createObjectURL(new Blob([text], { type }));
    const a = document.createElement('a'); a.href = url; a.download = name; a.click(); URL.revokeObjectURL(url);
  }
  const [copied, setCopied] = useState<string | null>(null);
  async function copy(text: string, key: string) { try { await navigator.clipboard.writeText(text); setCopied(key); setTimeout(() => setCopied(null), 1200); } catch {} }

  // Realtime feedback → self-improving codephreak.py engine (via /api/feedback).
  async function feedback(msg: any, rating: 'up' | 'down') {
    setFb((f) => ({ ...f, [msg.id]: rating }));
    const i = messages.findIndex((m) => m.id === msg.id);
    const prompt = i > 0 ? textOf(messages[i - 1]) : '';
    try {
      await fetch('/api/feedback', {
        method: 'POST', headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ persona: persona.id, model, prompt, response: textOf(msg), rating }),
      });
    } catch {}
  }

  // Pull codephreak.py's learned improvements into the active persona.
  const [learned, setLearned] = useState<string[] | null>(null);
  async function syncLearned() {
    try {
      const r = await fetch('/api/persona?id=' + encodeURIComponent(persona.id));
      const j = await r.json();
      setLearned(j.directives || []);
      if (j.ok && j.prompt) setPersonas((ps) => ps.map((p) => p.id === persona.id ? { ...p, prompt: j.prompt } : p));
    } catch { setLearned([]); }
  }

  const live = !!models?.ok;
  const nModels = (models?.local.length || 0) + (models?.cloud.length || 0);
  const isCloud = (n: string) => models?.cloud.some((m) => m.name === n);
  const sessionTokens = messages.reduce((a, m) => a + (metaOf(m).totalTokens || 0), 0);
  const lastAId = [...messages].reverse().find((m) => m.role === 'assistant')?.id;

  const SUGGEST = [
    'Write a production-ready Python rate limiter with tests.',
    'Explain the BDI agent model, step by step.',
    'Design a secure JWT auth flow — modular and concise.',
  ];

  return (
    <div className="app">
      <aside className="side">
        <div className="brand">
          <div className="logo">codephreak</div>
          <div className="sub">automindX · AI SDK</div>
        </div>
        <nav className="nav">
          <div className="grp">Session</div>
          <Nav id="chat" tab={tab} set={setTab} ico="◈" label="Chat" />
          <Nav id="history" tab={tab} set={setTab} ico="▤" label=".history" />
          <div className="grp">Tuning</div>
          <Nav id="advanced" tab={tab} set={setTab} ico="⚙" label="Advanced" />
          <Nav id="scientific" tab={tab} set={setTab} ico="⚛" label="Scientific" />
          <Nav id="persona" tab={tab} set={setTab} ico="☰" label=".persona" />
          <Nav id="about" tab={tab} set={setTab} ico="ⓘ" label="About" />
          <div style={{ padding: '14px 8px' }}>
            <button className="btn ghost sm" style={{ width: '100%' }} onClick={newChat}>+ New chat</button>
          </div>
        </nav>
        <div className="foot">Professor Codephreak<br />automindX · gpt-oss default</div>
      </aside>

      <main className="main">
        <div className="topbar">
          <span className="pill"><span className={'dot ' + (live ? 'ok pulse' : 'bad')} />{live ? `Ollama · ${nModels}` : 'Ollama offline'}</span>
          <span className="dim small">model</span>
          <select value={model} onChange={(e) => setModel(e.target.value)} style={{ minWidth: 210 }}>
            {models?.cloud.length ? <optgroup label="cloud">{models.cloud.map((m) => <option key={m.name} value={m.name}>☁ {m.name}{m.param_size ? ` · ${m.param_size}` : ''}</option>)}</optgroup> : null}
            {models?.local.length ? <optgroup label="local">{models.local.map((m) => <option key={m.name} value={m.name}>▣ {m.name}{m.param_size ? ` · ${m.param_size}` : ''}</option>)}</optgroup> : null}
            {!models || (!models.local.length && !models.cloud.length) ? <option value={model}>{model}</option> : null}
          </select>
          <span className={'tag ' + (isCloud(model) ? 'cloud' : 'local')}>{isCloud(model) ? 'cloud' : 'local'}</span>
          <div className="spacer" />
          <div className={'toggle' + (think ? ' on' : '')} onClick={() => setThink((v) => !v)} title="Show the model's reasoning stream"><span className="switch" /><span className="small">reasoning</span></div>
          <span className="pill" title="tokens this conversation">🪙 {sessionTokens.toLocaleString()}</span>
        </div>

        <div className="content">
          {/* CHAT */}
          <section className={'view' + (tab === 'chat' ? ' active' : '')}>
            <div className="chatwrap">
              <div className="chatlog" ref={logRef}>
                {messages.length === 0 && (
                  <div className="empty">
                    <p>Ask <b style={{ color: 'var(--accent)' }}>codephreak</b> anything — streaming from <b className="mono">{model}</b>.</p>
                    <div className="row" style={{ justifyContent: 'center', marginTop: 12 }}>
                      {SUGGEST.map((s) => <span key={s} className="chip" onClick={() => submit(s)}>{s}</span>)}
                    </div>
                  </div>
                )}
                {messages.map((m) => {
                  const reasoning = reasonOf(m); const body = textOf(m); const meta = metaOf(m);
                  const isLast = m.id === lastAId;
                  return (
                    <div key={m.id} className={'msg ' + m.role}>
                      <div className="av">{m.role === 'user' ? 'YOU' : 'cpk'}</div>
                      <div className="bubble">
                        {m.role === 'assistant' && reasoning && (
                          <details className="think" open={!body}>
                            <summary>{body ? '⚛ thought process' : <span><span className="spin" /> thinking…</span>}</summary>
                            <div className="thinktext">{reasoning}</div>
                          </details>
                        )}
                        {m.role === 'assistant'
                          ? (body ? <ReactMarkdown remarkPlugins={[remarkGfm]}>{body}</ReactMarkdown> : (busy && isLast && !reasoning ? <span className="cursor" /> : null))
                          : <div style={{ whiteSpace: 'pre-wrap' }}>{body}</div>}
                        {(body || meta.totalTokens != null) && (
                          <div className="msgmeta">
                            {meta.totalTokens != null && <span className="stat">🪙 {meta.totalTokens} tok</span>}
                            {meta.tokPerSec != null && <span className="stat">⚡ {meta.tokPerSec} tok/s</span>}
                            {meta.latencyMs != null && <span className="stat">⏱ {meta.latencyMs}ms</span>}
                            <span className="act">
                              {m.role === 'assistant' && body && (
                                <>
                                  <button className="btn ghost sm icon" title="helpful — teaches codephreak.py" onClick={() => feedback(m, 'up')} style={fb[m.id] === 'up' ? { color: 'var(--good)' } : undefined}>▲</button>
                                  <button className="btn ghost sm icon" title="not helpful — teaches codephreak.py" onClick={() => feedback(m, 'down')} style={fb[m.id] === 'down' ? { color: 'var(--bad)' } : undefined}>▼</button>
                                </>
                              )}
                              <button className="btn ghost sm icon" title="copy" onClick={() => copy(body, m.id)}>{copied === m.id ? '✓' : '⧉'}</button>
                              {m.role === 'assistant' && isLast && !busy && <button className="btn ghost sm icon" title="regenerate" onClick={regenerate}>↻</button>}
                            </span>
                          </div>
                        )}
                      </div>
                    </div>
                  );
                })}
                {error && <div className="msg assistant"><div className="av">!</div><div className="bubble" style={{ color: 'var(--bad)' }}>Error: {String(error.message || error)}. Is Ollama running and the model available?</div></div>}
              </div>

              <div className="composer">
                <textarea value={input} placeholder={`Message codephreak…  (Enter to send, Shift+Enter = newline)`} rows={1}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyDown={(e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); onSend(); } }} />
                {busy ? <button className="btn" onClick={() => stop()}>■ Stop</button>
                      : <button className="btn primary" onClick={onSend} disabled={!input.trim()}>Send →</button>}
              </div>
              {messages.length > 0 && (
                <div className="row" style={{ marginTop: 10, justifyContent: 'flex-end' }}>
                  <span className="dim small">export</span>
                  <button className="btn ghost sm" onClick={() => copy(toMarkdown(messages), 'all')}>{copied === 'all' ? '✓ copied' : '⧉ copy chat'}</button>
                  <button className="btn ghost sm" onClick={() => download('codephreak-chat.md', toMarkdown(messages))}>↓ .md</button>
                  <button className="btn ghost sm" onClick={() => download('codephreak-chat.json', JSON.stringify(messages, null, 2), 'application/json')}>↓ .json</button>
                </div>
              )}
            </div>
          </section>

          {/* ADVANCED */}
          <section className={'view' + (tab === 'advanced' ? ' active' : '')}>
            <h1>Advanced</h1>
            <p className="lead">Core sampling state, sent to the model on every turn and persisted in this browser.</p>
            <div className="card">
              <div className="presetrow">
                {Object.keys(PRESETS).map((p) => <span key={p} className="chip" onClick={() => setSettings((s) => ({ ...s, ...PRESETS[p] }))}>{p}</span>)}
                <span className="chip" onClick={() => setSettings(DEFAULTS)}>Reset</span>
              </div>
              <Slider k="temperature" label="Temperature" desc="randomness" min={0} max={2} step={0.05} s={settings} set={setSettings} />
              <Slider k="top_p" label="Top-p" desc="nucleus sampling" min={0} max={1} step={0.01} s={settings} set={setSettings} />
              <Slider k="top_k" label="Top-k" desc="candidate cutoff" min={0} max={200} step={1} s={settings} set={setSettings} />
              <Slider k="repeat_penalty" label="Repeat penalty" desc="anti-repetition" min={0.8} max={2} step={0.05} s={settings} set={setSettings} />
              <Slider k="presence_penalty" label="Presence penalty" desc="new topics" min={-2} max={2} step={0.1} s={settings} set={setSettings} />
              <Slider k="frequency_penalty" label="Frequency penalty" desc="less repetition" min={-2} max={2} step={0.1} s={settings} set={setSettings} />
            </div>
            <div className="card"><div className="grid2">
              <Num k="num_predict" label="Max output tokens (num_predict)" s={settings} set={setSettings} />
              <Num k="seed" label="Seed (blank = random)" s={settings} set={setSettings} blank />
            </div></div>
          </section>

          {/* SCIENTIFIC */}
          <section className={'view' + (tab === 'scientific' ? ' active' : '')}>
            <h1>Scientific</h1>
            <p className="lead">Ollama-native sampling internals. These are sent through the native <span className="mono">/api/chat</span> options — they genuinely take effect (unlike the OpenAI-compatible endpoint).</p>
            <div className="card">
              <div className="field"><div><label>Mirostat</label><div className="desc">0 off · 1 v1 · 2 v2</div></div>
                <select value={settings.mirostat} onChange={(e) => setSettings((s) => ({ ...s, mirostat: Number(e.target.value) }))}><option value={0}>0 — off</option><option value={1}>1</option><option value={2}>2</option></select>
                <div className="val">{settings.mirostat}</div></div>
              <Slider k="mirostat_tau" label="Mirostat τ (tau)" desc="target entropy" min={0} max={10} step={0.1} s={settings} set={setSettings} />
              <Slider k="mirostat_eta" label="Mirostat η (eta)" desc="learning rate" min={0} max={1} step={0.01} s={settings} set={setSettings} />
              <Slider k="tfs_z" label="Tail-free z" desc="tail cutoff (1 = off)" min={0.5} max={2} step={0.05} s={settings} set={setSettings} />
              <Slider k="typical_p" label="Typical-p" desc="locally typical (1 = off)" min={0} max={1} step={0.01} s={settings} set={setSettings} />
              <Slider k="min_p" label="Min-p" desc="min probability floor" min={0} max={1} step={0.01} s={settings} set={setSettings} />
              <Slider k="repeat_last_n" label="Repeat last-n" desc="lookback window" min={0} max={256} step={8} s={settings} set={setSettings} />
              <Slider k="num_ctx" label="Context window (num_ctx)" desc="tokens" min={512} max={32768} step={512} s={settings} set={setSettings} />
            </div>
            <div className="card"><h3>Stop sequences</h3><div className="hint">comma-separated; generation halts on any match</div>
              <input style={{ width: '100%' }} value={settings.stop} placeholder="e.g. \n\nUser:, </end>" onChange={(e) => setSettings((s) => ({ ...s, stop: e.target.value }))} /></div>
          </section>

          {/* PERSONA CREATOR */}
          <section className={'view' + (tab === 'persona' ? ' active' : '')}>
            <h1>.persona</h1>
            <p className="lead">The active persona <b>is</b> the system prompt sent to the model — edit it here and it takes effect on the next message. Create as many as you like; the default is the authentic Professor Codephreak prompt.</p>
            <div className="card">
              <div className="row" style={{ marginBottom: 12 }}>
                {personas.map((p) => (
                  <span key={p.id} className={'chip' + (p.id === activePersona ? '' : '')} onClick={() => setActivePersona(p.id)}
                    style={p.id === activePersona ? { borderColor: 'rgba(46,230,166,.5)', color: 'var(--accent)' } : undefined}>
                    {p.id === activePersona ? '● ' : '○ '}{p.name}
                  </span>
                ))}
                <button className="btn ghost sm" onClick={() => { const id = uid(); setPersonas((ps) => [...ps, { id, name: 'New persona', prompt: '' }]); setActivePersona(id); }}>+ New</button>
              </div>
              <div className="row" style={{ marginBottom: 10 }}>
                <input style={{ flex: 1 }} value={persona.name} onChange={(e) => setPersonas((ps) => ps.map((p) => p.id === persona.id ? { ...p, name: e.target.value } : p))} />
                {!BUILTIN_IDS.has(persona.id) && <button className="btn ghost sm" style={{ color: 'var(--bad)' }} onClick={() => { setPersonas((ps) => ps.filter((p) => p.id !== persona.id)); setActivePersona('codephreak'); }}>Delete</button>}
                <button className="btn ghost sm" onClick={() => { const base = BUILTIN_PERSONAS.find((b) => b.id === persona.id)?.prompt || CODEPHREAK_PERSONA; setPersonas((ps) => ps.map((p) => p.id === persona.id ? { ...p, prompt: base } : p)); }}>Reset</button>
                <button className="btn ghost sm" onClick={() => copy(persona.prompt, 'persona')}>{copied === 'persona' ? '✓' : '⧉ copy'}</button>
                <button className="btn ghost sm" title="pull learned directives from codephreak.py" onClick={syncLearned}>⟳ sync learned</button>
              </div>
              <textarea style={{ width: '100%', minHeight: 260, fontFamily: 'var(--mono)', fontSize: 12.5, lineHeight: 1.6 }}
                value={persona.prompt} onChange={(e) => setPersonas((ps) => ps.map((p) => p.id === persona.id ? { ...p, prompt: e.target.value } : p))} />
              <div className="small dim" style={{ marginTop: 8 }}>{persona.prompt.length} chars · this exact text is the system prompt</div>
              {learned && (
                <div className="small" style={{ marginTop: 10, color: learned.length ? 'var(--accent)' : 'var(--dim)' }}>
                  {learned.length
                    ? <>codephreak.py has learned {learned.length} directive(s) from 👍/👎 feedback and folded them into this persona.</>
                    : <>No learned directives yet — rate replies with ▲/▼ in Chat (needs <span className="mono">python3 codephreak.py</span> running).</>}
                </div>
              )}
            </div>
          </section>

          {/* HISTORY */}
          <section className={'view' + (tab === 'history' ? ' active' : '')}>
            <h1>.history</h1>
            <p className="lead">Every conversation is saved locally. Reopen to continue, export as Markdown/JSON, copy, or delete.</p>
            <div className="card">
              <div className="row" style={{ marginBottom: 12 }}>
                <button className="btn primary sm" onClick={newChat}>+ New chat</button>
                <span className="dim small">{sessions.length} saved</span>
              </div>
              {sessions.length === 0 ? <div className="dim small">No history yet — send a message.</div> : (
                <ul className="list">{sessions.map((s) => (
                  <li key={s.id}>
                    <div className="meta" onClick={() => openSession(s)}>
                      <div className="ttl">{s.title || 'Conversation'}</div>
                      <div className="sub">{new Date(s.ts).toLocaleString()} · {s.model} · {s.messages.length} msgs{activeSession.current === s.id ? ' · active' : ''}</div>
                    </div>
                    <button className="btn ghost sm" onClick={() => openSession(s)}>open</button>
                    <button className="btn ghost sm" onClick={() => copy(toMarkdown(s.messages), 'h' + s.id)}>{copied === 'h' + s.id ? '✓' : '⧉'}</button>
                    <button className="btn ghost sm" onClick={() => download(`codephreak-${s.id}.md`, toMarkdown(s.messages))}>↓</button>
                    <button className="btn ghost sm" style={{ color: 'var(--bad)' }} onClick={() => persistSessions(sessions.filter((x) => x.id !== s.id))}>✕</button>
                  </li>
                ))}</ul>
              )}
            </div>
          </section>

          {/* ABOUT */}
          <section className={'view' + (tab === 'about' ? ' active' : '')}>
            <h1>Professor Codephreak — the definitive console</h1>
            <p className="lead">A cutting-edge front end for automindX built on the Vercel AI SDK v7, streaming from Ollama via the native endpoint so every scientific parameter genuinely takes effect. <b>gpt-oss</b> is the default model.</p>
            <div className="card"><h3>Perfect symmetry</h3>
              <p className="small dim">The persona you edit in <span className="mono">.persona</span> is the exact system prompt sent to the model; the sliders in <span className="mono">Advanced</span>/<span className="mono">Scientific</span> are the exact Ollama options. Nothing is hidden between the prompt and the code — what you see is what runs.</p>
            </div>
            <div className="card"><h3>Bells & whistles</h3>
              <div className="row">{['streaming','reasoning stream','token counter','tok/s + latency','markdown + code','model picker (local + cloud)','advanced sampling','scientific / mirostat','persona creator','.history','export md/json','copy','regenerate','stop','presets'].map((b) => <span key={b} className="badge">{b}</span>)}</div>
            </div>
            <div className="card"><h3>Stack</h3>
              <div className="row">{['ai@7','@ai-sdk/react@4','createUIMessageStream','next@15','Ollama native /api/chat','react-markdown'].map((b) => <span key={b} className="badge">{b}</span>)}</div>
              <div className="small dim" style={{ marginTop: 10 }}>brain: <span className="mono">{models?.host || 'http://localhost:11434'}</span> · {nModels} models · {live ? 'live' : 'offline'}</div>
            </div>
          </section>
        </div>
      </main>
    </div>
  );
}

function Nav({ id, tab, set, ico, label }: { id: string; tab: string; set: (s: string) => void; ico: string; label: string }) {
  return <button className={tab === id ? 'active' : ''} onClick={() => set(id)}><span className="ico">{ico}</span><span className="lbl">{label}</span></button>;
}
function Slider({ k, label, desc, min, max, step, s, set }: any) {
  return (
    <div className="field">
      <div><label>{label}</label><div className="desc">{desc}</div></div>
      <input type="range" min={min} max={max} step={step} value={s[k]} onChange={(e) => set((p: any) => ({ ...p, [k]: Number(e.target.value) }))} />
      <div className="val">{s[k]}</div>
    </div>
  );
}
function Num({ k, label, s, set, blank }: any) {
  return (
    <div>
      <label className="small dim">{label}</label>
      <input type="number" style={{ width: '100%', marginTop: 4 }} value={s[k]}
        onChange={(e) => { const v = e.target.value; set((p: any) => ({ ...p, [k]: v === '' ? (blank ? '' : 0) : Number(v) })); }} />
    </div>
  );
}
