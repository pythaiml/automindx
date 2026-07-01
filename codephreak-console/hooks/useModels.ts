'use client';
import { useEffect, useState } from 'react';

export type Model = { name: string; param_size?: string; size_gb?: number | null; free?: boolean; pulled?: boolean };
export type Models = { ok: boolean; local: Model[]; cloud: Model[]; host?: string; error?: string };
type Verdict = { accessible: boolean; reason: string; detail?: string };

// Everything about model discovery, account-aware cloud access, and one-click
// provisioning (pull / install Ollama).
export function useModels(defaultModel: string) {
  const [models, setModels] = useState<Models | null>(null);
  const [access, setAccess] = useState<Record<string, Verdict>>({});
  const [model, setModel] = useState(defaultModel);
  const [pulling, setPulling] = useState<Record<string, string>>({});
  const [installMsg, setInstallMsg] = useState('');

  const isCloud = (n: string) => !!models?.cloud.some((m) => m.name === n);
  const isPulled = (name: string) => {
    const m = [...(models?.local || []), ...(models?.cloud || [])].find((x) => x.name === name);
    return m ? m.pulled !== false : false;
  };

  async function probeAccess(names: string[]) {
    const list = names.filter(Boolean);
    if (!list.length) return;
    try {
      const j = await (await fetch('/api/access', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ models: list }) })).json();
      if (j.ok) setAccess((a) => ({ ...a, ...j.results }));
    } catch {}
  }

  async function loadModels() {
    try {
      const j: Models = await (await fetch('/api/models')).json();
      setModels(j);
      if (j.ok && ![...j.local, ...j.cloud].some((m) => m.name === model) && [...j.local, ...j.cloud].some((m) => m.name === defaultModel)) setModel(defaultModel);
      if (j.ok && j.cloud.length) probeAccess(j.cloud.filter((m) => m.pulled !== false).map((m) => m.name));
    } catch { setModels({ ok: false, local: [], cloud: [], error: 'offline' }); }
  }

  async function pull(name: string): Promise<boolean> {
    setPulling((p) => ({ ...p, [name]: 'starting…' }));
    try {
      const r = await fetch('/api/pull', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ model: name }) });
      if (!r.ok || !r.body) { const j = await r.json().catch(() => ({})); setPulling((p) => ({ ...p, [name]: '✗ ' + (j.error || 'failed') })); return false; }
      const reader = r.body.getReader(); const dec = new TextDecoder(); let buf = '';
      for (;;) {
        const { done, value } = await reader.read(); if (done) break;
        buf += dec.decode(value, { stream: true }); const lines = buf.split('\n'); buf = lines.pop() || '';
        for (const line of lines) { if (!line.trim()) continue; try { const o = JSON.parse(line); setPulling((p) => ({ ...p, [name]: o.status || '…' })); } catch {} }
      }
      setPulling((p) => ({ ...p, [name]: '✓ pulled' }));
      await loadModels();
      setTimeout(() => setPulling((p) => { const n = { ...p }; delete n[name]; return n; }), 2500);
      return true;
    } catch (e: any) { setPulling((p) => ({ ...p, [name]: '✗ ' + String(e?.message || e) })); return false; }
  }

  async function installOllama() {
    if (!window.confirm('Run the official Ollama installer on this machine?\n\ncurl -fsSL https://ollama.com/install.sh | sh')) return;
    setInstallMsg('installing… (this can take a minute)');
    try {
      const j = await (await fetch('/api/install-ollama', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ confirm: 'install-ollama' }) })).json();
      setInstallMsg(j.note || (j.ok ? '✓ installed' : '✗ failed'));
      if (j.ok) await loadModels();
    } catch (e: any) { setInstallMsg('✗ ' + String(e?.message || e)); }
  }

  useEffect(() => { loadModels(); const iv = setInterval(loadModels, 8000); return () => clearInterval(iv); /* eslint-disable-next-line */ }, []);
  useEffect(() => { if (isCloud(model) && !access[model]) probeAccess([model]); /* eslint-disable-next-line */ }, [model, models]);

  const live = !!models?.ok;
  const nModels = (models?.local.length || 0) + (models?.cloud.length || 0);
  const cloudSignedIn = Object.values(access).some((v) => v.accessible);

  return { models, access, model, setModel, pulling, installMsg, live, nModels, cloudSignedIn,
    loadModels, probeAccess, pull, installOllama, isPulled, isCloud };
}
