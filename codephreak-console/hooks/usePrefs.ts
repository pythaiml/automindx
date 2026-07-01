'use client';
import { useEffect, useState } from 'react';
import { LS, PREFS_DEFAULT, type Prefs } from '@/lib/store';

// User preferences: identity, avatars, accent colours, chat font.
// Applied live as a preview; persisted only on save().
export function usePrefs() {
  const [prefs, setPrefs] = useState<Prefs>(PREFS_DEFAULT);
  const [saved, setSaved] = useState<Prefs>(PREFS_DEFAULT);

  useEffect(() => {
    try {
      const pr = localStorage.getItem(LS.prefs);
      if (pr) { const v = { ...PREFS_DEFAULT, ...JSON.parse(pr) }; setPrefs(v); setSaved(v); }
    } catch {}
  }, []);

  useEffect(() => {
    const root = document.documentElement.style;
    root.setProperty('--accent', prefs.accent);
    root.setProperty('--accent-2', prefs.accent2);
    root.setProperty('--chat-font', prefs.chatFont + 'px');
  }, [prefs]);

  const dirty = JSON.stringify(prefs) !== JSON.stringify(saved);
  function save() { localStorage.setItem(LS.prefs, JSON.stringify(prefs)); setSaved(prefs); }
  function revert() { setPrefs(saved); }
  function reset() { setPrefs(PREFS_DEFAULT); }

  // Avatar upload → 128px center-cropped data URL (keeps localStorage small).
  function uploadAvatar(file: File | undefined, key: 'avatar' | 'botAvatar') {
    if (!file || !file.type.startsWith('image/')) return;
    const reader = new FileReader();
    reader.onload = () => {
      const img = new Image();
      img.onload = () => {
        const size = 128;
        const c = document.createElement('canvas');
        c.width = c.height = size;
        const ctx = c.getContext('2d');
        if (!ctx) { setPrefs((p) => ({ ...p, [key]: String(reader.result) })); return; }
        const s = Math.min(img.width, img.height);
        ctx.drawImage(img, (img.width - s) / 2, (img.height - s) / 2, s, s, 0, 0, size, size);
        setPrefs((p) => ({ ...p, [key]: c.toDataURL('image/png') }));
      };
      img.src = String(reader.result);
    };
    reader.readAsDataURL(file);
  }

  return { prefs, setPrefs, dirty, save, revert, reset, uploadAvatar };
}
