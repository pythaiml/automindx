// Shared client-side storage keys + preference shape for the console hooks.
export const LS = {
  personas: 'cpk.personas', active: 'cpk.activePersona', settings: 'cpk.settings',
  history: 'cpk.history', think: 'cpk.think', prefs: 'cpk.prefs',
  autonomous: 'cpk.autonomous', sagi: 'cpk.sagi',
};

export const PREFS_DEFAULT = {
  name: 'You',
  avatar: '',      // data URL — your avatar
  botAvatar: '',   // data URL — codephreak's avatar
  accent: '#2ee6a6',
  accent2: '#37b6ff',
  chatFont: 14,    // px
};
export type Prefs = typeof PREFS_DEFAULT;
