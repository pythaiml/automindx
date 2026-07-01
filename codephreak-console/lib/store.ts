// Shared client-side storage keys + preference shape for the console hooks.
export const LS = {
  personas: 'cpk.personas', active: 'cpk.activePersona', settings: 'cpk.settings',
  history: 'cpk.history', think: 'cpk.think', prefs: 'cpk.prefs',
  autonomous: 'cpk.autonomous', sagi: 'cpk.sagi',
};

// Professor Codephreak is an NFT (github.com/Professor-Codephreak → OpenSea).
export const CODEPHREAK_AVATAR = 'https://avatars.githubusercontent.com/u/140855987?v=4';
export const CODEPHREAK_NFT = 'https://opensea.io/item/ethereum/0x52525cf31cc267d9635c38ec9ec99596f4664dc8/1';

export const PREFS_DEFAULT = {
  name: 'You',
  avatar: '',                     // data URL — your avatar
  botAvatar: CODEPHREAK_AVATAR,   // codephreak's NFT identity (overridable)
  accent: '#2ee6a6',
  accent2: '#37b6ff',
  chatFont: 14,                   // px
};
export type Prefs = typeof PREFS_DEFAULT;
