import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'automindX — Intelligent Autonomous ML',
  description: 'automindX console — Ollama, gpt-oss by default, Professor Codephreak persona',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
