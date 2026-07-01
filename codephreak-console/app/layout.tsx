import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'Professor Codephreak · automindX',
  description: 'Cutting-edge AI SDK console for automindX — Ollama, gpt-oss by default',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
