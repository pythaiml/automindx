#!/usr/bin/env bash
# Launch the Professor Codephreak AI SDK console for automindX.
#   - codephreak.py    self-improving persona engine (realtime feedback)  :5001
#   - codephreak-console  Next.js + Vercel AI SDK v7 UI (Ollama)           :3100
#
# Ollama must be running (`ollama serve`); gpt-oss:120b-cloud is the default model.
set -e
cd "$(dirname "$0")/.."   # repo root (this script lives in scripts/)

echo "▸ Starting self-improving engine (codephreak.py) on :5001 …"
python3 codephreak.py >/tmp/codephreak_engine.log 2>&1 &
ENGINE_PID=$!
trap "kill $ENGINE_PID 2>/dev/null" EXIT

cd codephreak-console
if [ ! -d node_modules ]; then
  echo "▸ Installing console dependencies (first run) …"
  npm install --no-audit --no-fund
fi
echo "▸ Starting Codephreak console on http://localhost:3100 …"
echo "  default model: gpt-oss:120b-cloud  ·  reasoning + scientific settings on"
npm run dev
