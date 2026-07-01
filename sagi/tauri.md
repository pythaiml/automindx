# Shipping sagi as a Tauri app

sagi is UI- and runtime-agnostic, so it wraps cleanly in [Tauri](https://tauri.app):

1. `npm create tauri-app` (or add Tauri to the existing console).
2. Expose the agnostic host over Tauri commands: `callModel`, `store`, `emit`.
   The web UI calls them via `@tauri-apps/api`.
3. Bundle `sagi/` (manifest + modules) as an app resource; the self-building loop
   reads/writes the manifest through the Tauri fs/command bridge.
4. Point the model host at a local Ollama (`http://localhost:11434`) or a remote
   endpoint — sagi doesn't care which.

Result: a self-building sAGI that runs as a native desktop app while remaining a
drop-in module for any other project.
