# Fyodor desktop

The frontend is pure vanilla HTML, CSS, and JavaScript. Tauri 2 only supplies the cross-platform window, native file picker, installers, and C sidecar supervision. Inference requests travel directly from the webview to the authenticated loopback HTTP API.

## Development

```sh
cmake -S .. -B ../build-cpu -DCMAKE_BUILD_TYPE=Release
cmake --build ../build-cpu --config Release --parallel
npm install
npm run desktop:dev
```

`prepare:sidecar` copies the native backend into Tauri resources. Set `FYODOR_BACKEND_BIN` for a different native build.

Use `npm run dev` for browser-only UI work. Set `VITE_FYODOR_URL` and `VITE_FYODOR_TOKEN` to connect it to an already running backend.

## Installers

`npm run desktop:build` produces formats supported by the current host: NSIS/MSI on Windows, DMG/app on macOS, and AppImage/deb/rpm on Linux. Build on each target platform with its matching C executable.
