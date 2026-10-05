# Fyodor

A desktop AI workspace with an independent native model backend.

The backend is written in C, with CPU execution and an optional Vulkan provider. Both compile without a CUDA directory or toolkit. Tauri can launch the backend while the frontend communicates directly over authenticated local HTTP.

See [backend/README.md](backend/README.md) for the architecture, supported models, CPU/Vulkan builds, API, configuration, ownership rules, tests, and current limitations.

The native `fyodor` resource client now supports local import/export, listing,
and revision-checked updates through shared C application services. See
[resource commands](backend/RESOURCE_CLIENT.md) for usage and current limits;
the pure-C [terminal client](backend/TUI.md) now provides resource browsing and
shared command entry. Broader CLI/TUI workspaces remain in development.

The desktop [Resources library](backend/RESOURCE_HTTP.md) uses the same native
store for documents, notes and resource packages through authenticated local HTTP.

See [frontend/README.md](frontend/README.md) for the pure vanilla HTML/CSS/JS desktop interface, C sidecar integration, and cross-platform installers.

Quick CPU build:

```sh
cmake -S . -B build-cpu -DCMAKE_BUILD_TYPE=Release
cmake --build build-cpu --config Release --parallel
ctest --test-dir build-cpu -C Release --output-on-failure
```
