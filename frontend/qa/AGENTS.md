# Agents frontend verification — 2026-10-08

Production Vite build and 75 selected frontend checks pass. The owned Edge/native
browser matrix verifies saved endpoint profiles, independent native edits,
stale-start rejection, CAS conflict retention, explicit reload, default-empty
tool choices, actual runtime data in the tool loop, inert endpoint output,
unsaved-result protection, run save/reload and Stop during a pending HTTP request.
Independent pure-C CLI exports verify exact definition revision, native child
identity, forwarded runtime observation, reported counters and stopped status.
Existing theme, authoring, tokenizer, Context, dataset, training and evaluation
checks also pass. No backend implementation changed.

The endpoint is an authored protocol fixture hosted on an owned loopback server;
it validates requests and actual native runtime output. Its response text/usage
are fixture values, not real model inference or reasoning/performance evidence.
Its only API key is a synthetic fixture value in a private browser session.
No user database, profile or credentials are used.

Visual review confirms clay surfaces, readable escaped output, visible controls,
responsive stacking and no horizontal overflow at 390px. Captures show an
explicitly saved run and its history. Broad reader/keyboard/zoom/high-contrast,
real endpoint/provider, Tauri/installer and realistic scale audits remain open.

| Appearance | Profile | Saved result/history |
| --- | --- | --- |
| Light, 1280px | [Profile](themes/agents-profile.png) | [Result](themes/agents-result.png) |
| Dark, 1280px | [Profile](themes/agents-dark-profile.png) | [Result](themes/agents-dark-result.png) |
| Light, 390px | [Profile](themes/agents-mobile-profile.png) | [Result](themes/agents-mobile-result.png) |

The full matrix captures privately in an owned temporary folder, then copies only
these new images after all browser and CLI checks succeed. An earlier attempt
failed while overwriting an unrelated existing screenshot (Windows file open
error); it is excluded. The isolated capture run passed. See
[formats, bounds and remaining backend work](../docs/AGENTS.md).
