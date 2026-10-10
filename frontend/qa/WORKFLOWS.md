# Workflow frontend verification — 2026-10-10

Production build, 83 selected frontend checks and the full owned browser/native
matrix pass. The workflow path uses actual tiny-GGUF native generation `a` → `bc`,
an exact output check and a saved Agent that reads actual native runtime status
through an authored HTTP tool-call fixture. Separate native CLI exports and an
independent native generation response verify exact references, input links,
observations, outputs and token counts. Endpoint text/usage are fixture values,
not evidence of real remote-model intelligence or performance.

Browser checks include native keyboard typing/caret retention, dependency-safe
remove/reorder rejection, profile selection/pinned inspection, independent native
stale-head/CAS conflict recovery, inert labels/output, unsaved-run protection,
save/reload and Stop during an actual pending Agent HTTP request. Stop retains
the completed native generation/check output and a stopped nested Agent trace
with unknown usage. Existing authoring, Context, dataset, training, evaluation,
switcher, comparison, inspection/tokenizer and Agent checks still pass.

The passing run used installed Chrome 154.0.8037.98, an owned private profile,
loopback servers, temporary database/model copies and a synthetic key. The first
attempt could not start the former Edge path, which no longer existed; it is
excluded. A subsequent integration attempt exposed the catalog's absent metadata;
the classifier now reads bounded exact workflow revisions instead of inferring
record kind from titles. Tests include lookup/page/cursor bounds.

Visual review confirms semantic clay surfaces, clear step/input controls, escaped
observations, history and responsive stacking with no horizontal overflow at
390px. No user database/browser profile/process or backend source is changed.
Broad reader/keyboard/zoom/high-contrast, actual Tauri/installer, real provider,
large workflow/catalog scale and native job/permission/node audits remain open.

| Appearance | Definition | Saved result/history |
| --- | --- | --- |
| Light, 1280px | [Definition](themes/workflows-definition.png) | [Result](themes/workflows-result.png) |
| Dark, 1280px | [Definition](themes/workflows-dark-definition.png) | [Result](themes/workflows-dark-result.png) |
| Light, 390px | [Definition](themes/workflows-mobile-definition.png) | [Result](themes/workflows-mobile-result.png) |

Only these new checkpoint images are copied after the full matrix passes; other
captures remain private temporary artifacts. See [formats, operational limits
and native handoff](../docs/WORKFLOWS.md). The full product goal remains active.
