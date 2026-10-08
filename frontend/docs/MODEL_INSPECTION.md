# Loaded model inspection and tokenization

Updated 2026-10-08. Overview and Models expose Inspect beside every loaded model.
This frontend uses existing native registry, runtime, shape and tokenizer APIs;
it introduces no backend implementation or persistent model identity schema.

## Workflow

Inspect opens a native modal without changing the global model selection,
execution provider, loaded weights or conversation settings. The loaded filename
heads the dialog; Source path retains the full reported path. The registry section
shows format/version, architecture, exact safe file-size/count values, tensor and
metadata counts, data offset, producer, reported capabilities and active compute.
Missing or unsafe values say Not reported rather than becoming zero or being
derived from a filename.

For a registry entry with a selectable native execution context, the inspector
requests shape information: context tokens, embedding size, layers, attention/KV
heads, vocabulary size, experts, sliding window and MTP assistant flag. Zero is a
reported zero, not an absent value. Shape reads can fail independently of the
registry display; errors stay in the dialog and the loaded-file observations stay
available. Inspection-only formats still expose reported registry metadata without
pretending that native text execution exists.
Registry error notes are labelled by tensor execution versus text generation;
an unavailable tensor adapter does not imply that native text generation failed.

Compiled providers lists CPU/CUDA/Vulkan/ROCm/MLX inclusion reported for the
running backend. Inclusion is not proof that a GPU driver works or that every
model architecture is supported on that provider. The existing Execution backend
controls and native initialization remain authoritative. The inspector itself
does not initialize or switch providers.

Close or Escape dismisses the modal, clears its submitted tokenizer text/result
DOM and returns focus to the connected Inspect button. If the originating button
was replaced, focus returns to the header quick switcher. Native modal containment
owns Tab focus; another modal cannot be deliberately opened over this one. The
dialog title and all field/error content are escaped. The model list and dialog
wrap long paths on narrow screens and use existing semantic clay tokens.
The close control remains in a sticky header while reviewing lower fields.

## Raw tokenizer tool

Enter text and click Tokenize text. This explicit action posts the exact textarea
value to the inspected model's native tokenizer. It does not trim whitespace,
append generation instructions, assemble Context, apply a chat template, generate
text, or save a resource. Model-specific special tokens follow native tokenizer
behavior; the UI does not supply an add-BOS switch or guess their policy.

The tool is enabled for selectable native text/draft models. Backend errors remain
authoritative if the execution context disappeared or tokenization fails. The
editor and repeat request are disabled during a request; Close remains available.
Native HTTP requests are not cancelled on dismissal. A late completion is ignored.

Input is bounded to 64 KiB of UTF-8 and rejects NUL before requesting native C
string tokenization. The character cap is also 65,536, but multibyte text reaches
the byte bound earlier. Empty text is permitted as a request; its count follows
the model tokenizer. Tokenization is independent of generation context capacity:
a long text can tokenize successfully even when it would exceed a later generation
request's context. This tool does not approve the text as a valid training sequence.

A result requires version 1, success, the exact requested runtime model ID, a
nonnegative safe integer count equal to the token-array length, and unsigned
32-bit integer token IDs. The frontend accepts at most 131,072 IDs; native request/
response limits can reject smaller requests/results. Invalid responses fail
without displaying partial successful counts. The screen displays at most the
first 256 IDs and explicitly labels that prefix. Editing text clears the result
and disables Download; old IDs cannot silently refer to the edited input.

Download token IDs is explicit and local. It creates `fyodor-tokenization.json`:

```json
{
  "schema": 1,
  "kind": "model-tokenization",
  "model": {"id": 1, "path": "reported local path", "file_size": 123, "format": "gguf"},
  "text": "submitted raw text",
  "count": 2,
  "tokens": [12, 34]
}
```

The example numbers are schema illustrations, not fixture measurements. The
download includes the complete validated token list and submitted text, even if
only a prefix is shown. No text/result is written to app localStorage, a native
resource or a model annotation. The object URL is revoked after download dispatch.
There is no import/replay mechanism or tokenizer decoder in this feature.

## Contracts and maintenance

| Source | Current adapter | Use |
| --- | --- | --- |
| `GET /api/v1/models` | Existing `api.models()` / shell state | Loaded-file registry observation. |
| `GET /api/v1/runtime` | Existing `api.runtime()` / shell state | Active compute, selectable contexts, compiled providers. |
| `POST /api/v1/model/info` with `{model_id}` | `api.modelInfo()` | Current native execution shape. |
| `POST /api/v1/tokenize` with `{model_id,text}` | `api.tokenize()` | Native raw token count/IDs. |

Local requests use the launcher's existing in-memory bearer connection. Opening
uses current shell registry/runtime observations; it does not silently refetch
every catalog. Close, use the existing Refresh action, then reopen to refresh those
observations. Shape/tokenization are live requests and can differ from an earlier
registry observation if another client changes the runtime concurrently. There
is no combined atomic snapshot of all four APIs.

`src/model-inspection-data.js` owns validated observations, shape/token response
boundaries and escaped field markup. It discards unknown fields and unsafe numeric
observations instead of manufacturing precision. `src/model-inspector.js` owns
modal/request/input/download lifetime. `src/main.js` supplies the API and current
state, while `src/workspace.js` supplies Inspect buttons. Styles live in shared
`clay.css`; no framework or editor dependency is added.

The modal is installed once per window. Each open receives a request ticket;
dismissal advances it and clears result state. Close events queued before an
immediate reopen do not clear the new open. Native shape/token completions verify
ticket/open state before painting, so an old model's response cannot populate a
new inspection. Results update only their own fields; tokenization does not rebuild
the textarea. Input handlers clear stale results before another request.

Runtime IDs must be positive safe integers. They identify a loaded slot in the
current backend process, not an immutable file/model lineage. Path/size/format
are observations, not a cryptographic identity or proof of training ancestry.
Stable IDs, quantization/tensor-type summaries, tokenizer metadata beyond current
shape/IDs, aliases/tags, last-used state, lineage and canonical evaluation links
remain backend contract work. Conversion, merging, quantization and surgery have
no claimed action here.

## Verification and remaining audits

`node --test frontend/tests/model-inspection.test.mjs` covers reported zero versus
missing counters, unsafe identities, version/model binding, malformed shape
values, raw Unicode/whitespace and byte/NUL bounds, token count/ID integrity and
escaped metadata/errors. The selected full suite now has 67 checks; see
`ARCHITECTURE.md` for its command and `npm.cmd run build --prefix frontend`.

`node frontend/tests/resource-browser.mjs` uses the owned native backend and two
tiny model copies. The inspector helper compares displayed context/vocabulary/
compute with independent native responses, checks real token IDs, reads both a
short and over-256-ID complete download, invalidates edited results, rejects NUL,
verifies Escape/focus/cleared DOM and preserves global model selection. It delays
one actual native HTTP shape response, dismisses the first inspection, opens the
other model and releases the old response to verify replacement lifetime. Response
contents are not fabricated. Earlier resource/authoring/dataset/evaluation/training
and independent native CLI verification still run in the same harness.

Light/dark/390px screenshots are in `../qa/themes/model-inspector*.png` and
`../qa/themes/model-tokenizer*.png`; overflow
is checked for the model view/modal. Full platform Tauri/window packaging,
screen-reader, zoom, forced-colors, large real model families and optional GPU
runtime audits remain open. This feature improves actual metadata inspection; it
does not establish completion of the full registry/Model Lab brief.
