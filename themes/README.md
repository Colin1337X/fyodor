# Shared semantic themes

`palettes.json` is the authoritative role mapping for the desktop and native
terminal. Eighteen roles cover surfaces, text, accents, status, focus, selection
and message backgrounds. The ten built-in presets are Fyodor (the light default),
Fyodor Dark, Cherenkov, Fyodor Mocha, Soda, Strawberry, and Catppuccin Latte,
Frappé, Macchiato and Mocha. Fyodor Mocha is an independent brown/cream palette.

Run `python scripts/generate-themes.py` after editing the source; `--check`
verifies committed adapters. It emits C tables, CSS semantic variables and a
blocking browser preset manifest. Python and network access are not runtime or
normal build dependencies. CSS aliases keep existing components on semantic
roles; generation prevents palette values drifting between clients.

Desktop Settings exposes the required names. Preferences survive reload;
`system` follows the OS while an explicit selection does not. Old blue-black,
blue-white and orange-white preferences migrate to Fyodor Dark, Fyodor and
Fyodor Mocha. Graphite, Forest and Violet remain desktop-only legacy choices.
The default for a new installation is Fyodor.

### Desktop customization

Settings → Appearance → Customize appearance exposes accent/code/syntax color
overrides, 0–24 px corners, 12–20 px base type, system/sans/serif font stacks,
three monospace stacks, 160–320 px sidebar width, 12–36 px workspace spacing,
centered/wide chat, panel borders, shadows, translucency and motion controls.
Compact density reduces the selected spacing. Customizations apply immediately
and persist separately from connection credentials. Reset preserves sidebar
collapse and does not reset pending connection form edits. Color controls show
the effective theme color when no override is active; Use theme removes an
override. Invalid values from storage or setters are ignored.

Accent overrides choose black/white text for contrast. Other custom colors are
user-selected and are not contrast constrained. Panel borders can be hidden
without removing control boundaries or keyboard focus outlines. Backdrop
translucency is enabled only with browser CSS support; this does not enable
native transparent windows. Reduced motion suppresses animations/transitions;
OS reduced-motion preferences are respected even in Standard mode. Font stacks
fall back to locally available fonts, without downloading assets.

Code fences for C/C++, JavaScript/TypeScript, Python and JSON receive basic
lexical keyword/string/comment/number highlighting with independently editable
colors. It is not a full language parser: advanced literals/preprocessor syntax
may remain unclassified. Unsupported languages and blocks over 64 KiB stay
plain escaped text. Highlighting adds only fixed span classes, preserves
`textContent` for copying, and never interprets model HTML as executable markup.

The TUI accepts `--theme ID`, `--themes`, and `--color auto|truecolor|256|16|none`.
`FYODOR_THEME` chooses a startup theme; `t` cycles presets for the current session.
Automatic color selection honors `NO_COLOR`, then `COLORTERM=truecolor/24bit`,
then `TERM` containing `256color`, otherwise 16 colors. Explicit `--color`
overrides automatic detection. Plain mode emits no styling sequences. The
256-color mapper uses the stable cube/grayscale entries; 16-color appearance
depends on terminal configuration. Quantized identical foreground/background
indices receive contrasting foregrounds. Monochrome retains selection via
reverse-video plus the existing selection marker. No theme selection modifies
the workspace database. Persistent interactive theme preference is still open.

## Catppuccin attribution

Palette data version 1.8.0 was retrieved on 2026-09-30 from the official
[palette repository](https://github.com/catppuccin/palette), using its
[JSON palette](https://raw.githubusercontent.com/catppuccin/palette/main/palette.json)
and [MIT license](https://raw.githubusercontent.com/catppuccin/palette/main/LICENSE).
The original license is retained in `vendor/CATPPUCCIN-LICENSE`, bundled in the
frontend at `/theme-licenses.txt`, linked from Settings, and compiled into
`fyodor-tui --licenses`. No Catppuccin runtime library is added. Latte uses
surface0 for selection and mauve for accent/hover to keep the tested text
pairings at least 4.5:1; other flavors use blue as the accent.

SHA-256 of downloaded files:

- `catppuccin.json`: `4bc114bb6b3c9a9c9e156564aa84625aef32c5da514d9dd431cf1fcad433a05f`
- `CATPPUCCIN-LICENSE`: `814096d2c34cc216c624738a49356f32b7237733b4f7edb0685f4e50ef5074ba`

## Verified scope

- All 16 frontend unit tests pass, including persistence, migration, invalid
  storage, required role coverage and primary text/background plus accent/hover
  contrast. These do not constitute an accessibility audit of every UI state.
- Browser QA renders all ten presets, checks body computed colors and page
  overflow, verifies reload persistence and changes theme through Settings.
  Screenshots/results are in `frontend/qa/themes`. Fyodor, Cherenkov and the
  Strawberry settings dialog and custom large serif/square-corner settings were visually inspected. The engine was offline
  for this isolated frontend check; no inference claim follows from it.
- Four native TUI/theme tests pass under C17, C23 and ASan/UBSan. The hidden
  Windows console exercises all four color modes, theme cycling, command
  dispatch, quit and Ctrl+C restoration. POSIX is not yet tested.
- Frontend production build passes. Browser QA additionally operates the
  customization controls, checks effective font/corner/shadow/sidebar/accent
  properties, reload persistence, reset and safe syntax spans/text preservation.
  Native Tauri packaging, platform-specific font/rendering behavior and a full
  responsive/accessibility audit remain unverified.

For browser QA, run the frontend on port 5178, then
`node frontend/tests/theme-browser.mjs` from the repository root. This optional
Windows test uses a separate temporary headless Edge profile and never connects
to an existing user browser. `FYODOR_EDGE`/`FYODOR_PREVIEW` may override paths.
