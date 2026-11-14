The existing resource workspace remains the default; use `--monitor` as the first argument for model/training telemetry. Both views use the same native C theme palettes; shared navigation remains pending.

# Native terminal workbench

Build `fyodor-tui` with the ordinary backend CMake build. It uses the C engine,
Windows Console or POSIX termios, and ANSI output. No Go, Rust, C++, curses,
Bubble Tea, or other terminal framework is required.

```
fyodor-tui --monitor --model E:\models\model.gguf
fyodor-tui --monitor --metrics training.csv --theme catppuccin-mocha
fyodor-tui --monitor --model model.gguf --snapshot
```

The model view performs structural container inspection, not model loading or
inference. Inspection errors distinguish malformed/truncated containers,
unsupported formats or versions, resource limits, and read failures; failed
inspection clears the previous model details. The training view follows the CSV produced by `fyodor-train --metrics`:
step, loss, tokens/s, compute provider, optional validation loss, gradient norm,
and device memory. Press `g` to cycle loss, tokens/s, gradient norm, and device memory graphs, or
use `:chart loss|rate|gradient|memory`. Each graph uses a labeled dynamic
vertical scale; missing optional values remain gaps rather than zeroes. Only complete, finite rows are accepted. Refresh reads at most
the last 64 KiB and retains at most 192 samples; malformed rows are counted.
A missing file clears old samples. File replacement/restart is supported.
Periodic polling uses a one-second monotonic interval, independent of typing
speed. Explicit refresh and metrics-path commands still refresh immediately;
pause suppresses reads without accumulating a backlog of refreshes.

Use 1/2/3, Tab, arrows, or hjkl to switch views; `?` opens help, Escape closes it,
Space freezes metrics, `r` refreshes, and `q` or Ctrl-C exits. The `:` command bar
accepts `open PATH`, `metrics PATH`, `overview`, `model`, `training`, `pause`,
`refresh`, `theme ID`, and `quit`. Paths include spaces without extra quoting inside the
command bar. Tab completes built-in command names and chart choices, extending a unique or shared prefix. Ambiguous matches appear in the status bar; completion never executes a command. Paths and theme IDs remain literal input. Up/Down while editing recalls the last 16 submitted commands, retaining the current draft when you return past the newest entry. Consecutive duplicates are collapsed. Recall does not execute a command; Enter runs it and Escape cancels. History stays in memory for this session only. Commands never invoke a shell. Press `t` to cycle the shared themes, or use
`--theme ID` / `:theme ID`. `FYODOR_THEME` sets the startup default; an explicit
`--theme` overrides it. `--themes` lists IDs and `--licenses` shows palette
attribution. `--color auto|truecolor|256|16|none` selects terminal color support.
Auto detects `COLORTERM`/`TERM`, falls back to 16 colors, and respects `NO_COLOR`;
an explicit color mode overrides detection. Theme changes last for this session.
Snapshot mode emits plain UTF-8 without terminal escape sequences. Interactive mode
requires a terminal and restores its modes and alternate screen on normal exit
and handled interruption. Long command paths scroll to keep the editing end
visible, with a `<` marker for clipped leading text. Scrolling preserves UTF-8
characters and accounts for wide Korean/CJK characters. While editing, Ctrl-W
deletes trailing ASCII spaces and the preceding space-delimited word without
splitting UTF-8 characters; Ctrl-U clears the active command. Both shortcuts
recover from command overflow and preserve submitted history. Neither executes
the command.

Windows UTF-16 input discards incomplete surrogate pairs when a command is
cancelled or a navigation/editing key intervenes. A later orphaned low surrogate
cannot insert an unintended character into a new command; valid emoji pairs
still decode as one Unicode character.

This is an initial inspection/monitoring workbench. It does not yet launch,
cancel, or resume training, or run interactive generation. Those controls require
engine lifecycle integration. Model inspection is synchronous. Unicode input is
supported; display width is a conservative CJK/emoji approximation, not full
Unicode grapheme segmentation. Combining marks are shown as placeholders.
Windows interaction has been exercised through console tests and Linux through
private PTY tests; macOS terminal validation remains pending. Terminals smaller than 48 x 16 show a
compact resize message. Rendering is capped at 240 x 80 cells.

## Design provenance

Original Fyodor C implementation, without copied project source. Interaction
inspiration: [lazygit](https://github.com/jesseduffield/lazygit) keyboard panels;
[btop](https://github.com/aristocratos/btop) compact telemetry and graphs;
[Bubble Tea](https://github.com/charmbracelet/bubbletea) state/event/view separation;
[Lip Gloss](https://github.com/charmbracelet/lipgloss) styled layout; and
[k9s](https://github.com/derailed/k9s) command and help navigation.

## Validation

`tui-state-render` checks navigation, Unicode command editing, overflow handling,
CSV partial writes, malformed/nonfinite rows, pause/restart, bounded large-file
tailing, terminal control sanitization, and thousands of dimension/view
combinations including extreme finite graph values. Theme tests cover palette switching, invalid
IDs, all shared palettes at 16/256/true color, and plain-output sanitization.
`tui-monitor-theme-cli` checks shared palette/license listings, snapshots, option
validation, and environment overrides through the executable. `tui-snapshot` checks the
plain executable path. The executable/theme tests pass in GCC C17 CPU and GCC C23
CUDA/Vulkan-enabled builds. The state/render test additionally passes under
Clang ASan/UBSan on Windows. ConPTY checks cover initial
render, help, idle rendering, q, and Ctrl-C cleanup. A real Gemma4 12B GGUF
inspection reports 667 tensors and 43 metadata entries.

The physical-removal build also passes all 79 registered tests (four expected
ROCm/MLX hardware skips) with optcpp and CUDA template directories absent,
including the monitor theme CLI test and training/export/inference workflows.

POSIX input keeps one byte of lookahead: a character immediately after Escape,
or a new key after malformed UTF-8, is preserved for the next event. Common
CSI/SS3 key sequences are consumed through their final byte (bounded to 64 bytes);
unsupported Delete/Home/function-key suffixes are not inserted as command text.
The plain C decoder has byte-stream regression coverage under C17 and C23,
including Unicode bounds, malformed input recovery and modified arrow keys.
The byte-stream checks run on Windows too. On Unix, CTest also registers
`tui-monitor-posix-pty`: it launches the actual monitor in a private PTY, sends
rapid Escape/command input, malformed UTF-8 before Enter, an unsupported Delete
sequence and a Unicode path, then verifies successful exit and exact terminal
mode restoration. Both C17 and C23 builds passed this PTY test and the state,
snapshot and CLI checks on Ubuntu 26.04 WSL with GCC 15.2.0, optional C++ and all
GPU backends disabled. The vendored SQLite build emitted a GCC string-overread
warning; the builds completed and Fyodor's strict-warning targets passed.

The resource browser (`fyodor-tui --store PATH --namespace NAME`) also decodes
POSIX CSI/SS3 arrow sequences, including modified arrows. Unsupported complete
sequences such as Delete are consumed without inserting their suffix into a
command. A normal key immediately after Escape is retained for the next event.
Decoding is bounded to 64 bytes and pending input resets on terminal entry.
`application-tui-posix-pty` exercises actual resource selection with Down/Up,
command editing, rapid Escape recovery, and terminal restoration. It reproduced
wrong-resource selection in the original executable and passes with the fix
under Linux C17/C23. The parser, workflow and Windows ConPTY checks also pass.
This does not add training start/cancel/resume or asynchronous generation.

The resource browser's Windows console adapter combines valid UTF-16 surrogate
pairs before UTF-8 encoding, so supplementary characters can be entered in
commands. Cancellation, navigation, and ordinary intervening characters clear
incomplete pairs; orphaned low surrogates are ignored. The private Windows
console regression covers emoji entry, cancellation and arrow interruption.
Display remains ASCII-escaped as documented for the resource browser.

### Resource browser page navigation

Page Up and Page Down move by the visible content height within the current
resource page or detail/command output. Home and End jump to its beginning or
end. `n` still fetches the next resource page; these keys do not issue additional
store requests. Terminal command entry ignores navigation key codes rather than
inserting escape-sequence suffixes. POSIX CSI/SS3 Home/End and numeric tilde
variants are supported, alongside Windows console virtual keys.

Linux C17/C23 parser, workflow and real-PTY checks pass, including a long resource
and list boundary movement. Windows parser, workflow and private-console checks
also pass, including Home/End and Page Up/Down in detail output. No Go or Rust
runtime is introduced.
