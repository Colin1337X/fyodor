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
inference. The training view follows the CSV produced by `fyodor-train --metrics`:
step, loss, tokens/s, compute provider, optional validation loss, gradient norm,
and device memory. Press `g` to cycle loss, tokens/s, gradient norm, and device memory graphs, or
use `:chart loss|rate|gradient|memory`. Each graph uses a labeled dynamic
vertical scale; missing optional values remain gaps rather than zeroes. Only complete, finite rows are accepted. Refresh reads at most
the last 64 KiB and retains at most 192 samples; malformed rows are counted.
A missing file clears old samples. File replacement/restart is supported.

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
characters and accounts for wide Korean/CJK characters.

This is an initial inspection/monitoring workbench. It does not yet launch,
cancel, or resume training, or run interactive generation. Those controls require
engine lifecycle integration. Model inspection is synchronous. Unicode input is
supported; display width is a conservative CJK/emoji approximation, not full
Unicode grapheme segmentation. Combining marks are shown as placeholders.
Windows interaction has been exercised through ConPTY; the POSIX adapter still
needs Linux/macOS terminal validation. Terminals smaller than 48 x 16 show a
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
