# Native terminal workbench

Build `fyodor-tui` with the ordinary backend CMake build. It uses the C engine,
Windows Console or POSIX termios, and ANSI output. No Go, Rust, C++, curses,
Bubble Tea, or other terminal framework is required.

```
fyodor-tui --model E:\models\model.gguf
fyodor-tui --metrics training.csv
fyodor-tui --model model.gguf --snapshot
```

The model view performs structural container inspection, not model loading or
inference. The training view follows the CSV produced by `fyodor-train --metrics`:
step, loss, tokens/s, compute provider, optional validation loss, gradient norm,
and device memory. The graph shows the latest loss samples with a labeled dynamic
vertical scale. Only complete, finite rows are accepted. Refresh reads at most
the last 64 KiB and retains at most 192 samples; malformed rows are counted.
A missing file clears old samples. File replacement/restart is supported.

Use 1/2/3, Tab, arrows, or hjkl to switch views; `?` opens help, Escape closes it,
Space freezes metrics, `r` refreshes, and `q` or Ctrl-C exits. The `:` command bar
accepts `open PATH`, `metrics PATH`, `overview`, `model`, `training`, `pause`,
`refresh`, and `quit`. Paths include spaces without extra quoting inside the
command bar. Commands never invoke a shell. `NO_COLOR` disables color; snapshot
mode emits plain UTF-8 without terminal escape sequences. Interactive mode
requires a terminal and restores its modes and alternate screen on normal exit
and handled interruption.

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
combinations including extreme finite graph values. `tui-snapshot` checks the
plain executable path. These pass in GCC C17 CPU, GCC C23 CUDA/Vulkan-enabled,
and Clang ASan/UBSan configurations on Windows. ConPTY checks cover initial
render, help, idle rendering, q, and Ctrl-C cleanup. A real Gemma4 12B GGUF
inspection reports 667 tensors and 43 metadata entries.

The physical-removal build also passes all 65 registered tests (four expected
ROCm/MLX hardware skips) with optcpp and CUDA template directories absent,
including both TUI tests and the training/export/inference workflows.
