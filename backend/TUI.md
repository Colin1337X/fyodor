# Native terminal client: initial resource workspace

The TUI now supports the ten shared semantic presets, `--theme`, `--color`,
`--themes`, `--licenses` and interactive `t` to cycle themes. See
[`themes/README.md`](../themes/README.md) for current behavior and validation.
The initial checkpoint below predates that addition; Unicode rendering and
the remaining workspaces are still open.

Build target `fyodor-tui` is pure C. It calls the same `fyodor-client` command
dispatcher and `fyodor-app` store services as the native CLI, without spawning
CLI processes. `fyodor_terminal.c` owns terminal modes, input and safe text;
`fyodor_tui.c` owns the resource workspace and presentation state.

```powershell
.\build-cpu\fyodor-tui.exe --store C:\work\workspace.db --namespace workspace
```

The store parent directory must exist. This is a trusted local administrator
interface, matching the existing resource CLI; selecting a namespace does not
authenticate a principal. Scoped `context` commands apply their explicit
principal grants through the shared service. No remote authentication is added.

Browse resources in pages of twenty: `j`/`k` select, Enter opens a live content
preview, `n` advances, `r` returns to the first page, `b` returns to the list,
`?` opens help, and `q` exits. Detail views use `j`/`k` to scroll lines. Windows
arrow keys also select/scroll. `:` opens command entry; Enter runs it and Escape
cancels. Omit the CLI executable and workspace options, for example:

```text
resource list
resource export fyodor://writing/notes/01234567-89ab-4cde-8123-456789abcdef
context search PRINCIPAL query
```

Quoted arguments support spaces; backslashes remain literal so Windows paths
work. There is no shell expansion. Commands use the shared service directly.
Import is excluded because its package input would conflict with terminal
input; use the native CLI to import. Command text is bounded to 8 KiB and 91
arguments. Overflow is rejected rather than executing a truncated command.
Output/content previews are bounded to 64 KiB and marked when truncated.

`--plain`, redirected input/output, `TERM=dumb`, or unsupported console modes
select line mode without terminal escape sequences. Its commands are `list`,
`next`, `open N`, `back`, `help`, `:COMMAND`, and `quit`; EOF exits. Both views
escape non-ASCII bytes and controls as printable `\xNN`, preventing saved text
or command output from injecting escape/clipboard sequences. Original UTF-8
data remains intact in storage and CLI exports. This is the ASCII fallback,
not a completed Unicode-width renderer or semantic theme system.

Interactive mode saves/restores input/output modes and uses an alternate
screen. Quit, Ctrl+C/Ctrl+D input and normal process cleanup restore modes.
SIGINT/SIGTERM are observed between input polls; abrupt process termination
cannot run cleanup. Native generation remains synchronous: streaming,
cancellation during commands and model/session lifecycle UI remain open.
Size is polled and very small screens receive a resize notice. POSIX termios
support is implemented but not yet built or exercised on POSIX.

## Verification

The parser/safe-text test covers argument bounds, quoting, literal paths and
terminal-control escaping. The subprocess workflow imports 22 resources using
the real CLI, browses both pages, reads content, dispatches commands and checks
EOF, invalid arguments, oversized input, Unicode paths and safe text in both
explicit and automatic plain modes.

The Windows console test creates its own hidden console, enters the alternate
screen, drives help and a shared-service command with native key events, then
checks restored input/output modes on quit and Ctrl+C. It never accesses the
user's terminal. A simulated resize attempt was not retained by this hidden
console host; interactive resize behavior is **not yet verified**. No visual
layout, POSIX, supplementary Unicode input, full workspace UI or GPU claims
follow from these tests. All three TUI tests pass in C17, C23 and ASan/UBSan.

This is an initial resource workspace. Chat, models, training, evaluations,
benchmarks, nodes, Explore, Writing/Dataset editors, agents, settings, command
search/history, Unicode rendering and shared themes remain work in progress.
