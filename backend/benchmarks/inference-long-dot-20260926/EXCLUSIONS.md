# Benchmark exclusions

Before timing on September 27, the verified `D:\models\koboldcpp.exe` process
was stopped under the user's standing authorization. Whole-GPU memory was
over 13 GiB; its identity and the pre-stop snapshot are retained alongside
this file.

The entire first `accepted/` attempt is excluded from performance acceptance.
Its guard detected a new RustDesk GPU process during `native-0-before-b` and
aborted. Three preceding runs had completed, including a noisy short-decode
baseline; none is pooled with a later attempt. RustDesk was preserved because
it provides remote access. Raw results and the detected process records remain
unchanged in that directory.

The subsequent `accepted-remote/` attempt starts with a fresh desktop process
snapshot, including the existing remote-access client. It uses the same idle
and process-identity guards. It is an interactive-desktop comparison, not a
headless benchmark. Individual slow observations are not discarded.
