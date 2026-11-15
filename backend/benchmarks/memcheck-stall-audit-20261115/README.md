# Live memcheck setup audit

The existing session 40343 was inspected without restarting it. Compute Sanitizer PID 17656 owns lifecycle PID 11036, started at 06:32:20 local time on 2026-11-15. At 11:07 it still had no numerical checkpoint. A contemporaneous nvidia-smi snapshot showed 100% GPU utilization, 12787 MiB used, and 114.83 W; this is not continuous benchmark coverage.

GDB briefly suspended the lifecycle process for stack inspection and detached successfully. Retained stack.txt locates execution in CUDA launch -> cuda_plan_batch -> cuda_plan_prefill -> nya_llm_session_run -> main+3151. The adjacent failure branch passes 0x52 (82) to the CHECK diagnostic. Line 82 of the exact lifecycle source is the first reference prefix run. Earlier samples showed different inner launch paths, including plan_matvec_impl. This establishes setup-stage execution, not deadlock freedom or candidate correctness. Debugger inspection means this run must never be used as a timing measurement.

The harness performs three roughly 8K prefix passes before its first numerical checkpoint and four across the full lifecycle. Its Python child.wait() has no phase reporting; CPU growth cannot identify useful phase progress. The separate verbose lifecycle build fixes operation visibility for future runs, but is not installed into this running executable.

The documented kernel-name regex filter is valid (https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/). It selects candidate kernels for checking; it does not remove reference prefill from execution. The cause of the extreme setup overhead is still unresolved. No pass, failure, new performance score, or justification to waive validation follows from this audit.
