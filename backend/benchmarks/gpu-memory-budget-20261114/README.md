# In-process GPU memory budget diagnostic

The original C helper queries IDXGIAdapter3 through its C ABI in the generation
process. It records local and nonlocal memory budget and usage immediately
before and after decode, outside the measured decode endpoints. It changes no
reservations. Adapters are matched by LUID, not enumeration order. Strict C17
and C23 standalone builds passed; build_generation.py retains the isolated
generation source, patch, build commands and hashes. Production code is unchanged.

Run build_generation.py, then run.py from this directory's repository checkout.
Existing output directories are refused. The short diagnostic retains continuous
PDH coverage, the idle gate, output parity and raw budget observations.

All twelve generation cases passed parity. On the CUDA adapter, all 24 boundary
checkpoints were below budget: local usage 13,977,518,080–14,254,342,144 bytes
against 15,887,024,128 bytes; nonlocal usage 62,914,560 bytes against
15,899,742,208 bytes. summary.json retains the observations and evidence hashes.

This rules out an observed budget overrun at these checkpoints, not transient
paging during decode. System PID 4 Copy-engine activity still invalidated every
decode interval. The collector remained healthy. No timing is accepted, no
ownership is inferred for PID 4, and the benchmark gate is unchanged. Accepted
Gemma scores remain 85.61025294% short and 106.8707255% at 8K versus llama.cpp.

DXGI semantics: https://learn.microsoft.com/en-us/windows/win32/api/dxgi1_4/nf-dxgi1_4-idxgiadapter3-queryvideomemoryinfo
