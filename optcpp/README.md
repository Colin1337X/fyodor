# Optional non-C optimization components

This root directory is physically removable. General-purpose C++ implementations,
generated kernel objects and their private dependencies belong here. CUDA/CUTLASS
and other dedicated accelerator compartments remain separately removable.
The public runtime and dispatch stay C17/C23. Any future C++ host bridge must
expose only a plain C ABI and translate exceptions internally.

Attention integration status and its upstream audit are in attention/README.md.
No FA4 or AMD support is advertised merely because the optional directory exists.
