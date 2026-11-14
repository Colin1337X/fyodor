# Bounded Q4 column-index experiment

Offline SM120 instruction inspection found 64-bit column induction/address
work in the fully unrolled metadata kernel. Partial unrolling reduced register
allocation but regressed all eight measured shapes. This candidate keeps full
unrolling and changes only column induction to unsigned 32-bit arithmetic.

It explicitly rejects column counts above 0xffffff00 or not divisible by 256;
this bounds the last increment and all group/lane offsets without wraparound.
Weight row offsets and strides remain 64-bit. Production integration would
need a matching host admission condition and fallback for unsupported sizes.
Floating-point expressions, group order, metadata decoding and warp reduction
remain unchanged. This is original Fyodor code under optional optcpp, with
no production dispatch. Compilation alone proves neither correctness nor speed.
Raw parity, boundary guards, sanitizers and multi-shape timing remain required.

NVRTC compilation and offline SM120 assembly passed with 66 registers and
zero stack, spill or shared-memory bytes. No GPU kernel has executed yet.
