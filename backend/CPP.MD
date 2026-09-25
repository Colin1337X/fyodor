OPTIONAL C++ POLICY

Fyodor remains fundamentally a C17/C23 project.

Portable backend code, runtime architecture, public interfaces, model execution,
training, server code, backend dispatch, and fallback implementations remain C.

General-purpose C++ is permitted only under:

    /backend/optcpp

CUDA-specific C++ remains separately quarantined under /CUDA, for example:

    /CUDA/cutlass
    /CUDA/cute
    /CUDA/<other CUDA-C++ optimization>

The repository must NEVER require /backend/optcpp to function.

PHYSICAL REMOVAL REQUIREMENT

The following must be a supported configuration:

    rm -rf backend/optcpp

After physically removing that directory:

- Fyodor must still configure successfully.
- Fyodor must still compile successfully.
- CPU inference must work.
- Vulkan must work where enabled.
- CUDA must still work independently.
- ROCm/MLX must remain independent.
- Training/autograd must work.
- Server/API functionality must work.
- Frontend/Tauri packaging must work.
- Existing correctness tests must pass.
- No placeholder optcpp source files may be required.
- No C++ compiler may be required solely because optcpp once existed.
- No libstdc++/libc++ runtime dependency may remain in the ordinary C build.

C ABI BOUNDARY

Everything crossing between /backend/optcpp and the C backend must use a plain C ABI.

Public-facing headers must expose only:

- C-compatible structs
- POD values
- opaque handles
- explicit ownership/lifetime functions
- integer/status error codes
- pointer + size interfaces where appropriate

Do not expose across the boundary:

- std::string
- std::vector
- templates
- classes
- exceptions
- RTTI
- C++ references
- C++ ownership objects
- C++ ABI-dependent types

Use extern "C" only inside the C++ implementation/shim where required.

The C runtime must not know or care how an optcpp implementation works.

PURPOSE OF /backend/optcpp

/backend/optcpp exists ONLY for optional performance implementations where C++
provides a substantial measured advantage or access to useful optimized libraries.

Examples may include:

- highly tuned matrix kernels
- optional CPU vector libraries
- specialized graph optimizers
- vendor libraries that genuinely require C++
- experimentally optimized inference operators

Do not move ordinary backend logic into optcpp simply because writing it in C++
is easier.

C remains the reference and fallback implementation.

PERFORMANCE REQUIREMENT

Removing /backend/optcpp must not reduce Fyodor to a slow reference/debug engine.

The native C implementation must remain a serious optimized production path.

For every optcpp acceleration path:

1. Maintain a native C implementation.
2. Benchmark both paths on equivalent workloads.
3. Validate both against the same numerical correctness tests.
4. Preserve automatic fallback.
5. Document the measured difference.
6. Continue optimizing the C implementation when the gap is substantial.

The no-optcpp build should deliver near-equivalent practical performance wherever
reasonably achievable.

Treat a large optcpp-vs-C performance gap as an optimization target, not as an
excuse to abandon the C path.

Do not make benchmark claims that silently depend on optcpp.

Benchmark output must identify the implementation actually used.

DISPATCH

Optional C++ implementations should register through the existing backend or
operation-dispatch architecture.

Preferred conceptual structure:

    C runtime
        |
        +-- native C implementation
        |
        +-- optional optcpp implementation
                |
                +-- C ABI shim
                +-- C++ implementation

Selection should be capability/benchmark driven, with the C implementation
remaining the guaranteed fallback.

Do not scatter:

    #ifdef OPTCPP

through ordinary runtime code.

Keep detection and dispatch localized.

BUILD POLICY

GCC remains the canonical compiler for ordinary Fyodor C code.

    gcc -O3

A C++ compiler such as g++ may be used ONLY for /backend/optcpp.

Building optcpp must not cause unrelated .c files to be compiled as C++.

The build should automatically detect whether /backend/optcpp exists and whether
its required compiler/libraries are available.

If not available:

    print a concise diagnostic
    disable the optional optimization
    continue building Fyodor normally

This is not a configuration error.

CUDA C++ remains governed independently by /CUDA.

NVCC may compile CUDA-specific C++ under /CUDA without granting C++ access to
ordinary backend code.

CORRECTNESS

Optional C++ does not receive relaxed numerical standards.

A faster C++ implementation that violates established parity bounds must not
become the default.

Use the same:

- real-model logit checks
- quantized kernel oracles
- boundary tests
- long-context tests
- sanitizer/failure tests where applicable

as the corresponding C implementation.

If a mathematically legitimate lower-precision implementation needs a different
precision contract, establish and validate that contract explicitly rather than
silently changing an existing threshold.

RESOURCE OWNERSHIP

C++ implementations may internally use RAII, templates, containers, or other
C++ mechanisms inside /backend/optcpp.

None of those ownership semantics may escape through the C ABI.

The C-facing lifecycle must remain explicit:

    create
    execute
    reset
    destroy

Every failure path must release all resources before returning control to C.

Exceptions must NEVER cross the C ABI boundary.

Catch them inside optcpp and translate them into Fyodor error codes/messages.

TEST REQUIREMENTS

Add a dedicated physical-removal verification target equivalent to:

    verify-without-optcpp

It must:

1. create a fresh source-tree copy,
2. physically exclude /backend/optcpp,
3. configure the normal optimized build,
4. build with warnings enabled,
5. run the complete applicable test suite,
6. run representative inference benchmarks,
7. compare performance against the optcpp-enabled build.

Maintain both:

    correctness parity
    performance comparison

The absence of optcpp is a first-class supported configuration.

PERFORMANCE ACCEPTANCE

The target architecture is:

    optimized C baseline
            +
    optional C++ acceleration

not:

    slow C compatibility implementation
            +
    real engine hidden in C++

If optcpp becomes dramatically faster than native C, investigate why.

Extract algorithmic improvements that can reasonably be re-expressed in C and
feed them back into the native implementation.

C++ should provide the final few difficult percentage points, access to genuinely
C++-only libraries, or useful experimental/reference implementations.

It must not become the place where all serious optimization work migrates.

CUDA-SPECIFIC EXCEPTION

CUDA C++ remains confined to /CUDA.

CUTLASS:
    /CUDA/cutlass

CuTe:
    /CUDA/cute

Other CUDA-C++ experiments:
    /CUDA/<isolated-name>

These follow the same philosophy:

- optional
- detachable
- C ABI facing Fyodor
- native CUDA fallback retained
- physical-removal tests
- no C++ types exposed to core

The existence of CUDA C++ does not permit general C++ elsewhere outside:

    /CUDA/*
    /backend/optcpp/*

FINAL INVARIANT

Fyodor should always pass this conceptual test:

    rm -rf backend/optcpp
    rm -rf CUDA/cutlass
    rm -rf CUDA/cute

    # Build the optimized production application.

It should still work.

It should still be fast.

It should still be Fyodor.