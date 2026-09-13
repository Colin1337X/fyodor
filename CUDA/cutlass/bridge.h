/* Versioned C boundary: neither CUTLASS templates nor CUDA runtime types cross
 * into the portable engine. Device addresses and the borrowed stream remain
 * owned by Fyodor. The bridge must not allocate, synchronize or free them. */
#ifndef NYA_CUTLASS_BRIDGE_H
#define NYA_CUTLASS_BRIDGE_H
#include <stdint.h>
#if defined(_WIN32) && defined(NYA_CUTLASS_IMPLEMENTATION)
#define NYA_CUTLASS_EXPORT __declspec(dllexport)
#elif !defined(_WIN32)
#define NYA_CUTLASS_EXPORT __attribute__((visibility("default")))
#else
#define NYA_CUTLASS_EXPORT
#endif
#ifdef __cplusplus
extern "C" {
#endif
NYA_CUTLASS_EXPORT unsigned nya_cutlass_abi_version(void);
/* A: row-major [batch,k], B: row-major [rows,k], Y: row-major
 * [batch,output_stride]. This writes a possibly chunked set of output columns.
 * Return 0 on enqueue success, 1 before enqueue for unsupported arguments,
 * negative on CUDA/CUTLASS failure (caller must replay the complete request). */
NYA_CUTLASS_EXPORT int nya_cutlass_gemm(uintptr_t a, uintptr_t b, uintptr_t y,
    int batch, int rows, int k, int output_stride, uintptr_t stream);
#ifdef __cplusplus
}
#endif
#endif
