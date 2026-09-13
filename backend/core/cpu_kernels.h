#ifndef NYA_CPU_KERNELS_H
#define NYA_CPU_KERNELS_H
#include <stddef.h>
typedef float (*nya_cpu_dot_function)(const unsigned char *, const float *, size_t, unsigned);
typedef void (*nya_cpu_decode_function)(const unsigned char *, float *, size_t, unsigned);
/* Decoded row tile, K-major padded input, token-major output. */
typedef void (*nya_cpu_gemm_function)(const float *, const float *, float *, size_t, size_t, size_t, size_t, size_t);
typedef int (*nya_cpu_attention_function)(const float *, const float *, const float *, float *, float *, size_t, size_t, size_t, float);
nya_cpu_dot_function nya_cpu_select_dot(const char **name);
nya_cpu_decode_function nya_cpu_select_decode(void);
nya_cpu_gemm_function nya_cpu_select_gemm(void);
nya_cpu_attention_function nya_cpu_select_attention(void);
#endif
