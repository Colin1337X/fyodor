#ifndef NYA_COMPUTE_H
#define NYA_COMPUTE_H

#include <stddef.h>

/* The core only sees a C handle. Vulkan/CUDA types never cross this boundary.
 * A context owns cached immutable weight copies and must be used by one thread
 * at a time. Its lifetime must end before the model's mapped weights are freed. */
typedef struct nya_compute_context nya_compute_context;

/* Default is CPU. NYA_COMPUTE=vulkan or cuda selects a compiled provider.
 * Failure or unavailable hardware returns NULL, leaving CPU execution intact. */
nya_compute_context *nya_compute_create(void);
/* Explicit selection for applications with per-model controls. This does not
 * mutate the process environment or affect other loaded models. NULL denotes
 * CPU, unavailable hardware, or initialization failure; inspect name on return. */
nya_compute_context *nya_compute_create_for(const char *provider);
void nya_compute_free(nya_compute_context *context);

/* Try row-major F32 matrix-vector multiplication. Zero means output is ready;
 * -1 means the caller must execute its CPU kernel. Weights are borrowed,
 * immutable, and must remain alive until context destruction. Input/output
 * contain columns/rows floats respectively and must not alias the weights. */
int nya_compute_matvec(nya_compute_context *context, const void *weights,
    size_t rows, size_t columns, const float *input, float *output);

/* GGML storage IDs: 0 F32, 1 F16, 2 Q4_0, 8 Q8_0, 12 Q4_K, 14 Q6_K,
   30 BF16. CUDA can keep these compressed on device; other providers decline
   unsupported types so the caller executes the native CPU kernel. */
int nya_compute_matvec_typed(nya_compute_context *context, const void *weights,
    size_t rows, size_t columns, unsigned int type, const float *input, float *output);

/* These report the actual active provider, not merely a compiled capability. */
const char *nya_compute_name(const nya_compute_context *context);
int nya_compute_vulkan_compiled(void);
int nya_compute_cuda_compiled(void);

#endif
