#ifndef NYA_COMPUTE_H
#define NYA_COMPUTE_H

#include <stddef.h>

/* The core only sees a C handle. Vulkan/CUDA types never cross this boundary.
 * A context owns cached immutable weight copies and must be used by one thread
 * at a time. Its lifetime must end before the model's mapped weights are freed. */
typedef struct nya_compute_context nya_compute_context;
typedef enum nya_backend_type { NYA_BACKEND_CPU, NYA_BACKEND_CUDA, NYA_BACKEND_VULKAN, NYA_BACKEND_ROCM, NYA_BACKEND_MLX } nya_backend_type;
enum { NYA_COMPUTE_MATVEC = 1U, NYA_COMPUTE_QUANTIZED = 2U, NYA_COMPUTE_RESIDENT = 4U,
       NYA_COMPUTE_MATMUL = 8U };
typedef struct nya_compute_stats {
    size_t weights_bytes, kv_bytes, scratch_bytes, prefill_batch;
    size_t external_matmul_bytes; /* included in scratch_bytes, never additive */
    unsigned long long kernel_launches, uploads, downloads, synchronizations;
    unsigned long long external_matmul_calls; /* opaque library dispatches */
    unsigned long long graph_captures, graph_replays; /* resident decode graph work */
    unsigned external_matmul_kind; /* 0 native, 1 cuBLAS, 2 CUTLASS 3xTF32, 3 rocBLAS F32, 4 MLX F32 */
    unsigned long long cutlass_matmul_calls; /* subset of external_matmul_calls */
    unsigned long long tiled_attention_calls; /* native query-tiled prefill */
} nya_compute_stats;
typedef struct nya_compute_plan nya_compute_plan;
/* Host F32 attention view: query/output are [batch,heads,width], KV is
   [capacity,kv_heads,width]. Only the causal prefix through position+batch is
   read. A nonzero window bounds that prefix per query. Inputs/output must not
   alias. The caller retains all buffers until synchronous dispatch returns. */
typedef struct nya_compute_attention {
    const float *query, *keys, *values;
    float *output;
    size_t batch, heads, kv_heads, width, capacity, position, window;
    float scale;
} nya_compute_attention;
int nya_compute_attention_f32(nya_compute_context *context, const nya_compute_attention *attention);
struct nya_llm_context;
unsigned nya_compute_capabilities(const nya_compute_context *context);
/* A plan is a request-local inference graph with persistent KV and scratch.
   NULL means the graph cannot run resident; it is not a successful no-op. */
nya_compute_plan *nya_compute_plan_create(nya_compute_context *context,
    const struct nya_llm_context *model, size_t capacity);
void nya_compute_plan_free(nya_compute_plan *plan);
int nya_compute_plan_token(nya_compute_plan *plan, unsigned int token, size_t position,
    int logits, float *output);
int nya_compute_plan_prefill(nya_compute_plan *plan, const unsigned int *tokens,
    size_t count, size_t position, float *output);
void nya_compute_plan_stats(const nya_compute_plan *plan, nya_compute_stats *stats);
/* Explicit synchronous diagnostic snapshot, expanded to F32. The two output
   arrays each hold exactly elements values and must not overlap. Normal
   inference never calls this readback. Only an already valid prefix is readable. */
int nya_compute_plan_read_kv(nya_compute_plan *plan, size_t layer, size_t position,
    size_t count, float *keys, float *values, size_t elements);

/* Default is CPU. NYA_COMPUTE selects cpu, vulkan, cuda, rocm or mlx.
 * Failure or unavailable hardware returns NULL, leaving CPU execution intact. */
nya_compute_context *nya_compute_create(void);
/* Explicit selection for applications with per-model controls. This does not
 * mutate the process environment or affect other loaded models. NULL denotes
 * unavailable hardware or initialization failure. A CPU handle owns SIMD
 * dispatch and a worker pool; NULL remains the scalar-reference fallback. */
nya_compute_context *nya_compute_create_for(const char *provider);
int nya_compute_backend_known(const char *provider);
int nya_compute_backend_compiled(const char *provider);
/* Assisted providers own a context cache rather than a resident transformer
   plan. This snapshot reports their actual retained resources and dispatches. */
void nya_compute_context_stats(const nya_compute_context *context, nya_compute_stats *stats);
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
/* Batch-major activations/output; each batch entry is a contiguous vector.
   Weights retain their GGUF row-major storage. */
int nya_compute_matmul_typed(nya_compute_context *context, const void *weights,
    size_t rows, size_t columns, unsigned type, const float *input, float *output, size_t batch);

/* These report the actual active provider, not merely a compiled capability. */
const char *nya_compute_name(const nya_compute_context *context);
int nya_compute_vulkan_compiled(void);
int nya_compute_cuda_compiled(void);

#endif
