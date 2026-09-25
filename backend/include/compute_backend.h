#ifndef NYA_COMPUTE_BACKEND_H
#define NYA_COMPUTE_BACKEND_H
#include "compute.h"
#include <stdint.h>

/* Provider contract, deliberately separate from application-facing handles.
   No driver types cross this interface. Buffers are byte-addressed allocations
   owned by one context. Plans own mutable activations/KV; model tensors remain
   immutable and borrowed for the context lifetime. Unsupported plans return
   NULL before execution, allowing the reference transformer to take over. */
struct nya_llm_context;
typedef struct nya_backend_interface {
    nya_backend_type type;
    const char *name;
    unsigned capabilities;
    void *(*create)(void);
    void (*destroy)(void *);
    int (*active)(const void *);
    int (*matvec)(void *, const void *, size_t, size_t, unsigned, const float *, float *);
    void *(*plan_create)(void *, const struct nya_llm_context *, size_t);
    void (*plan_free)(void *);
    int (*plan_token)(void *, uint32_t, size_t, int, float *);
    int (*plan_prefill)(void *, const uint32_t *, size_t, size_t, float *);
    void (*plan_stats)(const void *, nya_compute_stats *);
    int (*matmul)(void *, const void *, size_t, size_t, unsigned, const float *, float *, size_t);
    int (*attention_f32)(void *, const nya_compute_attention *);
    int (*plan_read_kv)(void *, size_t, size_t, size_t, float *, float *, size_t);
    void (*context_stats)(const void *, nya_compute_stats *);
} nya_backend_interface;
#endif
