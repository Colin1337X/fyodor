#ifndef NYA_TURBOQUANT_H
#define NYA_TURBOQUANT_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

/* Experimental native C TurboQuant cache. Lossy, opt-in, not F32 parity.
 * Keys: (bits-1)-bit MSE + one QJL residual bit. Values: bits-bit MSE.
 * Seed fixes the shared Gaussian rotation/sketch; no model calibration.
 * Width 32..512, bits 2..4. Objects are request-owned, not thread safe.
 * Buffers must not alias. All functions return -1 on invalid input/failure.
 * This CPU reference does not select a GPU provider or change model defaults. */
typedef struct nya_turboquant_cache nya_turboquant_cache;
nya_turboquant_cache *nya_turboquant_create(size_t width, size_t kv_heads,
    size_t capacity, unsigned bits, uint64_t seed);
void nya_turboquant_free(nya_turboquant_cache *cache);
/* Exact allocated bytes, including shared transforms and scratch. */
size_t nya_turboquant_bytes(const nya_turboquant_cache *cache);
size_t nya_turboquant_token_bytes(const nya_turboquant_cache *cache);
/* Append or overwrite an existing token; no gaps. Invalid input leaves cache
 * unchanged. Rewriting a token retains the subsequent prefix. */
int nya_turboquant_store(nya_turboquant_cache *cache, size_t position,
    const float *keys, const float *values);
int nya_turboquant_read(nya_turboquant_cache *cache, size_t position,
    float *keys, float *values);
/* Single-query causal GQA with optional sliding window. Decodes packed codes
 * inside online softmax, never materializes a full F32 KV cache/score matrix.
 * On numerical failure output is unspecified, cache remains usable. */
int nya_turboquant_attention(nya_turboquant_cache *cache, const float *query,
    size_t heads, size_t position, size_t window, float scale, float *output);
#ifdef __cplusplus
}
#endif
#endif
