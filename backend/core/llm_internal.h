#ifndef NYA_LLM_INTERNAL_H
#define NYA_LLM_INTERNAL_H

#include "generation.h"
#include "compute.h"

#include <stddef.h>
#include <stdint.h>

/* Generation intentionally supports a narrow, audited set of GGML tensor types. */
enum nya_llm_tensor_type {
    NYA_LLM_TENSOR_F32 = 0,
    NYA_LLM_TENSOR_F16 = 1,
    NYA_LLM_TENSOR_Q4_0 = 2,
    NYA_LLM_TENSOR_Q8_0 = 8,
    NYA_LLM_TENSOR_Q4_K = 12,
    NYA_LLM_TENSOR_Q6_K = 14,
    NYA_LLM_TENSOR_BF16 = 30
};

/* Tensor names and dimensions are bounded before they enter this structure. */
typedef struct nya_llm_tensor {
    char name[128];
    uint32_t type;
    uint32_t dimension_count;
    uint64_t dimensions[4];
    uint64_t relative_offset;
    const unsigned char *data;
    size_t data_size;
    int bound;
} nya_llm_tensor;

/* One transformer block binds the exact weights used by LLaMA inference. */
typedef struct nya_llm_layer {
    const nya_llm_tensor *attention_norm;
    const nya_llm_tensor *query;
    const nya_llm_tensor *key;
    const nya_llm_tensor *value;
    const nya_llm_tensor *attention_output;
    const nya_llm_tensor *feed_forward_norm;
    const nya_llm_tensor *feed_forward_gate;
    const nya_llm_tensor *feed_forward_down;
    const nya_llm_tensor *feed_forward_up;
    /* Gemma uses separate head norms, post-branch norms and, optionally, a
       routed expert branch plus a token-dependent per-layer residual. */
    const nya_llm_tensor *query_norm, *key_norm, *attention_post_norm, *ffn_post_norm;
    const nya_llm_tensor *router, *router_scale, *expert_scale;
    const nya_llm_tensor *expert_gate, *expert_up, *expert_gate_up, *expert_down;
    const nya_llm_tensor *expert_pre_norm, *shared_post_norm, *expert_post_norm;
    const nya_llm_tensor *ple_gate, *ple_projection, *ple_norm, *output_scale;
    const nya_llm_tensor *rope_factors;
    uint32_t head_count, kv_head_count, head_dimension, hidden_length;
    uint32_t expert_hidden_length, sliding_window, kv_source;
    /* Cache offsets count float elements per sequence position. Shared layers
       point at their source's slice and allocate no duplicate KV storage. */
    size_t cache_offset;
} nya_llm_layer;

/* Sorted vocabulary entries make exact token lookup logarithmic and simple. */
typedef struct nya_llm_token_index {
    const char *piece;
    uint32_t token;
} nya_llm_token_index;

/* A BPE rule is a pair, not merely its concatenated spelling: two different
   partitions of the same text may have different ranks or no rule at all. */
typedef struct nya_llm_bpe_rule {
    char *text;
    size_t left_length;
    uint32_t rank;
} nya_llm_bpe_rule;

/* The tokenizer owns strings copied out of untrusted model metadata. */
typedef struct nya_llm_tokenizer {
    char **pieces;
    size_t *piece_lengths;
    float *scores;
    int32_t *types;
    nya_llm_token_index *sorted;
    uint32_t vocabulary_size;
    size_t maximum_piece_length;
    int32_t byte_tokens[256];
    uint32_t unknown_token;
    uint32_t beginning_token;
    uint32_t end_token;
    int add_beginning_token;
    int add_end_token;
    int add_space_prefix;
    int uses_sentencepiece_space;
    int uses_merge_ranks;
    nya_llm_bpe_rule *merges;
    size_t merge_count;
    unsigned char *stop_tokens;
    uint32_t *suppressed_tokens;
    size_t suppressed_count;
    uint32_t *special_tokens;
    size_t special_count;
} nya_llm_tokenizer;

/* Native handles use integers so platform headers stay inside the loader. */
typedef struct nya_llm_mapping {
    unsigned char *data;
    size_t size;
    uintptr_t file_handle;
    uintptr_t mapping_handle;
} nya_llm_mapping;

/* Encoder-free Gemma 4 Unified projectors. Each tensor stays in its validated
   mapping; requests own only small normalization scratch and output vectors. */
typedef struct nya_llm_projector {
    char vision_type[32], audio_type[32];
    uint32_t vision_width, audio_width, output_width, audio_output_width;
    uint32_t vision_blocks, audio_blocks, patch_size, position_count;
    float vision_epsilon, audio_epsilon;
    const nya_llm_tensor *patch, *patch_bias, *position, *vision, *audio;
    const nya_llm_tensor *norm_weight[3], *norm_bias[3];
} nya_llm_projector;

/* One context owns immutable tensors and tokenizer data for a loaded GGUF model. */
typedef struct nya_llm_context {
    /* Optional accelerator storage belongs to the same lifetime as its mapped
       tensor addresses. A NULL provider always leaves native CPU inference usable. */
    nya_compute_context *compute;
    nya_llm_mapping mapping;
    /* Validated GGUF metadata prefix, retained for lossless training export.
       Tensor directories/payloads are rebuilt; tokenizer and architecture
       metadata remain byte-for-byte identical to the imported model. */
    size_t metadata_end;
    size_t file_type_begin, file_type_end;
    uint64_t gguf_metadata_count;
    uint32_t tensor_alignment;
    nya_llm_tensor *tensors;
    size_t tensor_count;
    nya_llm_layer *layers;
    nya_llm_tokenizer tokenizer;
    nya_llm_projector projector;
    const nya_llm_tensor *token_embedding;
    const nya_llm_tensor *output_norm;
    const nya_llm_tensor *output;
    uint32_t embedding_length;
    uint32_t feed_forward_length;
    uint32_t block_count;
    uint32_t head_count;
    uint32_t key_value_head_count;
    uint32_t context_length;
    uint32_t rope_dimension_count;
    float norm_epsilon;
    float rope_frequency_base;
    int is_gemma;
    int is_assistant;
    uint32_t backbone_length, nextn_layers;
    const nya_llm_tensor *nextn_pre, *nextn_post;
    uint32_t head_dimension, sliding_head_dimension, sliding_window, shared_kv_layers;
    uint32_t per_layer_embedding_length, expert_count, experts_used;
    float sliding_rope_base, final_logit_softcap;
    const nya_llm_tensor *ple_embedding, *ple_projection, *ple_norm;
    size_t cache_width, maximum_query_length, maximum_hidden_length;
    /* GGUF permits either a scalar or an array for these settings. Keep the
       parsed count so a truncated per-layer array cannot inherit silent zeros. */
    uint32_t layer_heads[256], layer_kv_heads[256], layer_hidden[256];
    uint32_t layer_expert_hidden[256], layer_sliding[256];
    uint32_t heads_count, kv_heads_count, hidden_count, expert_hidden_count, sliding_count;
    uint32_t value_dimension, sliding_value_dimension, sliding_rope_dimension;
} nya_llm_context;

/* Load, validate, and bind one supported GGUF LLaMA model. */
int nya_llm_load(
    const char *path,
    uint64_t expected_size,
    nya_llm_context **output,
    char *error,
    size_t error_capacity
);

int nya_llm_load_projector(const char *path, uint64_t expected_size,
    nya_llm_context **output, char *error, size_t error_capacity);

/* Shared byte-safe kernels are private to the native providers. */
float nya_llm_tensor_value(const nya_llm_tensor *tensor, size_t index);
void nya_llm_matvec(nya_compute_context *compute, float *output,
    const nya_llm_tensor *matrix, const float *input, size_t columns, size_t rows);

/* Release every allocation and mapping owned by one context. */
void nya_llm_free(nya_llm_context *context);

/* Encode a prompt into caller-owned token storage. */
int nya_llm_tokenize(
    const nya_llm_context *context,
    const char *text,
    uint32_t **tokens,
    size_t *token_count,
    char *error,
    size_t error_capacity
);

/* Append one decoded token piece into a bounded output buffer. */
int nya_llm_decode_token(
    const nya_llm_context *context,
    uint32_t previous_token,
    uint32_t token,
    char *output,
    size_t output_capacity,
    size_t *output_length
);

/* Byte-fallback tokens may temporarily split UTF-8; validate the final sequence
   before returning text to a caller such as the JSON HTTP serializer. */
int nya_llm_text_is_utf8(const char *text, size_t length);

/* Run the CPU transformer and sampling loop. */
int nya_llm_generate(
    const nya_llm_context *context,
    const nya_generation_request *request,
    nya_generation_response *response,
    char *error,
    size_t error_capacity
);

/* Exact rejection sampling uses the target distribution to verify a compatible
   draft. Passing NULL keeps the ordinary autoregressive path. */
int nya_llm_generate_draft(const nya_llm_context *context, const nya_llm_context *draft,
    const nya_generation_request *request, nya_generation_response *response,
    char *error, size_t error_capacity);

#endif
