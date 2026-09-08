#ifndef NYA_PRETRAINING_H
#define NYA_PRETRAINING_H

#include "training.h"

struct nya_model;
typedef struct nya_train_decoder nya_train_decoder;

/* Dense LLaMA-style decoder: RMSNorm, grouped-query causal attention, RoPE,
   SwiGLU, and an untied vocabulary head. Every matrix starts from seeded random
   weights; normalization scales start at one. */
typedef struct nya_train_decoder_config {
    uint32_t vocabulary_size, embedding_length, feed_forward_length, block_count;
    uint32_t head_count, kv_head_count, context_length;
    float norm_epsilon, rope_base, initializer_std;
    uint64_t seed;
    size_t parameter_memory_limit;
} nya_train_decoder_config;

void nya_train_decoder_defaults(nya_train_decoder_config *config);
nya_train_decoder *nya_train_decoder_create(const nya_train_decoder_config *config, char *error, size_t capacity);
/* Import a native dense LLaMA (untied head) or dense Gemma 4 (including PLE and
   shared KV). MoE, MTP and multimodal encoder training are not supported yet.
   rank=0 copies graph weights into trainable F32 parameters for SFT/CPT. Positive
   rank freezes the mapped model and trains LoRA on its linear matrices. The
   source model must remain loaded throughout the decoder's lifetime. */
nya_train_decoder *nya_train_decoder_from_model(struct nya_model *model, size_t rank,
    float alpha, size_t parameter_memory_limit, char *error, size_t capacity);
void nya_train_decoder_free(nya_train_decoder *model);
nya_train_tensor *nya_train_decoder_forward(nya_train_decoder *model, nya_train_graph *graph,
    const uint32_t *tokens, size_t count);
/* Encode using the source tokenizer or the random model's byte vocabulary.
   The returned IDs are caller-owned (free). Text must be NUL-terminated UTF-8. */
int nya_train_decoder_tokenize(nya_train_decoder *model, const char *text,
    uint32_t **tokens, size_t *count, char *error, size_t capacity);
const nya_train_decoder_config *nya_train_decoder_configuration(const nya_train_decoder *model);
/* Borrowed array of unique parameters for zero_grad, AdamW and checkpoint APIs. */
nya_train_parameter *const *nya_train_decoder_parameters(nya_train_decoder *model, size_t *count);

/* Export a complete GGUF, merging LoRA while streaming weight values. LLaMA
   exports F32; Gemma converts changed tensors to F32 and preserves frozen
   tensors and metadata, omitting the stale general.file_type summary. A
   newly initialized vocabulary of 259 uses an explicit byte tokenizer:
   UNK=0, BOS=1, EOS=2, byte b=3+b. Imports preserve their source tokenizer.
   Other vocabulary sizes require a future custom-tokenizer exporter. */
int nya_train_decoder_export(nya_train_decoder *model, FILE *file, char *error, size_t capacity);

#endif
