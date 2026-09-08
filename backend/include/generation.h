#ifndef NYA_GENERATION_H
#define NYA_GENERATION_H

#include <stddef.h>
#include <stdint.h>

/* Forward declaration keeps model ownership out of the public request types. */
struct nya_model;

/* Replace a contiguous span of tokenized prompt positions with projected
   embeddings. Values are already in language-model space (no sqrt(D) rescale).
   Each vision span is one image/frame; local attention is bidirectional inside
   that span. Audio spans retain causal attention. The caller owns all storage. */
typedef struct nya_generation_soft_tokens {
    size_t position;
    size_t count;
    size_t embedding_length;
    const float *embeddings;
    size_t embedding_count;
    int vision;
} nya_generation_soft_tokens;

/* A generation request contains only bounded, deterministic sampling controls. */
typedef struct nya_generation_request {
    /* The prompt is borrowed UTF-8 text and must remain valid for the call. */
    const char *prompt;

    /* This many new tokens may be sampled after the complete prompt. */
    size_t max_tokens;

    /* Zero selects greedy decoding; positive values scale model logits. */
    float temperature;

    /* Nucleus sampling retains this probability mass after the top-k filter. */
    float top_p;

    /* Zero disables the top-k filter; positive values keep at most this many. */
    size_t top_k;

    /* The caller supplies the exact seed so responses can be reproduced. */
    uint64_t seed;

    /* Generated UTF-8 bytes may not exceed this caller-selected boundary. */
    size_t max_output_bytes;
    /* Optional separately loaded draft. Its tokenizer must have exactly the
       same token IDs and normalization rules as the target. Both model objects
       and their mappings must stay alive for this synchronous call. */
    struct nya_model *draft_model;
    /* Maximum proposals per round, 1..32; zero chooses four when a draft exists. */
    size_t speculative_tokens;
    /* Spans must be sorted, non-overlapping, and fit the tokenized prompt.
       The prompt's placeholder IDs still occupy one position per soft token. */
    const nya_generation_soft_tokens *soft_tokens;
    size_t soft_token_spans;
} nya_generation_request;

/* These stable reasons explain why generation stopped. */
typedef enum nya_generation_stop_reason {
    NYA_GENERATION_STOP_LENGTH = 0,
    NYA_GENERATION_STOP_EOS = 1,
    NYA_GENERATION_STOP_CONTEXT = 2
} nya_generation_stop_reason;

/* The generation provider owns the returned text until it is explicitly freed. */
typedef struct nya_generation_response {
    char *text;
    size_t text_length;
    size_t prompt_tokens;
    size_t generated_tokens;
    uint64_t seed;
    nya_generation_stop_reason stop_reason;
    size_t draft_tokens;
    size_t accepted_draft_tokens;
    size_t target_steps;
} nya_generation_response;

/* Attach a generation provider when the model is a supported local LLM. */
int nya_generation_attach(struct nya_model *model, char *error, size_t error_capacity);

/* Release one model's tokenizer, tensor bindings, and memory mapping. */
void nya_generation_detach(struct nya_model *model);

/* Generate synchronously using CPU kernels and any selected accelerator. */
int nya_generation_run(
    struct nya_model *model,
    const nya_generation_request *request,
    nya_generation_response *response,
    char *error,
    size_t error_capacity
);

/* Release text allocated by the generation provider. */
void nya_generation_response_free(nya_generation_response *response);

/* Convert a stop reason into its stable JSON name. */
const char *nya_generation_stop_reason_name(nya_generation_stop_reason reason);

#endif
