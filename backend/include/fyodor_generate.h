#ifndef FYODOR_GENERATE_H
#define FYODOR_GENERATE_H
#include "fyodor_context.h"
#include "generation.h"
#include "model.h"
#include "fyodor_receipt.h"

typedef struct {
    fyodor_context_bundle context;
    char *prompt; /* exact engine input, including separator and user prompt */
    nya_generation_response generation;
    size_t model_context_length;
    fyodor_uuid receipt_id;
} fyodor_context_generation;

/* Explicit context and real native inference. The caller owns/keeps the loaded
 * model alive and serializes its use. All context is authorized before inference.
 * Native tokenizer checks reserve ALL requested output tokens. If necessary,
 * context is halved at UTF-8 boundaries until it fits; the user prompt is never
 * truncated. This is deterministic, conservative, not maximal context packing.
 * Sampling request.prompt is the user prompt. Draft/soft tokens are not yet
 * supported by this bridge. Returned attribution reflects actual included bytes.
 * Success requires durable receipt storage. If persistence fails after native
 * inference, returns an error (inference cannot be rolled back). No successful
 * receipt is claimed. On any failure output remains unchanged. */
fyodor_store_result fyodor_generate_with_context(fyodor_store *store,nya_model *model,
    const fyodor_uuid *principal,const char *name_space,const fyodor_resource_ref *refs,
    size_t count,size_t byte_budget,const nya_generation_request *request,
    fyodor_context_generation *out,char *error,size_t error_capacity);
/* Same native generation policy using priority-ordered context layers. */
fyodor_store_result fyodor_generate_with_layers(fyodor_store *store,nya_model *model,
    const fyodor_uuid *principal,const char *name_space,const fyodor_context_entry *entries,
    size_t count,size_t byte_budget,const nya_generation_request *request,
    fyodor_context_generation *out,char *error,size_t error_capacity);
typedef enum { FYODOR_WRITING_GENERATE, FYODOR_WRITING_REWRITE, FYODOR_WRITING_CONTINUE } fyodor_writing_mode;
/* Preview only: never modifies the target. Target is a saved Writing resource,
 * first in explicit context, with its observed revision checked on the same
 * assembly snapshot before inference. Extra sources retain their layer policy.
 * Readable parent-project text is discovered as workspace context in that
 * same snapshot; denied parents stop traversal and never grant access.
 * request.prompt supplies writing instructions; the native service adds the
 * action prefix. Saved receipts retain exact prompt, source revisions/output. */
fyodor_store_result fyodor_writing_generate(fyodor_store *store,nya_model *model,
    const fyodor_uuid *principal,const char *name_space,const fyodor_resource_ref *target,
    uint64_t expected_revision,fyodor_writing_mode mode,const fyodor_context_entry *extra,
    size_t count,size_t byte_budget,const nya_generation_request *request,
    fyodor_context_generation *out,char *error,size_t error_capacity);
void fyodor_context_generation_free(fyodor_context_generation *result);
#endif
