#ifndef FYODOR_RECEIPT_H
#define FYODOR_RECEIPT_H
#include "fyodor_context.h"

typedef struct {
    const fyodor_context_bundle *context;
    const char *prompt, *output, *model_path, *stop_reason;
    uint64_t model_file_size, seed;
    size_t prompt_tokens, generated_tokens, max_tokens, top_k, model_context_length;
    float temperature, top_p;
    const char *writing_mode; /* NULL for non-Writing generation */
} fyodor_receipt_input;
typedef struct {
    fyodor_uuid id;
    char *prompt, *output, *metadata, *sources;
    int64_t created_ms;
} fyodor_receipt;
/* Trusted completion recorder, not an arbitrary remote import API. Stores one
 * immutable, schema-versioned executed-request receipt atomically. */
fyodor_store_result fyodor_receipt_save(fyodor_store *store,const fyodor_uuid *principal,
    const char *name_space,const fyodor_receipt_input *input,fyodor_uuid *id);
/* Owner AND current READ grants for every source are required. Revocation or
 * deletion hides saved receipts too. Missing/other-owner receipts return DENIED.
 * Output unchanged on failure. Text/JSON strings owned by result. */
fyodor_store_result fyodor_receipt_read(fyodor_store *store,const fyodor_uuid *principal,
    const char *name_space,const fyodor_uuid *id,fyodor_receipt *out);
/* Trusted local administrator save of an accepted Writing preview, optionally
 * edited. Receipt owner/current READ on all sources, original target/revision,
 * CAS update and provenance/history/index changes share one writer transaction.
 * No implicit grant changes. Preserves metadata and other provenance fields.
 * Output unchanged on failure; old receipts without writing_mode are ineligible. */
fyodor_store_result fyodor_writing_save(fyodor_store *store,const fyodor_uuid *principal,
    const char *name_space,const fyodor_resource_ref *target,uint64_t expected_revision,
    const fyodor_uuid *receipt_id,const char *title,const char *content,uint64_t *revision);
void fyodor_receipt_free(fyodor_receipt *receipt);
typedef struct { fyodor_uuid id; int64_t created_ms; } fyodor_receipt_summary;
/* One authorized snapshot, UUID order, 1..100 entries. after_id may be NULL.
 * Returns only this principal/namespace's receipts whose every source still
 * has READ. No counts/previews for denied receipts. Outputs unchanged on error. */
fyodor_store_result fyodor_receipt_list(fyodor_store *store,const fyodor_uuid *principal,
    const char *name_space,const fyodor_uuid *after_id,size_t limit,
    fyodor_receipt_summary **items,size_t *count);
void fyodor_receipt_summaries_free(fyodor_receipt_summary *items);
#endif
