#ifndef FYODOR_CONTEXT_H
#define FYODOR_CONTEXT_H
#include "fyodor_store.h"

enum {
    FYODOR_PERMISSION_READ=1, FYODOR_PERMISSION_WRITE=2,
    FYODOR_PERMISSION_SEARCH=4, FYODOR_PERMISSION_APPEND=8,
    FYODOR_PERMISSION_DELETE=16, FYODOR_PERMISSION_EXECUTE=32,
    FYODOR_PERMISSION_ALL=63
};

/* Trusted local administrator operation. Replaces the exact principal/resource
 * grant; zero revokes. Never infer principal identity from model/package data.
 * Principals are canonical non-nil UUIDs supplied by an authenticated caller.
 * No implicit namespace-wide, parent, owner or remote-node grants exist. */
fyodor_store_result fyodor_context_permissions_set(fyodor_store *store,
    const fyodor_uuid *principal,const char *name_space,const fyodor_resource_ref *ref,unsigned permissions);
fyodor_store_result fyodor_context_permissions_get(fyodor_store *store,
    const fyodor_uuid *principal,const char *name_space,const fyodor_resource_ref *ref,unsigned *permissions);
/* Principal-scoped read. Grant and data are checked in one read transaction.
 * Missing grants/resources return DENIED without revealing which was missing.
 * Revocation affects subsequent snapshots, not text already disclosed. */
fyodor_store_result fyodor_context_read(fyodor_store *store,const fyodor_uuid *principal,
    const char *name_space,const fyodor_resource_ref *ref,fyodor_resource_record *out);
/* Existing-resource mutations require their exact independent permission and
 * a nonzero observed revision. Grant check, revision check, mutation and history
 * are one writer transaction. READ is not implied. Append changes only content,
 * preserving title/metadata/provenance; returns only the new revision. No
 * operation here can create resources, change grants or bypass parent rules. */
fyodor_store_result fyodor_context_write(fyodor_store *store,const fyodor_uuid *principal,
    const fyodor_resource_input *input,uint64_t expected_revision,uint64_t *revision);
fyodor_store_result fyodor_context_append(fyodor_store *store,const fyodor_uuid *principal,
    const char *name_space,const fyodor_resource_ref *ref,const char *suffix,
    uint64_t expected_revision,uint64_t *revision);
fyodor_store_result fyodor_context_delete(fyodor_store *store,const fyodor_uuid *principal,
    const char *name_space,const fyodor_resource_ref *ref,uint64_t expected_revision);
/* Literal case-insensitive ASCII substring search; non-ASCII exact bytes.
 * Persistent trigram index for 3+ Unicode scalars; shorter queries scan.
 * Requires READ and SEARCH. Title matches rank ahead of content-only matches,
 * then URI byte order. 1..100 summaries; query 1..128 UTF-8 bytes. */
fyodor_store_result fyodor_context_search(fyodor_store *store,const fyodor_uuid *principal,
    const char *name_space,const char *query,size_t limit,fyodor_resource_summary **items,size_t *count);

typedef enum {
    FYODOR_CONTEXT_EXPLICIT, FYODOR_CONTEXT_WORKSPACE, FYODOR_CONTEXT_SESSION,
    FYODOR_CONTEXT_RETRIEVED, FYODOR_CONTEXT_GLOBAL, FYODOR_CONTEXT_LAYER_COUNT
} fyodor_context_layer;
const char *fyodor_context_layer_name(fyodor_context_layer layer);
typedef struct { fyodor_resource_ref ref; fyodor_context_layer layer; } fyodor_context_entry;
typedef struct {
    fyodor_resource_ref ref;
    uint64_t revision;
    size_t offset, length, original_length;
    fyodor_context_layer layer;
} fyodor_context_source;
typedef struct {
    char *text;
    size_t length, source_count;
    fyodor_context_source *sources;
} fyodor_context_bundle;
/* Explicit sources in caller order; duplicates rejected. At most 64 sources,
 * budget 0..1 MiB of content bytes, not tokens. Every source must be readable,
 * including sources past the budget. All checks/reads use one snapshot.
 * UTF-8 prefixes only; newline separators count against the budget. Attribution
 * includes zero-length omitted sources. Failure leaves output unchanged. */
fyodor_store_result fyodor_context_assemble(fyodor_store *store,const fyodor_uuid *principal,
    const char *name_space,const fyodor_resource_ref *refs,size_t count,size_t byte_budget,
    fyodor_context_bundle *out);
/* Stable layer priority: explicit, workspace, session, retrieved, global.
 * Caller order within each layer; duplicate URIs across any layers rejected.
 * All sources require READ; retrieved entries additionally require SEARCH.
 * No layer grants access or implicitly enumerates resources. Same snapshot,
 * total byte budget and failure guarantees as explicit assembly above. */
fyodor_store_result fyodor_context_assemble_layers(fyodor_store *store,const fyodor_uuid *principal,
    const char *name_space,const fyodor_context_entry *entries,size_t count,size_t byte_budget,
    fyodor_context_bundle *out);
void fyodor_context_bundle_free(fyodor_context_bundle *bundle);
/* Writing provider: saved target first/explicit, then caller extras, then
 * readable linked lore and parent projects in the workspace layer. Each
 * resource's ordered lore precedes its parent (nearest first). Discovery,
 * revision check, permissions and text assembly share one read snapshot.
 * A denied parent stops discovery without revealing its content/ancestors.
 * Caller extras retain their layer and permission policy; discovered duplicates
 * are omitted. At most 64 sources/traversed ancestors; extras reserve slots.
 * Denied lore is skipped. No implicit grants, sibling enumeration or
 * cross-namespace traversal. */
fyodor_store_result fyodor_writing_assemble(fyodor_store *store,const fyodor_uuid *principal,
    const char *name_space,const fyodor_resource_ref *target,uint64_t expected_revision,
    const fyodor_context_entry *extra,size_t count,size_t byte_budget,fyodor_context_bundle *out);
#endif
