#ifndef FYODOR_STORE_H
#define FYODOR_STORE_H

#include "fyodor_resource.h"

#ifdef __cplusplus
extern "C" {
#endif

#define FYODOR_STORE_SCHEMA 4
#define FYODOR_STORE_CONTENT_LIMIT (1024u * 1024u)
#define FYODOR_STORE_METADATA_LIMIT (64u * 1024u)
#define FYODOR_STORE_PACKAGE_LIMIT (8u * 1024u * 1024u)

typedef struct fyodor_store fyodor_store;
typedef enum {
    FYODOR_STORE_OK = 0,
    FYODOR_STORE_INVALID = -1,
    FYODOR_STORE_NOT_FOUND = -2,
    FYODOR_STORE_CONFLICT = -3,
    FYODOR_STORE_BUSY = -4,
    FYODOR_STORE_IO = -5,
    FYODOR_STORE_SCHEMA_ERROR = -6,
    FYODOR_STORE_NOMEM = -7,
    FYODOR_STORE_DENIED = -8
} fyodor_store_result;

/* Trusted local persistence boundary, NOT a remote authorization API.
 * Callers serialize operations on each handle. Separate handles/processes
 * may access the same local file; writes use revision compare-and-swap.
 * Paths are UTF-8. Parent directories must already exist. Unknown/non-Fyodor
 * databases are rejected, never adopted or migrated implicitly. */
fyodor_store_result fyodor_store_open(const char *path, fyodor_store **out);
void fyodor_store_close(fyodor_store *store);

typedef struct {
    fyodor_resource_ref ref;
    const char *name_space; /* 1..64 ASCII letters, digits, underscore or hyphen */
    const char *title;      /* UTF-8, <= 1024 bytes, no control characters */
    const char *content;    /* UTF-8 text, may contain newlines; no embedded NUL */
    const char *metadata;   /* JSON object, bounded by METADATA_LIMIT */
    const char *provenance; /* JSON object, bounded by METADATA_LIMIT */
} fyodor_resource_input;

typedef struct {
    fyodor_resource_ref ref;
    char *name_space, *title, *content, *metadata, *provenance;
    uint64_t revision;
    int64_t created_ms, modified_ms;
    int deleted;
} fyodor_resource_record;

/* expected_revision=0 creates, otherwise updates an existing live resource.
 * Every successful mutation appends history atomically. Revisions start at 1.
 * Child creation requires a live parent in the SAME namespace. Identity and
 * namespace cannot be moved by an update. Output stays unchanged on failure. */
fyodor_store_result fyodor_store_put(fyodor_store *store, const fyodor_resource_input *input,
                                    uint64_t expected_revision, uint64_t *revision);
/* revision=0 reads the live head; explicit revisions also expose tombstones.
 * Returned strings are owned by the record, released with record_free.
 * Output must not own an earlier record; it stays unchanged on failure. */
fyodor_store_result fyodor_store_get(fyodor_store *store, const char *name_space,
                                    const fyodor_resource_ref *ref, uint64_t revision,
                                    fyodor_resource_record *out);
void fyodor_resource_record_free(fyodor_resource_record *record);
/* Soft deletion preserves history and prevents identity reuse. A live parent
 * with live children cannot be deleted. expected_revision must be nonzero. */
fyodor_store_result fyodor_store_delete(fyodor_store *store, const char *name_space,
                                       const fyodor_resource_ref *ref, uint64_t expected_revision);

typedef struct {
    fyodor_resource_ref ref;
    char *title;
    uint64_t revision;
    int64_t modified_ms;
} fyodor_resource_summary;
/* Keyset page, URI byte order, live heads only. after_uri is NULL for the first
 * page, otherwise a canonical URI. limit is 1..100. Each page is one database
 * snapshot; separate pages may see concurrent changes. Outputs unchanged on
 * failure. Free the returned page even when count is zero. */
fyodor_store_result fyodor_store_list(fyodor_store *store, const char *name_space,
                                     const char *after_uri, size_t limit,
                                     fyodor_resource_summary **items, size_t *count);
/* Writing-only page (documents, notes, characters and projects), same cursor rules. */
fyodor_store_result fyodor_store_list_writing(fyodor_store *store,const char *name_space,
    const char *after_uri,size_t limit,fyodor_resource_summary **items,size_t *count);
fyodor_store_result fyodor_store_list_projects(fyodor_store *store,const char *name_space,
    const char *after_uri,size_t limit,fyodor_resource_summary **items,size_t *count);
/* Direct children of a Writing project, or unfiled Writing resources when
 * parent is NULL. Membership never grants context access. Same paging rules. */
fyodor_store_result fyodor_store_list_folder(fyodor_store *store,const char *name_space,
    const fyodor_resource_ref *parent,const char *after_uri,size_t limit,
    fyodor_resource_summary **items,size_t *count);
/* Trusted local administrator move. NULL parent moves to the root. Preserves
 * identity/content/provenance and other metadata, appends a CAS-checked revision.
 * metadata.writing_parent is reserved: absent at root, otherwise a live project
 * URI in the same namespace. All put/import paths enforce parent/cycle checks. */
fyodor_store_result fyodor_store_move_writing(fyodor_store *store,const char *name_space,
    const fyodor_resource_ref *ref,uint64_t expected_revision,
    const fyodor_resource_ref *parent,uint64_t *revision);
/* Live world-lore resources, same trusted-admin paging contract. */
fyodor_store_result fyodor_store_list_lore(fyodor_store *store,const char *name_space,
    const char *after_uri,size_t limit,fyodor_resource_summary **items,size_t *count);
/* Replace ordered Writing lore links (0..32 unique live world-lore references
 * in the same namespace). Zero removes metadata.writing_lore. Preserves other
 * fields; CAS/history/index commit together. Links never grant access. */
fyodor_store_result fyodor_store_set_lore(fyodor_store *store,const char *name_space,
    const fyodor_resource_ref *ref,uint64_t expected_revision,
    const fyodor_resource_ref *lore,size_t count,uint64_t *revision);
typedef struct { uint64_t revision; int64_t modified_ms; int deleted; } fyodor_revision_summary;
/* Trusted administrator history, newest revision first. before=0 starts at the
 * newest; otherwise exclusive. Includes tombstones. Missing resources have an
 * empty page. Limit 1..100; caller frees with free(). Outputs unchanged on error. */
fyodor_store_result fyodor_store_history(fyodor_store *store,const char *name_space,
    const fyodor_resource_ref *ref,uint64_t before,size_t limit,fyodor_revision_summary **items,size_t *count);
void fyodor_resource_summaries_free(fyodor_resource_summary *items, size_t count);

/* Version-1 single-resource JSON package: schema, uri, title, content,
 * metadata, provenance. No namespace, ACL, credentials or database history is
 * implicitly exported. Metadata/provenance remain untrusted user data.
 * Returned UTF-8 JSON is owned by the caller; release with package_free. */
fyodor_store_result fyodor_store_export(fyodor_store *store, const char *name_space,
                                       const fyodor_resource_ref *ref, char **json, size_t *length);
/* Explicit historical package. revision=0 is the live head. Historical
 * tombstones may be exported as content packages, not as deletion commands. */
fyodor_store_result fyodor_store_export_revision(fyodor_store *store,const char *name_space,
    const fyodor_resource_ref *ref,uint64_t revision,char **json,size_t *length);
void fyodor_resource_package_free(char *json);
/* Explicit target namespace and revision policy; package cannot choose them.
 * Strict bounded UTF-8 JSON, no embedded NUL, duplicate keys or unknown fields.
 * The entire resource is validated before put; failures make no changes. Child
 * packages require their parent to have already been imported/created. */
fyodor_store_result fyodor_store_import(fyodor_store *store, const char *name_space,
                                       const char *json, size_t length, uint64_t expected_revision,
                                       fyodor_resource_ref *ref, uint64_t *revision);

#ifdef __cplusplus
}
#endif
#endif
