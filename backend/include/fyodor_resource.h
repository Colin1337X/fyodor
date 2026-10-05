#ifndef FYODOR_RESOURCE_H
#define FYODOR_RESOURCE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Application identities are independent of engine runtime model IDs.
 * Canonical URIs use lowercase UUIDs; they are neither paths nor permissions. */
#define FYODOR_UUID_TEXT_CAPACITY 37
#define FYODOR_RESOURCE_URI_CAPACITY 128

typedef enum {
    FYODOR_RESOURCE_MODEL = 1,
    FYODOR_RESOURCE_DATASET,
    FYODOR_RESOURCE_DOCUMENT,
    FYODOR_RESOURCE_CHARACTER,
    FYODOR_RESOURCE_NOTE,
    FYODOR_RESOURCE_PROJECT,
    FYODOR_RESOURCE_WORLD,
    FYODOR_RESOURCE_WORLD_LORE,
    FYODOR_RESOURCE_WORLD_SAVE,
    FYODOR_RESOURCE_AGENT,
    FYODOR_RESOURCE_AGENT_RUN,
    FYODOR_RESOURCE_EVALUATION,
    FYODOR_RESOURCE_NODE,
    FYODOR_RESOURCE_WORKFLOW
} fyodor_resource_kind;

typedef enum {
    FYODOR_RESOURCE_OK = 0,
    FYODOR_RESOURCE_INVALID = -1,
    FYODOR_RESOURCE_CAPACITY = -2,
    FYODOR_RESOURCE_ENTROPY = -3
} fyodor_resource_result;

typedef struct { uint8_t bytes[16]; } fyodor_uuid;
typedef struct {
    fyodor_resource_kind kind;
    fyodor_uuid id;
    /* Required for world lore/saves and agent runs; zero for all other kinds. */
    fyodor_uuid parent_id;
} fyodor_resource_ref;

/* OS entropy only. Failure leaves output unchanged; never falls back to rand(). */
fyodor_resource_result fyodor_uuid_generate(fyodor_uuid *out);
/* Length excludes NUL. Parsing never reads beyond length. Only canonical
 * lowercase, hyphenated, non-nil UUIDs are accepted. Outputs stay unchanged
 * on every error, including insufficient format capacity. */
fyodor_resource_result fyodor_uuid_parse(const char *text, size_t length, fyodor_uuid *out);
fyodor_resource_result fyodor_uuid_format(const fyodor_uuid *id, char *out, size_t capacity);
fyodor_resource_result fyodor_resource_parse(const char *text, size_t length, fyodor_resource_ref *out);
fyodor_resource_result fyodor_resource_format(const fyodor_resource_ref *ref, char *out, size_t capacity);
/* Rejects structurally invalid references, even when their bytes are equal. */
int fyodor_resource_equal(const fyodor_resource_ref *a, const fyodor_resource_ref *b);
/* NULL for unsupported kind. Stable labels are suitable for machine output. */
const char *fyodor_resource_kind_name(fyodor_resource_kind kind);

#ifdef __cplusplus
}
#endif
#endif
