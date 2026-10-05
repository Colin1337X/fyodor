#include "fyodor_resource.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(test) do { if (!(test)) { \
    fprintf(stderr, "line %d: %s\n", __LINE__, #test); return 1; \
} } while (0)

static const char id_text[] = "01234567-89ab-4cde-8f01-23456789abcd";
static const char parent_text[] = "fedcba98-7654-4321-8123-456789abcdef";

static int uuid_contract(void)
{
    fyodor_uuid id, again;
    char text[FYODOR_UUID_TEXT_CAPACITY];
    CHECK(fyodor_uuid_parse(id_text, 36, &id) == FYODOR_RESOURCE_OK);
    CHECK(id.bytes[0] == 0x01 && id.bytes[6] == 0x4c && id.bytes[15] == 0xcd);
    CHECK(fyodor_uuid_format(&id, text, sizeof(text)) == FYODOR_RESOURCE_OK);
    CHECK(strcmp(text, id_text) == 0);
    for (size_t size = 0; size < sizeof(text); ++size) {
        memset(text, '!', sizeof(text));
        CHECK(fyodor_uuid_format(&id, text, size) == FYODOR_RESOURCE_CAPACITY);
        for (size_t i = 0; i < sizeof(text); ++i) CHECK(text[i] == '!');
    }
    static const char *bad[] = {
        "00000000-0000-0000-0000-000000000000",
        "01234567-89AB-4cde-8f01-23456789abcd",
        "01234567_89ab-4cde-8f01-23456789abcd",
        "0123456789ab4cde8f0123456789abcd",
        "01234567-89ab-4cde-8f01-23456789abcg",
        "{01234567-89ab-4cde-8f01-23456789abcd}"
    };
    for (size_t i = 0; i < sizeof(bad)/sizeof(bad[0]); ++i) {
        again = id;
        CHECK(fyodor_uuid_parse(bad[i], strlen(bad[i]), &again) == FYODOR_RESOURCE_INVALID);
        CHECK(memcmp(&id, &again, sizeof(id)) == 0);
    }
    CHECK(fyodor_uuid_parse(NULL, 36, &again) == FYODOR_RESOURCE_INVALID);
    CHECK(fyodor_uuid_parse(id_text, SIZE_MAX, &again) == FYODOR_RESOURCE_INVALID);
    CHECK(fyodor_uuid_parse(id_text, 36, NULL) == FYODOR_RESOURCE_INVALID);
    CHECK(fyodor_uuid_format(NULL, text, sizeof(text)) == FYODOR_RESOURCE_INVALID);
    CHECK(fyodor_uuid_format(&id, NULL, sizeof(text)) == FYODOR_RESOURCE_INVALID);
    CHECK(fyodor_uuid_generate(NULL) == FYODOR_RESOURCE_INVALID);
    /* A smoke check on the real OS entropy path, not a statistical RNG proof. */
    fyodor_uuid generated[128];
    for (size_t i = 0; i < sizeof(generated)/sizeof(generated[0]); ++i) {
        CHECK(fyodor_uuid_generate(&generated[i]) == FYODOR_RESOURCE_OK);
        CHECK((generated[i].bytes[6] & 0xf0u) == 0x40u);
        CHECK((generated[i].bytes[8] & 0xc0u) == 0x80u);
        for (size_t j = 0; j < i; ++j)
            CHECK(memcmp(&generated[i], &generated[j], sizeof(generated[i])) != 0);
        CHECK(fyodor_uuid_format(&generated[i], text, sizeof(text)) == FYODOR_RESOURCE_OK);
        CHECK(fyodor_uuid_parse(text, 36, &again) == FYODOR_RESOURCE_OK);
        CHECK(memcmp(&again, &generated[i], sizeof(again)) == 0);
    }
    return 0;
}

static int routes_contract(void)
{
    static const struct {
        fyodor_resource_kind kind;
        const char *name, *prefix, *child;
    } cases[] = {
        {FYODOR_RESOURCE_MODEL, "model", "fyodor://models/", NULL},
        {FYODOR_RESOURCE_DATASET, "dataset", "fyodor://datasets/", NULL},
        {FYODOR_RESOURCE_DOCUMENT, "document", "fyodor://writing/documents/", NULL},
        {FYODOR_RESOURCE_CHARACTER, "character", "fyodor://writing/characters/", NULL},
        {FYODOR_RESOURCE_NOTE, "note", "fyodor://writing/notes/", NULL},
        {FYODOR_RESOURCE_PROJECT, "project", "fyodor://writing/projects/", NULL},
        {FYODOR_RESOURCE_WORLD, "world", "fyodor://explore/worlds/", NULL},
        {FYODOR_RESOURCE_WORLD_LORE, "world_lore", "fyodor://explore/worlds/", "/lore/"},
        {FYODOR_RESOURCE_WORLD_SAVE, "world_save", "fyodor://explore/worlds/", "/saves/"},
        {FYODOR_RESOURCE_AGENT, "agent", "fyodor://agents/", NULL},
        {FYODOR_RESOURCE_AGENT_RUN, "agent_run", "fyodor://agents/", "/runs/"},
        {FYODOR_RESOURCE_EVALUATION, "evaluation", "fyodor://evaluations/", NULL},
        {FYODOR_RESOURCE_NODE, "node", "fyodor://nodes/", NULL},
        {FYODOR_RESOURCE_WORKFLOW, "workflow", "fyodor://workflows/", NULL}
    };
    for (size_t i = 0; i < sizeof(cases)/sizeof(cases[0]); ++i) {
        fyodor_resource_ref ref = {0}, parsed = {0};
        char expected[FYODOR_RESOURCE_URI_CAPACITY], actual[FYODOR_RESOURCE_URI_CAPACITY];
        ref.kind = cases[i].kind;
        CHECK(fyodor_uuid_parse(id_text, 36, &ref.id) == FYODOR_RESOURCE_OK);
        if (cases[i].child != NULL) {
            CHECK(fyodor_uuid_parse(parent_text, 36, &ref.parent_id) == FYODOR_RESOURCE_OK);
            snprintf(expected, sizeof(expected), "%s%s%s%s", cases[i].prefix,
                     parent_text, cases[i].child, id_text);
        } else snprintf(expected, sizeof(expected), "%s%s", cases[i].prefix, id_text);
        CHECK(strcmp(fyodor_resource_kind_name(ref.kind), cases[i].name) == 0);
        CHECK(fyodor_resource_format(&ref, actual, sizeof(actual)) == FYODOR_RESOURCE_OK);
        CHECK(strcmp(actual, expected) == 0);
        CHECK(fyodor_resource_parse(expected, strlen(expected), &parsed) == FYODOR_RESOURCE_OK);
        CHECK(fyodor_resource_equal(&ref, &parsed));
        parsed.id.bytes[0] ^= 1;
        CHECK(!fyodor_resource_equal(&ref, &parsed));
        /* Every truncated URI must fail. Exact-length non-NUL input must work. */
        size_t length = strlen(expected);
        for (size_t n = 0; n < length; ++n) {
            /* A complete parent world/agent URI is independently valid. */
            int parent = cases[i].child != NULL && n == strlen(cases[i].prefix) + 36;
            char *bounded = malloc(n == 0 ? 1 : n);
            CHECK(bounded != NULL);
            memcpy(bounded, expected, n);
            fyodor_resource_result result = fyodor_resource_parse(bounded, n, &parsed);
            free(bounded);
            CHECK(result == (parent ? FYODOR_RESOURCE_OK : FYODOR_RESOURCE_INVALID));
        }
        char *bounded = malloc(length);
        CHECK(bounded != NULL);
        memcpy(bounded, expected, length);
        CHECK(fyodor_resource_parse(bounded, length, &parsed) == FYODOR_RESOURCE_OK);
        free(bounded);
        CHECK(fyodor_resource_equal(&ref, &parsed));
        for (size_t n = 0; n <= length; ++n) {
            memset(actual, '!', sizeof(actual));
            CHECK(fyodor_resource_format(&ref, actual, n) == FYODOR_RESOURCE_CAPACITY);
            for (size_t j = 0; j < sizeof(actual); ++j) CHECK(actual[j] == '!');
        }
        CHECK(fyodor_resource_format(&ref, actual, length + 1) == FYODOR_RESOURCE_OK);
        CHECK(strcmp(actual, expected) == 0);
        /* Child identities require parents; flat identities forbid them. */
        ref.parent_id.bytes[0] ^= 1;
        if (cases[i].child != NULL) {
            CHECK(!fyodor_resource_equal(&ref, &parsed));
            memset(&ref.parent_id, 0, sizeof(ref.parent_id));
        }
        CHECK(fyodor_resource_format(&ref, actual, sizeof(actual)) == FYODOR_RESOURCE_INVALID);
    }
    return 0;
}

static int adversarial_contract(void)
{
    static const char *bad[] = {
        "", "fyodor://", "http://models/01234567-89ab-4cde-8f01-23456789abcd",
        "FYODOR://models/01234567-89ab-4cde-8f01-23456789abcd",
        "fyodor://models/01234567-89ab-4cde-8f01-23456789abcd/",
        "fyodor://models/01234567-89ab-4cde-8f01-23456789abcd?read=1",
        "fyodor://models/01234567-89ab-4cde-8f01-23456789abcd#x",
        "fyodor://models/../writing/documents/01234567-89ab-4cde-8f01-23456789abcd",
        "fyodor://models/%30%31234567-89ab-4cde-8f01-23456789abcd",
        "fyodor://user@models/01234567-89ab-4cde-8f01-23456789abcd",
        "fyodor://models:80/01234567-89ab-4cde-8f01-23456789abcd",
        "fyodor://models\\01234567-89ab-4cde-8f01-23456789abcd",
        "fyodor://models//01234567-89ab-4cde-8f01-23456789abcd",
        "fyodor://models/00000000-0000-0000-0000-000000000000",
        "fyodor://agents/01234567-89ab-4cde-8f01-23456789abcd/saves/fedcba98-7654-4321-8123-456789abcdef",
        "fyodor://explore/worlds/01234567-89ab-4cde-8f01-23456789abcd/runs/fedcba98-7654-4321-8123-456789abcdef"
    };
    fyodor_resource_ref sentinel = {0}, parsed;
    sentinel.kind = FYODOR_RESOURCE_MODEL;
    CHECK(fyodor_uuid_parse(id_text, 36, &sentinel.id) == FYODOR_RESOURCE_OK);
    for (size_t i = 0; i < sizeof(bad)/sizeof(bad[0]); ++i) {
        parsed = sentinel;
        CHECK(fyodor_resource_parse(bad[i], strlen(bad[i]), &parsed) == FYODOR_RESOURCE_INVALID);
        CHECK(fyodor_resource_equal(&parsed, &sentinel));
    }
    char valid[FYODOR_RESOURCE_URI_CAPACITY];
    CHECK(fyodor_resource_format(&sentinel, valid, sizeof(valid)) == FYODOR_RESOURCE_OK);
    size_t length = strlen(valid);
    /* NUL/control/UTF-8 bytes cannot hide suffixes or change canonical identity. */
    for (size_t i = 0; i < length; ++i) {
        char saved = valid[i];
        static const unsigned char corrupt[] = {0, 1, 9, 10, 13, 127, 128, 255};
        for (size_t j = 0; j < sizeof(corrupt); ++j) {
            valid[i] = (char)corrupt[j];
            CHECK(fyodor_resource_parse(valid, length, &parsed) == FYODOR_RESOURCE_INVALID);
        }
        valid[i] = saved;
    }
    CHECK(fyodor_resource_parse(valid, SIZE_MAX, &parsed) == FYODOR_RESOURCE_INVALID);
    CHECK(fyodor_resource_parse(NULL, 0, &parsed) == FYODOR_RESOURCE_INVALID);
    CHECK(fyodor_resource_parse(valid, length, NULL) == FYODOR_RESOURCE_INVALID);
    CHECK(!fyodor_resource_equal(NULL, &sentinel));
    CHECK(fyodor_resource_kind_name((fyodor_resource_kind)999) == NULL);
    parsed = sentinel;
    parsed.kind = (fyodor_resource_kind)999;
    CHECK(!fyodor_resource_equal(&parsed, &parsed));
    CHECK(fyodor_resource_format(&parsed, valid, sizeof(valid)) == FYODOR_RESOURCE_INVALID);
    memset(&parsed, 0, sizeof(parsed));
    CHECK(!fyodor_resource_equal(&parsed, &parsed));
    CHECK(fyodor_resource_format(NULL, valid, sizeof(valid)) == FYODOR_RESOURCE_INVALID);
    CHECK(fyodor_resource_format(&sentinel, NULL, sizeof(valid)) == FYODOR_RESOURCE_INVALID);
    return 0;
}

int main(void)
{
    if (uuid_contract() || routes_contract() || adversarial_contract()) return 1;
    puts("resource identity: UUID entropy, 14 routes, bounded parsing and rejection contracts passed");
    return 0;
}
