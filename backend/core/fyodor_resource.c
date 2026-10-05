#include "fyodor_resource.h"

#include <string.h>

#ifdef _WIN32
#include <windows.h>
#include <bcrypt.h>
#else
#include <errno.h>
#include <fcntl.h>
#include <unistd.h>
#endif

typedef struct {
    fyodor_resource_kind kind;
    const char *name;
    const char *prefix;
    const char *child;
} resource_route;

static const resource_route routes[] = {
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

static int uuid_nonzero(const fyodor_uuid *id)
{
    uint8_t combined = 0;
    for (size_t i = 0; i < sizeof(id->bytes); ++i) combined |= id->bytes[i];
    return combined != 0;
}

static const resource_route *route_for(fyodor_resource_kind kind)
{
    for (size_t i = 0; i < sizeof(routes)/sizeof(routes[0]); ++i)
        if (routes[i].kind == kind) return &routes[i];
    return NULL;
}

static int valid_ref(const fyodor_resource_ref *ref)
{
    if (ref == NULL || !uuid_nonzero(&ref->id)) return 0;
    const resource_route *route = route_for(ref->kind);
    return route != NULL && uuid_nonzero(&ref->parent_id) == (route->child != NULL);
}

const char *fyodor_resource_kind_name(fyodor_resource_kind kind)
{
    const resource_route *route = route_for(kind);
    return route != NULL ? route->name : NULL;
}

fyodor_resource_result fyodor_uuid_generate(fyodor_uuid *out)
{
    fyodor_uuid id;
    if (out == NULL) return FYODOR_RESOURCE_INVALID;
#ifdef _WIN32
    if (BCryptGenRandom(NULL, id.bytes, (ULONG)sizeof(id.bytes),
                        BCRYPT_USE_SYSTEM_PREFERRED_RNG) != 0)
        return FYODOR_RESOURCE_ENTROPY;
#else
    int fd = open("/dev/urandom", O_RDONLY | O_CLOEXEC);
    if (fd < 0) return FYODOR_RESOURCE_ENTROPY;
    size_t offset = 0;
    while (offset < sizeof(id.bytes)) {
        ssize_t n = read(fd, id.bytes + offset, sizeof(id.bytes) - offset);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) { close(fd); return FYODOR_RESOURCE_ENTROPY; }
        offset += (size_t)n;
    }
    close(fd);
#endif
    id.bytes[6] = (uint8_t)((id.bytes[6] & 0x0fu) | 0x40u);
    id.bytes[8] = (uint8_t)((id.bytes[8] & 0x3fu) | 0x80u);
    *out = id;
    return FYODOR_RESOURCE_OK;
}

static int hex_value(char c)
{
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    return -1;
}

fyodor_resource_result fyodor_uuid_parse(const char *text, size_t length, fyodor_uuid *out)
{
    fyodor_uuid id = {{0}};
    if (text == NULL || out == NULL || length != 36) return FYODOR_RESOURCE_INVALID;
    size_t pos = 0;
    for (size_t i = 0; i < sizeof(id.bytes); ++i) {
        if (i == 4 || i == 6 || i == 8 || i == 10)
            if (text[pos++] != '-') return FYODOR_RESOURCE_INVALID;
        int high = hex_value(text[pos++]);
        int low = hex_value(text[pos++]);
        if (high < 0 || low < 0) return FYODOR_RESOURCE_INVALID;
        id.bytes[i] = (uint8_t)((unsigned)high * 16u + (unsigned)low);
    }
    if (!uuid_nonzero(&id)) return FYODOR_RESOURCE_INVALID;
    *out = id;
    return FYODOR_RESOURCE_OK;
}

fyodor_resource_result fyodor_uuid_format(const fyodor_uuid *id, char *out, size_t capacity)
{
    static const char hex[] = "0123456789abcdef";
    char text[FYODOR_UUID_TEXT_CAPACITY];
    size_t pos = 0;
    if (id == NULL || out == NULL || !uuid_nonzero(id)) return FYODOR_RESOURCE_INVALID;
    if (capacity < sizeof(text)) return FYODOR_RESOURCE_CAPACITY;
    for (size_t i = 0; i < sizeof(id->bytes); ++i) {
        if (i == 4 || i == 6 || i == 8 || i == 10) text[pos++] = '-';
        text[pos++] = hex[id->bytes[i] >> 4];
        text[pos++] = hex[id->bytes[i] & 15u];
    }
    text[pos] = '\0';
    memcpy(out, text, sizeof(text));
    return FYODOR_RESOURCE_OK;
}

fyodor_resource_result fyodor_resource_parse(const char *text, size_t length, fyodor_resource_ref *out)
{
    if (text == NULL || out == NULL || length >= FYODOR_RESOURCE_URI_CAPACITY)
        return FYODOR_RESOURCE_INVALID;
    for (size_t i = 0; i < sizeof(routes)/sizeof(routes[0]); ++i) {
        const resource_route *route = &routes[i];
        size_t prefix = strlen(route->prefix);
        size_t child = route->child != NULL ? strlen(route->child) : 0;
        size_t expected = prefix + 36 + (child != 0 ? child + 36 : 0);
        if (length != expected || memcmp(text, route->prefix, prefix) != 0) continue;
        fyodor_resource_ref ref = {0};
        ref.kind = route->kind;
        if (child != 0) {
            if (memcmp(text + prefix + 36, route->child, child) != 0 ||
                fyodor_uuid_parse(text + prefix, 36, &ref.parent_id) != FYODOR_RESOURCE_OK)
                continue;
            prefix += 36 + child;
        }
        if (fyodor_uuid_parse(text + prefix, 36, &ref.id) != FYODOR_RESOURCE_OK) continue;
        *out = ref;
        return FYODOR_RESOURCE_OK;
    }
    return FYODOR_RESOURCE_INVALID;
}

fyodor_resource_result fyodor_resource_format(const fyodor_resource_ref *ref, char *out, size_t capacity)
{
    char text[FYODOR_RESOURCE_URI_CAPACITY];
    if (out == NULL || !valid_ref(ref)) return FYODOR_RESOURCE_INVALID;
    const resource_route *route = route_for(ref->kind);
    size_t pos = strlen(route->prefix);
    size_t child = route->child != NULL ? strlen(route->child) : 0;
    size_t needed = pos + 37 + (child != 0 ? child + 36 : 0);
    if (needed > sizeof(text) || capacity < needed) return FYODOR_RESOURCE_CAPACITY;
    memcpy(text, route->prefix, pos);
    if (child != 0) {
        (void)fyodor_uuid_format(&ref->parent_id, text + pos, sizeof(text) - pos);
        pos += 36;
        memcpy(text + pos, route->child, child);
        pos += child;
    }
    (void)fyodor_uuid_format(&ref->id, text + pos, sizeof(text) - pos);
    memcpy(out, text, needed);
    return FYODOR_RESOURCE_OK;
}

int fyodor_resource_equal(const fyodor_resource_ref *a, const fyodor_resource_ref *b)
{
    return valid_ref(a) && valid_ref(b) && a->kind == b->kind &&
        memcmp(a->id.bytes, b->id.bytes, sizeof(a->id.bytes)) == 0 &&
        memcmp(a->parent_id.bytes, b->parent_id.bytes, sizeof(a->parent_id.bytes)) == 0;
}
