#ifndef NYA_VENDOR_H
#define NYA_VENDOR_H
#include <stddef.h>

/* Shared C mechanics for optional library-backed matrix providers. No vendor
   handles or platform declarations cross into public application headers. */
void *nya_vendor_open(const char *explicit_path, const char *name);
int nya_vendor_symbol(void *library, const char *name, void *target, size_t size);
void nya_vendor_close(void *library);
int nya_vendor_matrix(size_t rows, size_t columns, size_t batch, unsigned type,
    size_t *weight_bytes, size_t *input_bytes, size_t *output_bytes);
void nya_vendor_decode(float *out, const void *source, size_t elements, unsigned type);
size_t nya_vendor_budget(size_t available, const char *variable);

typedef struct nya_vendor_weight {
    const void *source;
    size_t rows, columns, bytes;
    unsigned type;
    void *value; /* provider-owned allocation or opaque array context */
    struct nya_vendor_weight *next;
} nya_vendor_weight;
typedef struct nya_vendor_cache {
    nya_vendor_weight *head;
    size_t bytes, limit;
} nya_vendor_cache;
typedef void (*nya_vendor_release)(void *context, void *value);
nya_vendor_weight *nya_vendor_find(nya_vendor_cache *cache, const void *source,
    size_t rows, size_t columns, unsigned type);
/* Eviction runs only between synchronized operations, never while a matrix is
   borrowed by a launch. scratch includes existing and prospective temporary
   allocations, so admission is bounded even when a buffer grows. */
int nya_vendor_reserve(nya_vendor_cache *cache, size_t bytes, size_t scratch,
    nya_vendor_release release, void *context);
void nya_vendor_clear(nya_vendor_cache *cache, nya_vendor_release release, void *context);
void nya_vendor_insert(nya_vendor_cache *cache, nya_vendor_weight *weight);
#endif
