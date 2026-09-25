#include "compute_backend.h"
#include <stdlib.h>
#include <string.h>

struct nya_compute_context { const nya_backend_interface *api; void *state; };
struct nya_compute_plan { const nya_backend_interface *api; void *state; };
#ifdef NYA_ENABLE_CUDA
const nya_backend_interface *nya_cuda_backend(void);
#endif
#ifdef NYA_ENABLE_VULKAN
const nya_backend_interface *nya_vulkan_backend(void);
#endif
#ifdef NYA_ENABLE_ROCM
const nya_backend_interface *nya_rocm_backend(void);
#endif
#ifdef NYA_ENABLE_MLX
const nya_backend_interface *nya_mlx_backend(void);
#endif
/* Registration is the only platform-conditional list. Every operation uses
   the selected vtable, so adding a backend never changes call-site dispatch. */
typedef const nya_backend_interface *(*nya_backend_factory)(void);
const nya_backend_interface *nya_cpu_backend(void);
static const nya_backend_factory factories[] = {
    nya_cpu_backend,
#ifdef NYA_ENABLE_CUDA
    nya_cuda_backend,
#endif
#ifdef NYA_ENABLE_VULKAN
    nya_vulkan_backend,
#endif
#ifdef NYA_ENABLE_ROCM
    nya_rocm_backend,
#endif
#ifdef NYA_ENABLE_MLX
    nya_mlx_backend,
#endif
    NULL
};
nya_compute_context *nya_compute_create(void) { return nya_compute_create_for(getenv("NYA_COMPUTE")); }
int nya_compute_backend_known(const char *provider)
{
    static const char *names[]={"cpu","cuda","vulkan","rocm","mlx"};
    if (provider) for (size_t i=0; i<sizeof(names)/sizeof(names[0]); ++i)
        if (!strcmp(provider,names[i])) return 1;
    return 0;
}
int nya_compute_backend_compiled(const char *provider)
{
    if (provider) for (size_t i=0; factories[i]; ++i)
        if (!strcmp(provider,factories[i]()->name)) return 1;
    return 0;
}
void nya_compute_context_stats(const nya_compute_context *c, nya_compute_stats *stats)
{
    if (!stats) return;
    memset(stats,0,sizeof(*stats));
    if (c && c->api->context_stats) c->api->context_stats(c->state,stats);
}
nya_compute_context *nya_compute_create_for(const char *selection)
{
    if (selection == NULL) selection = "cpu";
    for (size_t i = 0; factories[i] != NULL; ++i) {
        const nya_backend_interface *api = factories[i]();
        if (strcmp(api->name, selection)) continue;
        nya_compute_context *c = (nya_compute_context *)calloc(1, sizeof(*c));
        if (c == NULL) return NULL;
        c->api = api; c->state = api->create();
        if (c->state != NULL) return c;
        free(c); return NULL;
    }
    return NULL;
}
void nya_compute_free(nya_compute_context *c)
{
    if (c != NULL) { c->api->destroy(c->state); free(c); }
}
const char *nya_compute_name(const nya_compute_context *c)
{
    return c != NULL && c->api->active(c->state) ? c->api->name : "cpu";
}
unsigned nya_compute_capabilities(const nya_compute_context *c)
{
    return c != NULL && c->api->active(c->state) ? c->api->capabilities : 0;
}
int nya_compute_matvec_typed(nya_compute_context *c, const void *w, size_t rows,
    size_t cols, unsigned type, const float *x, float *y)
{
    return c != NULL && c->api->active(c->state) ? c->api->matvec(c->state, w, rows, cols, type, x, y) : -1;
}
int nya_compute_matvec(nya_compute_context *c, const void *w, size_t rows, size_t cols, const float *x, float *y)
{
    return nya_compute_matvec_typed(c, w, rows, cols, 0, x, y);
}
int nya_compute_matmul_typed(nya_compute_context *c, const void *w, size_t rows, size_t cols,
    unsigned type, const float *x, float *y, size_t batch)
{
    return c != NULL && c->api->active(c->state) && c->api->matmul != NULL ?
        c->api->matmul(c->state, w, rows, cols, type, x, y, batch) : -1;
}
int nya_compute_attention_f32(nya_compute_context *c, const nya_compute_attention *a)
{
    return c && c->api->active(c->state) && c->api->attention_f32 ? c->api->attention_f32(c->state,a) : -1;
}
nya_compute_plan *nya_compute_plan_create(nya_compute_context *c, const struct nya_llm_context *model, size_t capacity)
{
    if (c == NULL || !c->api->active(c->state) || c->api->plan_create == NULL) return NULL;
    nya_compute_plan *p = (nya_compute_plan *)calloc(1, sizeof(*p));
    if (p == NULL) return NULL;
    p->api = c->api; p->state = c->api->plan_create(c->state, model, capacity);
    if (p->state != NULL) return p;
    free(p); return NULL;
}
void nya_compute_plan_free(nya_compute_plan *p)
{
    if (p != NULL) { p->api->plan_free(p->state); free(p); }
}
int nya_compute_plan_token(nya_compute_plan *p, unsigned int token, size_t position, int logits, float *output)
{
    return p != NULL ? p->api->plan_token(p->state, token, position, logits, output) : -1;
}
int nya_compute_plan_prefill(nya_compute_plan *p, const unsigned int *tokens, size_t count, size_t position, float *output)
{
    return p != NULL && p->api->plan_prefill != NULL ? p->api->plan_prefill(p->state, tokens, count, position, output) : -1;
}
void nya_compute_plan_stats(const nya_compute_plan *p, nya_compute_stats *stats)
{
    if (stats == NULL) return;
    memset(stats, 0, sizeof(*stats));
    if (p != NULL && p->api->plan_stats != NULL) p->api->plan_stats(p->state, stats);
}
/* Provider-owned indexing and validity checks prevent exposing raw pointers. */
int nya_compute_plan_read_kv(nya_compute_plan *p, size_t layer, size_t position,
    size_t count, float *keys, float *values, size_t elements)
{
    return p && p->api->plan_read_kv ? p->api->plan_read_kv(p->state,layer,position,count,keys,values,elements) : -1;
}
int nya_compute_vulkan_compiled(void)
{
#ifdef NYA_ENABLE_VULKAN
    return 1;
#else
    return 0;
#endif
}
int nya_compute_cuda_compiled(void)
{
#ifdef NYA_ENABLE_CUDA
    return 1;
#else
    return 0;
#endif
}
