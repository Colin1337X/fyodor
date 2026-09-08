#include "compute.h"

#include <stdlib.h>
#include <string.h>

/* One public handle wraps a typed provider pointer. CUDA and Vulkan private
   structures never alias each other when both providers are compiled. */
typedef struct nya_vulkan_context nya_vulkan_context;
typedef struct nya_cuda_context nya_cuda_context;
struct nya_compute_context {
    int provider;
    union { nya_vulkan_context *vulkan; nya_cuda_context *cuda; } state;
};
#ifdef NYA_ENABLE_VULKAN
nya_vulkan_context *nya_vulkan_create(void);
void nya_vulkan_free(nya_vulkan_context *context);
int nya_vulkan_matvec(nya_vulkan_context *context, const void *weights,
    size_t rows, size_t columns, const float *input, float *output);
int nya_vulkan_active(const nya_vulkan_context *context);
#endif
#ifdef NYA_ENABLE_CUDA
nya_cuda_context *nya_cuda_create(void);
void nya_cuda_free(nya_cuda_context *context);
int nya_cuda_matvec(nya_cuda_context *context, const void *weights,
    size_t rows, size_t columns, const float *input, float *output);
int nya_cuda_active(const nya_cuda_context *context);
int nya_cuda_matvec_typed(nya_cuda_context *context, const void *weights,
    size_t rows, size_t columns, unsigned int type, const float *input, float *output);
#endif

nya_compute_context *nya_compute_create(void)
{
    return nya_compute_create_for(getenv("NYA_COMPUTE"));
}

nya_compute_context *nya_compute_create_for(const char *selection)
{
#if defined(NYA_ENABLE_VULKAN) || defined(NYA_ENABLE_CUDA)
    if (selection == NULL) return NULL;
    nya_compute_context *context = (nya_compute_context *)calloc(1, sizeof(*context));
    if (context == NULL) return NULL;
#ifdef NYA_ENABLE_VULKAN
    if (strcmp(selection, "vulkan") == 0) {
        context->state.vulkan = nya_vulkan_create();
        if (context->state.vulkan != NULL) { context->provider = 1; return context; }
    }
#endif
#ifdef NYA_ENABLE_CUDA
    if (strcmp(selection, "cuda") == 0) {
        context->state.cuda = nya_cuda_create();
        if (context->state.cuda != NULL) { context->provider = 2; return context; }
    }
#endif
    free(context);
#endif
    (void)selection;
    return NULL;
}

void nya_compute_free(nya_compute_context *context)
{
    if (context == NULL) return;
#ifdef NYA_ENABLE_VULKAN
    if (context->provider == 1) nya_vulkan_free(context->state.vulkan);
#endif
#ifdef NYA_ENABLE_CUDA
    if (context->provider == 2) nya_cuda_free(context->state.cuda);
#endif
    free(context);
}

int nya_compute_matvec(nya_compute_context *context, const void *weights,
    size_t rows, size_t columns, const float *input, float *output)
{
#ifdef NYA_ENABLE_VULKAN
    if (context != NULL && context->provider == 1)
        return nya_vulkan_matvec(context->state.vulkan, weights, rows, columns, input, output);
#endif
#ifdef NYA_ENABLE_CUDA
    if (context != NULL && context->provider == 2)
        return nya_cuda_matvec(context->state.cuda, weights, rows, columns, input, output);
#endif
    (void)context; (void)weights; (void)rows; (void)columns; (void)input; (void)output;
    return -1;
}

const char *nya_compute_name(const nya_compute_context *context)
{
#ifdef NYA_ENABLE_VULKAN
    if (context != NULL && context->provider == 1 && nya_vulkan_active(context->state.vulkan)) return "vulkan";
#endif
#ifdef NYA_ENABLE_CUDA
    if (context != NULL && context->provider == 2 && nya_cuda_active(context->state.cuda)) return "cuda";
#endif
    (void)context;
    return "cpu";
}

int nya_compute_matvec_typed(nya_compute_context *context, const void *weights,
    size_t rows, size_t columns, unsigned int type, const float *input, float *output)
{
#ifdef NYA_ENABLE_CUDA
    if (context != NULL && context->provider == 2)
        return nya_cuda_matvec_typed(context->state.cuda, weights, rows, columns, type, input, output);
#endif
    if (type == 0) return nya_compute_matvec(context, weights, rows, columns, input, output);
    return -1;
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
