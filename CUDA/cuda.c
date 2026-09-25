#include "compute.h"
#include "cuda_source.h"

#include <cuda.h>
#include <nvrtc.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <wchar.h>
#include <limits.h>
#include "blas.h"
#ifdef NYA_ENABLE_CUTLASS
#include "cutlass/host.h"
#endif

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#define NYA_CUDA_WEIGHT_LIMIT 2048

typedef struct nya_cuda_weight {
    const void *source;
    size_t rows, columns, bytes;
    unsigned int type;
    CUdeviceptr memory;
} nya_cuda_weight;

/* API function pointers, headers, modules and device allocations are all owned
   by this removable directory. The rest of the backend only sees C handles. */
typedef struct nya_cuda_context {
    void *driver_library, *compiler_library, *builtins_library;
    CUdevice device;
    CUcontext context;
    CUmodule module;
    CUfunction function;
    CUfunction dot[31];
    CUfunction gemm[31];
    CUfunction gemm_large[31];
    CUfunction expand[31];
    CUfunction expand_transposed[31];
    nya_cuda_blas blas;
#ifdef NYA_ENABLE_CUTLASS
    nya_cuda_cutlass cutlass;
#endif
    CUdeviceptr blas_storage;
    size_t blas_bytes;
    unsigned gemm_tile;
    int reference;
    CUstream stream;
    CUdeviceptr input, output;
    size_t input_bytes, output_bytes, cached_bytes, cache_limit, weight_count;
    int grid_limit, failed;
    nya_cuda_weight weights[NYA_CUDA_WEIGHT_LIMIT];
    CUresult (CUDAAPI *init)(unsigned int);
    CUresult (CUDAAPI *device_get)(CUdevice *, int);
    CUresult (CUDAAPI *attribute)(int *, CUdevice_attribute, CUdevice);
    CUresult (CUDAAPI *retain)(CUcontext *, CUdevice);
    CUresult (CUDAAPI *release)(CUdevice);
    CUresult (CUDAAPI *push)(CUcontext);
    CUresult (CUDAAPI *pop)(CUcontext *);
    CUresult (CUDAAPI *memory_info)(size_t *, size_t *);
    CUresult (CUDAAPI *allocate)(CUdeviceptr *, size_t);
    CUresult (CUDAAPI *deallocate)(CUdeviceptr);
    CUresult (CUDAAPI *upload)(CUdeviceptr, const void *, size_t);
    CUresult (CUDAAPI *download)(void *, CUdeviceptr, size_t);
    CUresult (CUDAAPI *module_load)(CUmodule *, const void *);
    CUresult (CUDAAPI *module_unload)(CUmodule);
    CUresult (CUDAAPI *function_get)(CUfunction *, CUmodule, const char *);
    CUresult (CUDAAPI *stream_create)(CUstream *, unsigned int);
    CUresult (CUDAAPI *stream_destroy)(CUstream);
    CUresult (CUDAAPI *stream_sync)(CUstream);
    CUresult (CUDAAPI *launch)(CUfunction, unsigned int, unsigned int, unsigned int,
        unsigned int, unsigned int, unsigned int, unsigned int, CUstream, void **, void **);
    CUresult (CUDAAPI *capture_begin)(CUstream, CUstreamCaptureMode);
    CUresult (CUDAAPI *capture_end)(CUstream, CUgraph *);
    CUresult (CUDAAPI *graph_instantiate)(CUgraphExec *, CUgraph, unsigned long long);
    CUresult (CUDAAPI *graph_launch)(CUgraphExec, CUstream);
    CUresult (CUDAAPI *graph_free)(CUgraph);
    CUresult (CUDAAPI *graph_exec_free)(CUgraphExec);
    CUresult (CUDAAPI *event_create)(CUevent *, unsigned);
    CUresult (CUDAAPI *event_record)(CUevent, CUstream);
    CUresult (CUDAAPI *event_elapsed)(float *, CUevent, CUevent);
    CUresult (CUDAAPI *event_free)(CUevent);
    nvrtcResult (*program_create)(nvrtcProgram *, const char *, const char *, int, const char *const *, const char *const *);
    nvrtcResult (*program_compile)(nvrtcProgram, int, const char *const *);
    nvrtcResult (*program_destroy)(nvrtcProgram *);
    nvrtcResult (*ptx_size)(nvrtcProgram, size_t *);
    nvrtcResult (*ptx_get)(nvrtcProgram, char *);
    nvrtcResult (*log_size)(nvrtcProgram, size_t *);
    nvrtcResult (*log_get)(nvrtcProgram, char *);
    nvrtcResult (*arch_count)(int *);
    nvrtcResult (*arch_get)(int *);
    nvrtcResult (*version)(int *, int *);
} nya_cuda_context;

static void *nya_cuda_open(const char *path, int driver)
{
#ifdef _WIN32
    if (driver) return (void *)LoadLibraryExW(L"nvcuda.dll", NULL, LOAD_LIBRARY_SEARCH_SYSTEM32);
    int n = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, path, -1, NULL, 0);
    if (n <= 0 || (size_t)n > SIZE_MAX / sizeof(wchar_t)) return NULL;
    wchar_t *wide = (wchar_t *)malloc((size_t)n * sizeof(wchar_t));
    if (wide == NULL) return NULL;
    void *library = NULL;
    if (MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, path, -1, wide, n) == n)
        library = (void *)LoadLibraryExW(wide, NULL, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
    free(wide);
    return library;
#else
    return dlopen(driver ? "libcuda.so.1" : path, RTLD_NOW | RTLD_LOCAL);
#endif
}

static void nya_cuda_close(void *library)
{
    if (library == NULL) return;
#ifdef _WIN32
    FreeLibrary((HMODULE)library);
#else
    dlclose(library);
#endif
}

static int nya_cuda_symbol(void *library, const char *name, void *destination, size_t size)
{
#ifdef _WIN32
    FARPROC address = GetProcAddress((HMODULE)library, name);
#else
    void *address = dlsym(library, name);
#endif
    /* These OS APIs define conversion to callable symbols. memcpy avoids
       ISO-C object/function pointer casts and verifies the platform ABI size. */
    if (address == NULL || sizeof(address) != size) return -1;
    memcpy(destination, &address, size);
    return 0;
}

#ifdef NYA_ENABLE_CUTLASS
#include "cutlass/loader.inc"
#endif
#include "blas.inc"

static void *nya_cuda_compiler(void)
{
    const char *explicit_path = getenv("NYA_CUDA_NVRTC");
    if (explicit_path != NULL && explicit_path[0] != '\0') return nya_cuda_open(explicit_path, 0);
    /* A packaged executable may move away from the build machine. A missing
       configured default must not prevent discovery in the installed toolkit. */
    if (nya_nvrtc_default[0] != '\0') {
        void *configured = nya_cuda_open(nya_nvrtc_default, 0);
        if (configured != NULL) return configured;
    }
#ifdef _WIN32
    const char *toolkit = getenv("CUDA_PATH");
    if (toolkit != NULL) {
        const char *names[] = {"bin/x64/nvrtc64_130_0.dll", "bin/nvrtc64_130_0.dll", "bin/nvrtc64_120_0.dll"};
        for (size_t i = 0; i < sizeof(names) / sizeof(names[0]); ++i) {
            char path[32768];
            int n = snprintf(path, sizeof(path), "%s/%s", toolkit, names[i]);
            if (n > 0 && (size_t)n < sizeof(path)) {
                void *library = nya_cuda_open(path, 0);
                if (library != NULL) return library;
            }
        }
    }
    return NULL;
#else
    void *library = nya_cuda_open("libnvrtc.so", 0);
    if (library == NULL) library = nya_cuda_open("libnvrtc.so.13", 0);
    if (library == NULL) library = nya_cuda_open("libnvrtc.so.12", 0);
    return library;
#endif
}

void nya_cuda_free(nya_cuda_context *c)
{
    if (c == NULL) return;
    if (c->context != NULL) {
        /* Restore the caller's current context after cleanup. A retained
           primary context is released, never reset or destroyed globally. */
        if (c->push(c->context) == CUDA_SUCCESS) {
            CUcontext previous;
            if (c->stream != NULL) c->stream_sync(c->stream);
            nya_blas_close(c);
#ifdef NYA_ENABLE_CUTLASS
            nya_cutlass_close(c);
#endif
            if (c->blas_storage) c->deallocate(c->blas_storage);
            for (size_t i = 0; i < c->weight_count; ++i) c->deallocate(c->weights[i].memory);
            if (c->input != 0) c->deallocate(c->input);
            if (c->output != 0) c->deallocate(c->output);
            if (c->module != NULL) c->module_unload(c->module);
            if (c->stream != NULL) c->stream_destroy(c->stream);
            c->pop(&previous);
        }
        c->release(c->device);
    }
    nya_cuda_close(c->compiler_library); nya_cuda_close(c->builtins_library);
    nya_cuda_close(c->driver_library); free(c);
}

nya_cuda_context *nya_cuda_create(void)
{
    nya_cuda_context *c = (nya_cuda_context *)calloc(1, sizeof(*c));
    nvrtcProgram program = NULL;
    char *ptx = NULL;
    int pushed = 0, major, minor, count, architectures[256], selected = 0;
    CUcontext previous;
    size_t free_bytes, total_bytes, ptx_bytes;
    if (c == NULL) return NULL;
    c->driver_library = nya_cuda_open(NULL, 1);
    c->compiler_library = nya_cuda_compiler();
    if (c->driver_library == NULL || c->compiler_library == NULL) goto failure;
#define DRIVER(field, symbol) do { if (nya_cuda_symbol(c->driver_library, symbol, &c->field, sizeof(c->field)) != 0) goto failure; } while (0)
#define COMPILER(field, symbol) do { if (nya_cuda_symbol(c->compiler_library, symbol, &c->field, sizeof(c->field)) != 0) goto failure; } while (0)
    DRIVER(init, "cuInit"); DRIVER(device_get, "cuDeviceGet"); DRIVER(attribute, "cuDeviceGetAttribute");
    DRIVER(retain, "cuDevicePrimaryCtxRetain"); DRIVER(release, "cuDevicePrimaryCtxRelease_v2");
    DRIVER(push, "cuCtxPushCurrent_v2"); DRIVER(pop, "cuCtxPopCurrent_v2");
    DRIVER(memory_info, "cuMemGetInfo_v2"); DRIVER(allocate, "cuMemAlloc_v2"); DRIVER(deallocate, "cuMemFree_v2");
    DRIVER(upload, "cuMemcpyHtoD_v2"); DRIVER(download, "cuMemcpyDtoH_v2");
    DRIVER(module_load, "cuModuleLoadData"); DRIVER(module_unload, "cuModuleUnload"); DRIVER(function_get, "cuModuleGetFunction");
    DRIVER(stream_create, "cuStreamCreate"); DRIVER(stream_destroy, "cuStreamDestroy_v2"); DRIVER(stream_sync, "cuStreamSynchronize");
    DRIVER(launch, "cuLaunchKernel");
    DRIVER(capture_begin, "cuStreamBeginCapture_v2"); DRIVER(capture_end, "cuStreamEndCapture");
    DRIVER(graph_instantiate, "cuGraphInstantiateWithFlags");
    DRIVER(graph_launch, "cuGraphLaunch"); DRIVER(graph_free, "cuGraphDestroy"); DRIVER(graph_exec_free, "cuGraphExecDestroy");
    DRIVER(event_create, "cuEventCreate"); DRIVER(event_record, "cuEventRecord");
    DRIVER(event_elapsed, "cuEventElapsedTime"); DRIVER(event_free, "cuEventDestroy_v2");
    COMPILER(program_create, "nvrtcCreateProgram"); COMPILER(program_compile, "nvrtcCompileProgram"); COMPILER(program_destroy, "nvrtcDestroyProgram");
    COMPILER(ptx_size, "nvrtcGetPTXSize"); COMPILER(ptx_get, "nvrtcGetPTX");
    COMPILER(log_size, "nvrtcGetProgramLogSize"); COMPILER(log_get, "nvrtcGetProgramLog");
    COMPILER(arch_count, "nvrtcGetNumSupportedArchs"); COMPILER(arch_get, "nvrtcGetSupportedArchs");
    COMPILER(version, "nvrtcVersion");
#undef DRIVER
#undef COMPILER
#ifdef _WIN32
    /* NVRTC loads its builtins lazily during compilation. The temporary DLL
       search scope of LoadLibraryEx has ended by then. Preload the exact sibling
       DLL by absolute path without changing the process-wide DLL search path. */
    if (c->version(&major, &minor) != NVRTC_SUCCESS || major < 1 || major > 100 || minor < 0 || minor > 99) goto failure;
    wchar_t compiler_path[32768];
    DWORD path_length = GetModuleFileNameW((HMODULE)c->compiler_library, compiler_path, 32768);
    if (path_length == 0 || path_length >= 32768) goto failure;
    wchar_t *separator = wcsrchr(compiler_path, L'\\');
    if (separator == NULL) goto failure;
    size_t prefix_length = (size_t)(separator + 1 - compiler_path);
    if (swprintf(compiler_path + prefix_length, 32768 - prefix_length, L"nvrtc-builtins64_%d%d.dll", major, minor) < 0) goto failure;
    c->builtins_library = (void *)LoadLibraryExW(compiler_path, NULL, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
    if (c->builtins_library == NULL) goto failure;
#endif
    if (c->init(0) != CUDA_SUCCESS || c->device_get(&c->device, 0) != CUDA_SUCCESS ||
        c->attribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, c->device) != CUDA_SUCCESS ||
        c->attribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, c->device) != CUDA_SUCCESS ||
        major < 1 || major > 100 || minor < 0 || minor > 9 ||
        c->attribute(&c->grid_limit, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X, c->device) != CUDA_SUCCESS || c->grid_limit <= 0 ||
        c->arch_count(&count) != NVRTC_SUCCESS || count <= 0 || count > 256 || c->arch_get(architectures) != NVRTC_SUCCESS) goto failure;
    for (int i = 0; i < count; ++i) if (architectures[i] <= major * 10 + minor && architectures[i] > selected) selected = architectures[i];
    if (selected == 0) goto failure;
    char architecture[64];
    snprintf(architecture, sizeof(architecture), "--gpu-architecture=compute_%d", selected);
    const char *reference = getenv("NYA_CUDA_REFERENCE");
    c->reference = reference != NULL && strcmp(reference, "1") == 0;
    const char *options[] = {architecture, "--std=c++11", c->reference ? "--fmad=false" : "--fmad=true", c->reference ? "-DNYA_CUDA_REFERENCE_MATH=1" : "-DNYA_CUDA_FAST_MATH=1"};
    if (c->program_create(&program, (const char *)nya_cuda_source, "fyodor_matvec.cu", 0, NULL, NULL) != NVRTC_SUCCESS) goto failure;
    if (c->program_compile(program, 4, options) != NVRTC_SUCCESS) {
        if (getenv("NYA_CUDA_DEBUG") != NULL) {
            size_t log_bytes;
            if (c->log_size(program, &log_bytes) == NVRTC_SUCCESS && log_bytes > 0 && log_bytes < 65536) {
                char *log = (char *)calloc(log_bytes + 1, 1);
                if (log != NULL) { if (c->log_get(program, log) == NVRTC_SUCCESS) fprintf(stderr, "CUDA compile: %s\n", log); free(log); }
            }
        }
        goto failure;
    }
    if (c->ptx_size(program, &ptx_bytes) != NVRTC_SUCCESS || ptx_bytes == 0 || ptx_bytes > 16U * 1024U * 1024U) goto failure;
    ptx = (char *)malloc(ptx_bytes);
    if (ptx == NULL || c->ptx_get(program, ptx) != NVRTC_SUCCESS) goto failure;
    c->program_destroy(&program); program = NULL;
    if (c->retain(&c->context, c->device) != CUDA_SUCCESS) goto failure;
    if (c->push(c->context) != CUDA_SUCCESS) goto failure;
    pushed = 1;
    if (c->module_load(&c->module, ptx) != CUDA_SUCCESS || c->function_get(&c->function, c->module, "nya_matvec") != CUDA_SUCCESS ||
        c->stream_create(&c->stream, CU_STREAM_NON_BLOCKING) != CUDA_SUCCESS || c->memory_info(&free_bytes, &total_bytes) != CUDA_SUCCESS) goto failure;
    const unsigned types[] = {0, 1, 2, 8, 12, 14, 30};
    for (size_t i = 0; i < sizeof(types)/sizeof(types[0]); ++i) {
        char name[32]; snprintf(name, sizeof(name), "nya_dot_%u", types[i]);
        if (c->function_get(&c->dot[types[i]], c->module, name) != CUDA_SUCCESS) goto failure;
        const char *tile_mode = getenv("NYA_CUDA_GEMM_TILE");
        c->gemm_tile = c->reference || (tile_mode && !strcmp(tile_mode,"32")) ? 32 : 64;
        snprintf(name, sizeof(name), "nya_gemm_%u", types[i]);
        if (c->function_get(&c->gemm[types[i]], c->module, name) != CUDA_SUCCESS) goto failure;
        snprintf(name, sizeof(name), "nya_gemm64_%u", types[i]);
        if (c->function_get(&c->gemm_large[types[i]], c->module, name) != CUDA_SUCCESS) goto failure;
        snprintf(name, sizeof(name), "nya_expand_%u", types[i]);
        if (c->function_get(&c->expand[types[i]], c->module, name) != CUDA_SUCCESS) goto failure;
        snprintf(name, sizeof(name), "nya_expand_t_%u", types[i]);
        if (c->function_get(&c->expand_transposed[types[i]], c->module, name) != CUDA_SUCCESS) goto failure;
    }
    nya_blas_open(c);
#ifdef NYA_ENABLE_CUTLASS
    nya_cutlass_open(c);
#endif
    /* Leave at least a quarter of currently free device memory for other work.
       This is a per-context upper bound, not a reservation or eviction policy. */
    size_t reserve = free_bytes / 10;
    if (reserve < 256U * 1024U * 1024U) reserve = 256U * 1024U * 1024U;
    c->cache_limit = free_bytes > reserve ? free_bytes - reserve : 0;
    /* Optional upper bound for reproducible admission/OOM testing and sharing
       a GPU. The resident planner also reserves its complete KV/scratch budget. */
    const char *budget = getenv("NYA_CUDA_MEMORY_MIB");
    if (budget != NULL) {
        char *end;
        unsigned long long mib = strtoull(budget, &end, 10);
        if (budget[0] < '0' || budget[0] > '9' || *end || mib > SIZE_MAX / (1024U * 1024U)) goto failure;
        size_t limit = (size_t)mib * 1024U * 1024U;
        if (limit < c->cache_limit) c->cache_limit = limit;
    }
    if (c->pop(&previous) != CUDA_SUCCESS) { pushed = 0; goto failure; }
    free(ptx);
    return c;
failure:
    if (pushed) c->pop(&previous);
    if (program != NULL && c->program_destroy != NULL) c->program_destroy(&program);
    if (getenv("NYA_CUDA_DEBUG") != NULL) fprintf(stderr, "CUDA provider unavailable; CPU fallback selected\n");
    free(ptx); nya_cuda_free(c);
    return NULL;
}

int nya_cuda_active(const nya_cuda_context *c) { return c != NULL && !c->failed; }

/* Grow vector buffers only when necessary; callers serialize this context.
   Free the previous buffer after successful allocation, preserving cleanup
   state if the driver reports out-of-memory. */
static int nya_cuda_grow(nya_cuda_context *c, CUdeviceptr *memory, size_t *capacity, size_t bytes)
{
    if (*capacity >= bytes) return 0;
    CUdeviceptr replacement = 0;
    if (c->allocate(&replacement, bytes) != CUDA_SUCCESS) return -1;
    if (*memory != 0 && c->deallocate(*memory) != CUDA_SUCCESS) { c->deallocate(replacement); return -1; }
    *memory = replacement; *capacity = bytes;
    return 0;
}

/* Standalone host-buffer dispatch is also useful for numerical kernel tests.
   Normal dense inference uses resident plans and never passes through here. */
static int nya_cuda_matmul_typed(nya_cuda_context *c, const void *weights, size_t rows, size_t columns,
    unsigned int type, const float *input, float *output, size_t batch)
{
    size_t block = 1, block_bytes = 4, row_bytes, bytes, index;
    CUcontext previous;
    int result = -1;
    if (!nya_cuda_active(c) || weights == NULL || input == NULL || output == NULL || rows == 0 || columns == 0 ||
        batch == 0 || batch > 512 || rows > (size_t)c->grid_limit ||
        columns > SIZE_MAX / sizeof(float) / batch || rows > SIZE_MAX / sizeof(float) / batch) return -1;
    switch (type) {
        case 0: break;
        case 1: case 30: block_bytes = 2; break;
        case 2: block = 32; block_bytes = 18; break;
        case 8: block = 32; block_bytes = 34; break;
        case 12: block = 256; block_bytes = 144; break;
        case 14: block = 256; block_bytes = 210; break;
        default: return -1;
    }
    if (columns % block != 0 || columns / block > SIZE_MAX / block_bytes) return -1;
    row_bytes = columns / block * block_bytes;
    if (rows > SIZE_MAX / row_bytes) return -1;
    bytes = rows * row_bytes;
    for (index = 0; index < c->weight_count; ++index) {
        const nya_cuda_weight *w = &c->weights[index];
        if (w->source == weights && w->rows == rows && w->columns == columns && w->type == type) break;
    }
    if (index == c->weight_count && (index == NYA_CUDA_WEIGHT_LIMIT || bytes > c->cache_limit - c->cached_bytes)) return -1;
    if (c->push(c->context) != CUDA_SUCCESS) { c->failed = 1; return -1; }
    if (index == c->weight_count) {
        nya_cuda_weight *w = &c->weights[index];
        if (c->allocate(&w->memory, bytes) != CUDA_SUCCESS) goto done;
        /* Register ownership before any fallible copy so teardown frees it. */
        w->source = weights; w->rows = rows; w->columns = columns; w->type = type; w->bytes = bytes;
        ++c->weight_count; c->cached_bytes += bytes;
        if (c->upload(w->memory, weights, bytes) != CUDA_SUCCESS) goto done;
    }
    if (nya_cuda_grow(c, &c->input, &c->input_bytes, columns * batch * sizeof(float)) != 0 ||
        nya_cuda_grow(c, &c->output, &c->output_bytes, rows * batch * sizeof(float)) != 0 ||
        c->upload(c->input, input, columns * batch * sizeof(float)) != CUDA_SUCCESS) goto done;
    /* Pageable HtoD may return after staging, before DMA completes. Kernels
       use a nonblocking stream, which does not implicitly wait on the legacy
       copy stream. Finish that stream before consuming the uploaded buffers.
       Resident inference performs this dependency once at plan creation. */
    if (c->stream_sync(NULL) != CUDA_SUCCESS) goto done;
    unsigned long long cols = (unsigned long long)columns, stride = (unsigned long long)row_bytes;
    CUdeviceptr matrix = c->weights[index].memory;
    void *arguments[] = {&matrix, &c->input, &c->output, &cols, &stride, &type};
    unsigned long long row_count = rows;
    CUfunction function = c->reference ? c->function : c->dot[type];
    unsigned blocks = (unsigned)(c->reference ? rows : (rows + 7) / 8);
    if (!c->reference) arguments[5] = &row_count;
    if (batch > 1) {
        size_t expansion = nya_blas_bytes(c, rows, columns, batch);
        if (expansion && expansion + NYA_BLAS_WORKSPACE <= c->cache_limit - c->cached_bytes &&
            nya_cuda_grow(c, &c->blas_storage, &c->blas_bytes, expansion + NYA_BLAS_WORKSPACE) == 0) {
            if (nya_blas_multiply(c, matrix, c->input, c->output, rows, columns, type, row_bytes,
                batch, c->blas_storage + NYA_BLAS_WORKSPACE, expansion, c->blas_storage, NULL)) goto done;
        } else {
        unsigned r = (unsigned)rows, b = (unsigned)batch;
        void *matrix_arguments[] = {&matrix, &c->input, &c->output, &cols, &stride, &r, &b};
        unsigned tile = b >= 32 && r >= 32 ? c->gemm_tile : 32, bx = tile == 64 ? 16 : 32;
        CUfunction kernel = tile == 64 ? c->gemm_large[type] : c->gemm[type];
        if (c->launch(kernel, (r+tile-1)/tile, (b+tile-1)/tile, 1, bx, 256/bx, 1, 0, c->stream, matrix_arguments, NULL) != CUDA_SUCCESS) goto done;
        }
    } else if (c->launch(function, blocks, 1, 1, 256, 1, 1, 0, c->stream, arguments, NULL) != CUDA_SUCCESS) goto done;
    if (c->stream_sync(c->stream) != CUDA_SUCCESS ||
        c->download(output, c->output, rows * batch * sizeof(float)) != CUDA_SUCCESS) goto done;
    result = 0;
done:
    if (c->pop(&previous) != CUDA_SUCCESS) result = -1;
    /* Driver failures poison dispatch, but all allocations remain owned until
       unload. The caller can safely recompute the complete output on CPU. */
    if (result != 0) c->failed = 1;
    return result;
}

int nya_cuda_matvec_typed(nya_cuda_context *c, const void *weights, size_t rows, size_t columns,
    unsigned int type, const float *input, float *output)
{ return nya_cuda_matmul_typed(c, weights, rows, columns, type, input, output, 1); }

int nya_cuda_matvec(nya_cuda_context *c, const void *weights, size_t rows, size_t columns,
    const float *input, float *output)
{
    return nya_cuda_matvec_typed(c, weights, rows, columns, 0, input, output);
}

#include "compute_backend.h"
#include "resident.inc"
static void *cuda_backend_create(void) { return nya_cuda_create(); }
static void cuda_backend_free(void *p) { nya_cuda_free(p); }
static int cuda_backend_active(const void *p) { return nya_cuda_active(p); }
static int cuda_backend_matvec(void *p, const void *w, size_t r, size_t c,
    unsigned t, const float *x, float *y)
{ return nya_cuda_matvec_typed(p, w, r, c, t, x, y); }
static int cuda_backend_matmul(void *p, const void *w, size_t r, size_t c,
    unsigned t, const float *x, float *y, size_t batch)
{ return nya_cuda_matmul_typed(p, w, r, c, t, x, y, batch); }
const nya_backend_interface *nya_cuda_backend(void)
{
    static const nya_backend_interface api = {NYA_BACKEND_CUDA, "cuda", NYA_COMPUTE_MATVEC | NYA_COMPUTE_QUANTIZED | NYA_COMPUTE_RESIDENT | NYA_COMPUTE_MATMUL,
        cuda_backend_create, cuda_backend_free, cuda_backend_active, cuda_backend_matvec, cuda_plan_create, cuda_plan_free, cuda_plan_token, cuda_plan_prefill, cuda_plan_stats, cuda_backend_matmul, NULL, cuda_plan_read_kv, NULL};
    return &api;
}
