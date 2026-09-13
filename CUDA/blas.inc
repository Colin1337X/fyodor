/* Optional F32 prefill accelerator. The native quantized kernels remain the
   universal CUDA path. Discovery failure does not disable the CUDA provider. */
static void nya_blas_close(nya_cuda_context *c)
{
    if (c->blas.handle) c->blas.destroy(c->blas.handle);
    nya_cuda_close(c->blas.library);
    memset(&c->blas, 0, sizeof(c->blas));
}

static void nya_blas_open(nya_cuda_context *c)
{
    const char *mode = getenv("NYA_CUDA_BLAS");
    c->blas.expansion_limit = NYA_BLAS_EXPANSION_MAX;
    if (c->reference) return;
    const char *scratch = getenv("NYA_CUDA_BLAS_SCRATCH_MIB");
    if (scratch) {
        char *end; unsigned long value = strtoul(scratch, &end, 10);
        if (*end || scratch[0] < '1' || scratch[0] > '9' || value > 64) {
            fprintf(stderr, "CUDA cuBLAS scratch must be 1..64 MiB; retaining native kernels\n");
            c->blas.expansion_limit = 0;
            return;
        }
        c->blas.expansion_limit = (size_t)value * 1024U * 1024U;
    }
    if (mode && !strcmp(mode, "0")) return;
    const char *explicit_path = getenv("NYA_CUDA_BLAS_LIBRARY");
    if (explicit_path && *explicit_path) c->blas.library = nya_cuda_open(explicit_path, 0);
    else {
#ifdef _WIN32
        /* A portable desktop package may carry optional DLLs next to its C
           sidecar. Resolve from the executable, never the working directory. */
        wchar_t executable[32768];
        DWORD length = GetModuleFileNameW(NULL, executable, 32768);
        if (length > 0 && length < 32768) {
            wchar_t *separator = wcsrchr(executable, L'\\');
            if (separator) {
                size_t prefix = (size_t)(separator + 1 - executable);
                const wchar_t *names[] = {L"cublas64_13.dll", L"cublas64_12.dll"};
                for (size_t i = 0; i < sizeof(names)/sizeof(names[0]); ++i) {
                    if (swprintf(executable+prefix, 32768-prefix, L"%ls", names[i]) < 0) break;
                    c->blas.library = (void *)LoadLibraryExW(executable, NULL, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
                    if (c->blas.library) break;
                }
            }
        }
        const char *toolkit = getenv("CUDA_PATH");
        const char *names[] = {"bin/x64/cublas64_13.dll", "bin/cublas64_13.dll", "bin/cublas64_12.dll"};
        if (toolkit && !c->blas.library) for (size_t i = 0; i < sizeof(names)/sizeof(names[0]); ++i) {
            char path[32768];
            int n = snprintf(path, sizeof(path), "%s/%s", toolkit, names[i]);
            if (n > 0 && (size_t)n < sizeof(path)) c->blas.library = nya_cuda_open(path, 0);
            if (c->blas.library) break;
        }
#else
        const char *names[] = {"libcublas.so.13", "libcublas.so.12", "libcublas.so"};
        for (size_t i = 0; i < sizeof(names)/sizeof(names[0]) && !c->blas.library; ++i)
            c->blas.library = nya_cuda_open(names[i], 0);
#endif
    }
    if (!c->blas.library) {
        if (explicit_path && *explicit_path) fprintf(stderr, "CUDA optional cuBLAS library could not be loaded; retaining native kernels\n");
        return;
    }
#define BLAS(field, name) if (nya_cuda_symbol(c->blas.library, name, &c->blas.field, sizeof(c->blas.field))) goto unavailable
    BLAS(create, "cublasCreate_v2"); BLAS(destroy, "cublasDestroy_v2");
    BLAS(set_stream, "cublasSetStream_v2"); BLAS(set_math, "cublasSetMathMode");
    BLAS(sgemm, "cublasSgemm_v2");
    /* CUDA 13 headers name this _v2; older runtimes export the original name. */
    if (nya_cuda_symbol(c->blas.library, "cublasSetWorkspace_v2", &c->blas.set_workspace, sizeof(c->blas.set_workspace)) &&
        nya_cuda_symbol(c->blas.library, "cublasSetWorkspace", &c->blas.set_workspace, sizeof(c->blas.set_workspace))) goto unavailable;
#undef BLAS
    if (c->blas.create(&c->blas.handle) != NYA_BLAS_SUCCESS ||
        c->blas.set_stream(c->blas.handle, c->stream) != NYA_BLAS_SUCCESS ||
        c->blas.set_math(c->blas.handle, NYA_BLAS_F32) != NYA_BLAS_SUCCESS) goto unavailable;
    if (getenv("NYA_CUDA_DEBUG")) fprintf(stderr, "CUDA prefill: optional cuBLAS F32 enabled\n");
    return;
unavailable:
    fprintf(stderr, "CUDA cuBLAS initialization failed; retaining native F32 kernels\n");
    {
    size_t expansion_limit=c->blas.expansion_limit;
    nya_blas_close(c);
    c->blas.expansion_limit=expansion_limit;
    }
}

/* The byte budget also bounds BLAS's signed-int dimensions. Return zero for
   unsupported/too-small shapes so no optional resource can block inference. */
static size_t nya_blas_bytes(nya_cuda_context *c, size_t rows, size_t columns, size_t batch)
{
    int available=c->blas.handle != NULL;
#ifdef NYA_ENABLE_CUTLASS
    available |= nya_cutlass_shape(c,rows,columns,batch);
#endif
    if (!available || batch < 32 || rows > INT_MAX || columns > INT_MAX ||
        batch > INT_MAX || columns == 0 || rows == 0 || columns > c->blas.expansion_limit/sizeof(float)) return 0;
    size_t limit = c->blas.expansion_limit / sizeof(float) / columns;
#ifdef NYA_ENABLE_CUTLASS
    if (nya_cutlass_shape(c,rows,columns,batch)) limit-=limit%4;
#endif
    if (rows > limit) rows = limit;
    return rows * columns * sizeof(float);
}

/* All operands and scratch are device addresses. W is row-major [R,K], X is
   row-major [B,K], and Y is row-major [B,R]. BLAS sees transposed W and column
   major X/Y, so no activation transpose/copy is necessary. Chunking W only
   changes the output row range, never a dot-product reduction or precision. */
static int nya_blas_multiply(nya_cuda_context *c, CUdeviceptr matrix, CUdeviceptr input,
    CUdeviceptr output, size_t rows, size_t columns, unsigned type, size_t stride,
    size_t batch, CUdeviceptr expanded, size_t expanded_bytes, CUdeviceptr workspace,
    unsigned long long *cutlass_calls)
{
    if (c->blas.handle && c->blas.set_workspace(c->blas.handle, (void *)(uintptr_t)workspace, NYA_BLAS_WORKSPACE) != NYA_BLAS_SUCCESS) return -1;
    size_t chunk_rows = expanded_bytes / sizeof(float) / columns;
#ifdef NYA_ENABLE_CUTLASS
    int cutlass=nya_cutlass_shape(c,rows,columns,batch);
    if (cutlass) chunk_rows-=chunk_rows%4;
#else
    (void)cutlass_calls;
#endif
    if (!chunk_rows) return -1;
    const float alpha = 1, beta = 0;
    for (size_t first = 0; first < rows; first += chunk_rows) {
        size_t count = rows - first;
        if (count > chunk_rows) count = chunk_rows;
        CUdeviceptr source = matrix + first * stride;
        unsigned long long k = columns, pitch = stride, elements = count * columns;
        CUdeviceptr operand = source;
        if (type != 0) {
            void *args[] = {&source, &expanded, &k, &pitch, &elements};
            if (c->launch(c->expand[type], (unsigned)((elements + 255)/256), 1, 1, 256, 1, 1, 0, c->stream, args, NULL) != CUDA_SUCCESS) return -1;
            operand = expanded;
        }
#ifdef NYA_ENABLE_CUTLASS
        if (cutlass) {
            int status=c->cutlass.gemm((uintptr_t)input,(uintptr_t)operand,
                (uintptr_t)(output+first*sizeof(float)),(int)batch,(int)count,(int)columns,(int)rows,(uintptr_t)c->stream);
            if (!status) { if (cutlass_calls) ++*cutlass_calls; continue; }
            if (status<0) {
                fprintf(stderr,"CUDA CUTLASS GEMM failed: status %d; request must replay on CPU\n",status);
                return -1;
            }
            /* A shape can be declined without enqueueing. cuBLAS can overwrite
             * this chunk; without it the caller replays the complete request. */
        }
#endif
        if (!c->blas.handle) return -1;
        nya_blas_status status = c->blas.sgemm(c->blas.handle, NYA_BLAS_T, NYA_BLAS_N,
            (int)count, (int)batch, (int)columns, &alpha, (const float *)(uintptr_t)operand, (int)columns,
            (const float *)(uintptr_t)input, (int)columns, &beta,
            (float *)(uintptr_t)(output + first*sizeof(float)), (int)rows);
        if (status != NYA_BLAS_SUCCESS) {
            fprintf(stderr, "CUDA cuBLAS GEMM failed: status %d; request must replay on CPU\n", (int)status);
            return -1;
        }
    }
    return 0;
}
