/* Compiled as C inside cuda.c, only while CUDA/cutlass exists. A missing or
 * incompatible optional DLL must not disable CUDA or require a C++ toolchain. */
static void nya_cutlass_close(nya_cuda_context *c)
{
    nya_cuda_close(c->cutlass.library);
    memset(&c->cutlass,0,sizeof(c->cutlass));
}

static void nya_cutlass_open(nya_cuda_context *c)
{
    const char *mode=getenv("NYA_CUDA_CUTLASS");
    /* Measured policy: keep cuBLAS when available; CUTLASS improves the native
     * fallback on validated dense prefill. Explicit 1 forces CUTLASS for tests,
     * 0 disables it. Selection is fixed before a request owns any buffers. */
    if (c->reference || (mode && !strcmp(mode,"0")) || (!mode && c->blas.handle)) return;
    if (mode && strcmp(mode,"1")) {
        fprintf(stderr,"CUDA CUTLASS selector must be 0 or 1; retaining existing kernels\n");
        return;
    }
    int major=0;
    if (c->attribute(&major,CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,c->device) != CUDA_SUCCESS || major<8) return;
    const char *path=getenv("NYA_CUDA_CUTLASS_LIBRARY");
    if (path && *path) c->cutlass.library=nya_cuda_open(path,0);
    else {
#ifdef _WIN32
        wchar_t executable[32768];
        DWORD length=GetModuleFileNameW(NULL,executable,32768);
        if (length>0 && length<32768) {
            wchar_t *separator=wcsrchr(executable,L'\\');
            if (separator) {
                size_t prefix=(size_t)(separator+1-executable);
                if (swprintf(executable+prefix,32768-prefix,L"fyodor-cutlass.dll")>=0)
                    c->cutlass.library=(void *)LoadLibraryExW(executable,NULL,LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR|LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
            }
        }
#else
        c->cutlass.library=nya_cuda_open("libfyodor-cutlass.so",0);
#endif
    }
    unsigned (*version)(void)=NULL;
    if (!c->cutlass.library ||
        nya_cuda_symbol(c->cutlass.library,"nya_cutlass_abi_version",&version,sizeof(version)) ||
        nya_cuda_symbol(c->cutlass.library,"nya_cutlass_gemm",&c->cutlass.gemm,sizeof(c->cutlass.gemm)) || version()!=1) {
        if (mode || (path && *path) || getenv("NYA_CUDA_DEBUG"))
            fprintf(stderr,"CUDA optional CUTLASS bridge unavailable or incompatible; retaining existing kernels\n");
        nya_cutlass_close(c); return;
    }
    if (getenv("NYA_CUDA_DEBUG")) fprintf(stderr,"CUDA prefill: optional CUTLASS 3xTF32 enabled\n");
}

static int nya_cutlass_shape(const nya_cuda_context *c, size_t rows, size_t columns, size_t batch)
{
    return c->cutlass.library && batch>=32 && rows && columns &&
        !(rows&3) && !(columns&3) && rows<=INT_MAX && columns<=INT_MAX && batch<=INT_MAX;
}
