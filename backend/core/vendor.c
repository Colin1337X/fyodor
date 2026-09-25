#include "vendor.h"
#include "llm_internal.h"
#include <errno.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <dlfcn.h>
#endif

void *nya_vendor_open(const char *path, const char *name)
{
    if ((!path || !*path) && (!name || !*name)) return NULL;
#ifdef _WIN32
    wchar_t filename[32768];
    if (path && *path) {
        if (!MultiByteToWideChar(CP_UTF8,MB_ERR_INVALID_CHARS,path,-1,filename,32768)) return NULL;
    } else {
        /* Packaged optional libraries are resolved beside the executable.
           Do not search the current working directory or an ambient PATH. */
        DWORD n=GetModuleFileNameW(NULL,filename,32768);
        if (!n || n>=32768) return NULL;
        wchar_t *last=wcsrchr(filename,L'\\');
        if (!last) return NULL;
        size_t prefix=(size_t)(last+1-filename);
        if (!MultiByteToWideChar(CP_UTF8,MB_ERR_INVALID_CHARS,name,-1,last+1,(int)(32768-prefix))) return NULL;
    }
    return (void *)LoadLibraryExW(filename,NULL,LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR|LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
#else
    return dlopen(path && *path ? path : name,RTLD_NOW|RTLD_LOCAL);
#endif
}
int nya_vendor_symbol(void *library, const char *name, void *target, size_t size)
{
    if (!library || !name || !target) return -1;
#ifdef _WIN32
    FARPROC address=GetProcAddress((HMODULE)library,name);
#else
    void *address=dlsym(library,name);
#endif
    if (!address || size!=sizeof(address)) return -1;
    memcpy(target,&address,size);
    return 0;
}
void nya_vendor_close(void *library)
{
    if (!library) return;
#ifdef _WIN32
    FreeLibrary((HMODULE)library);
#else
    dlclose(library);
#endif
}
int nya_vendor_matrix(size_t rows, size_t columns, size_t batch, unsigned type,
    size_t *weights, size_t *input, size_t *output)
{
    size_t block=1, storage=4;
    switch (type) {
    case 0: break;
    case 1: case 30: storage=2; break;
    case 2: block=32; storage=18; break;
    case 8: block=32; storage=34; break;
    case 12: block=256; storage=144; break;
    case 14: block=256; storage=210; break;
    default: return -1;
    }
    /* Both vendor APIs use signed int matrix dimensions. Validate compressed
       storage as well as expanded F32 and the two batch-major host buffers. */
    if (!weights || !input || !output || !rows || !columns || !batch || rows>INT_MAX || columns>INT_MAX || batch>INT_MAX ||
        columns%block || columns/block>SIZE_MAX/storage/rows ||
        columns>SIZE_MAX/sizeof(float)/rows || columns>SIZE_MAX/sizeof(float)/batch ||
        rows>SIZE_MAX/sizeof(float)/batch) return -1;
    *weights=rows*columns*sizeof(float); *input=batch*columns*sizeof(float); *output=batch*rows*sizeof(float);
    return 0;
}
void nya_vendor_decode(float *out, const void *source, size_t count, unsigned type)
{
    nya_llm_tensor tensor={0}; tensor.data=source; tensor.type=type;
    for (size_t i=0; i<count; ++i) out[i]=nya_llm_tensor_value(&tensor,i);
}
size_t nya_vendor_budget(size_t available, const char *variable)
{
    size_t reserve=available/10, floor=256U*1024U*1024U;
    if (reserve<floor) reserve=floor;
    if (available<=reserve) return 0;
    size_t budget=available-reserve;
    const char *text=getenv(variable);
    if (text) {
        char *end; errno=0;
        unsigned long long mib=strtoull(text,&end,10);
        if (errno || text[0]<'1' || text[0]>'9' || *end || mib>SIZE_MAX/(1024U*1024U)) {
            fprintf(stderr,"%s must be a positive MiB count\n",variable); return 0;
        }
        size_t requested=(size_t)mib*1024U*1024U;
        if (requested<budget) budget=requested;
    }
    return budget;
}
nya_vendor_weight *nya_vendor_find(nya_vendor_cache *cache, const void *source,
    size_t rows, size_t columns, unsigned type)
{
    nya_vendor_weight **at=&cache->head;
    while (*at) {
        nya_vendor_weight *w=*at;
        if (w->source==source && w->rows==rows && w->columns==columns && w->type==type) {
            *at=w->next; w->next=cache->head; cache->head=w; return w;
        }
        at=&w->next;
    }
    return NULL;
}
int nya_vendor_reserve(nya_vendor_cache *cache, size_t bytes, size_t scratch,
    nya_vendor_release release, void *context)
{
    if (scratch>cache->limit || bytes>cache->limit-scratch) return -1;
    while (cache->bytes>cache->limit-scratch-bytes) {
        nya_vendor_weight **tail=&cache->head;
        while ((*tail)->next) tail=&(*tail)->next;
        nya_vendor_weight *w=*tail; *tail=NULL;
        cache->bytes-=w->bytes; release(context,w->value); free(w);
    }
    return 0;
}
void nya_vendor_insert(nya_vendor_cache *cache, nya_vendor_weight *w)
{
    w->next=cache->head; cache->head=w; cache->bytes+=w->bytes;
}
void nya_vendor_clear(nya_vendor_cache *cache, nya_vendor_release release, void *context)
{
    while (cache->head) {
        nya_vendor_weight *w=cache->head; cache->head=w->next;
        release(context,w->value); free(w);
    }
    cache->bytes=0;
}
