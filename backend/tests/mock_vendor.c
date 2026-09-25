/* Test-only shared library. Never installed or named after a vendor library.
   This independently implements the external C ABIs with host arithmetic to
   exercise ownership, layout, error propagation and loader behavior without
   claiming hardware validation. Every fallible operation can fail once. */
#include <stdbool.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include "vendor_contract.h"
#ifdef _WIN32
#define API __declspec(dllexport)
#else
#define API __attribute__((visibility("default")))
#endif
static int live, countdown, device;
static size_t index_at(int row,int stride,int col) { return (size_t)row*(size_t)stride+(size_t)col; }
static int fail(void) { return countdown>0 && --countdown==0; }
static void *allocate(size_t bytes) { void *p=calloc(1,bytes); if (p) ++live; return p; }
static void release(void *p) { if (p) { --live; free(p); } }
API void nya_mock_fail_after(int n) { countdown=n; }
API int nya_mock_live(void) { return live; }
API hipError_t hipInit(unsigned flags) { return flags || fail(); }
API hipError_t hipGetDeviceCount(int *n) { *n=1; return (hipError_t)fail(); }
API hipError_t hipGetDevice(int *n) { *n=device; return (hipError_t)fail(); }
API hipError_t hipSetDevice(int n) { if (fail() || n) return 1; device=n; return 0; }
API hipError_t hipMemGetInfo(size_t *free_bytes,size_t *total) { *free_bytes=*total=(size_t)1024*1024*1024; return (hipError_t)fail(); }
API hipError_t hipMalloc(void **p,size_t n) { if (fail()) return 1; *p=allocate(n); return !*p; }
API hipError_t hipFree(void *p) { release(p); return 0; }
API hipError_t hipMemcpyAsync(void *dst,const void *src,size_t n,hipMemcpyKind kind,hipStream_t stream)
{ if (fail() || !stream || (kind!=1 && kind!=2)) return 1; memcpy(dst,src,n); return 0; }
API hipError_t hipStreamCreateWithFlags(hipStream_t *p,unsigned flags)
{ if (fail() || flags!=1) return 1; *p=allocate(1); return !*p; }
API hipError_t hipStreamDestroy(hipStream_t p) { release(p); return 0; }
API hipError_t hipStreamSynchronize(hipStream_t p) { return !p || fail(); }
API rocblas_status rocblas_create_handle(rocblas_handle *p) { if (fail()) return 1; *p=allocate(1); return !*p; }
API rocblas_status rocblas_destroy_handle(rocblas_handle p) { release(p); return 0; }
API rocblas_status rocblas_set_stream(rocblas_handle h,hipStream_t s) { return !h || !s || fail(); }
API rocblas_status rocblas_set_pointer_mode(rocblas_handle h,rocblas_pointer_mode mode) { return !h || mode || fail(); }
API rocblas_status rocblas_sgemm(rocblas_handle h,rocblas_operation ta,rocblas_operation tb,int m,int n,int k,const float *alpha,
    const float *a,int lda,const float *b,int ldb,const float *beta,float *c,int ldc)
{
    if (fail() || !h || ta!=112 || tb!=111 || lda<k || ldb<k || ldc<m) return 1;
    for (int j=0;j<n;++j) for (int i=0;i<m;++i) {
        double sum=0;
        for (int at=0;at<k;++at) sum+=(double)a[index_at(i,lda,at)]*b[index_at(j,ldb,at)];
        c[index_at(j,ldc,i)]=*alpha*(float)sum+*beta*c[index_at(j,ldc,i)];
    }
    return 0;
}
/* MLX handles are passed by value. Defining this independently of the adapter
   catches accidental pointer/struct calling-convention mismatches. */

typedef struct { int rows,cols; float *data; } array;
static void (*error_callback)(const char *,void *);
static void *error_payload;
static int mlx_fail(void)
{
    if (!fail()) return 0;
    if (error_callback) error_callback("injected test failure",error_payload);
    return 1;
}
API void mlx_set_error_handler(void (*fn)(const char *,void *),void *payload,void (*destructor)(void *))
{ (void)destructor; error_callback=fn; error_payload=payload; }
API mlx_device mlx_device_new_type(mlx_device_type type,int index)
{ mlx_device h={NULL}; if (!mlx_fail() && type<=MLX_GPU && !index) { h.ctx=allocate(sizeof(int)); if (h.ctx) *(int *)h.ctx=(int)type; } return h; }
API int mlx_device_is_available(bool *available,mlx_device h) { *available=h.ctx!=NULL; return mlx_fail(); }
API int mlx_device_free(mlx_device h) { release(h.ctx); return 0; }
API mlx_stream mlx_stream_new_thread_unsafe(mlx_device d)
{ mlx_stream h={NULL}; if (!mlx_fail() && d.ctx) { h.ctx=allocate(sizeof(int)); if (h.ctx) *(int *)h.ctx=*(int *)d.ctx; } return h; }
API int mlx_stream_free(mlx_stream h) { release(h.ctx); return 0; }
API int mlx_synchronize(mlx_stream h) { return !h.ctx || mlx_fail(); }
static mlx_array new_array(const float *data,int rows,int cols)
{
    mlx_array h={NULL}; array *a=allocate(sizeof(*a)); if (!a) return h;
    a->rows=rows; a->cols=cols; a->data=allocate((size_t)rows*(size_t)cols*sizeof(float));
    if (!a->data) { release(a); return h; }
    if (data) memcpy(a->data,data,(size_t)rows*(size_t)cols*sizeof(float));
    h.ctx=a; return h;
}
API mlx_array mlx_array_new_data(const void *data,const int *shape,int dim,mlx_dtype dtype)
{ mlx_array h={NULL}; return mlx_fail() || dim!=2 || dtype!=10 ? h : new_array(data,shape[0],shape[1]); }
API int mlx_array_free(mlx_array h) { if (h.ctx) { array *a=h.ctx; release(a->data); release(a); } return 0; }
API int mlx_array_eval(mlx_array h) { return !h.ctx || mlx_fail(); }
API const float *mlx_array_data_float32(mlx_array h) { return mlx_fail() || !h.ctx ? NULL : ((array *)h.ctx)->data; }
API int mlx_transpose(mlx_array *out,mlx_array in,mlx_stream stream)
{
    if (mlx_fail() || !stream.ctx || !in.ctx) return 1;
    array *a=in.ctx; *out=new_array(NULL,a->cols,a->rows); if (!out->ctx) return 1;
    array *b=out->ctx;
    for (int i=0;i<a->rows;++i) for (int j=0;j<a->cols;++j) b->data[index_at(j,a->rows,i)]=a->data[index_at(i,a->cols,j)];
    return 0;
}
API int mlx_contiguous(mlx_array *out,mlx_array in,bool col_major,mlx_stream stream)
{
    if (mlx_fail() || col_major || !stream.ctx || *(int *)stream.ctx!=1 || !in.ctx) return 1;
    array *a=in.ctx; *out=new_array(a->data,a->rows,a->cols); return !out->ctx;
}
API int mlx_copy(mlx_array *out,mlx_array in,mlx_stream stream)
{
    if (mlx_fail() || !stream.ctx || *(int *)stream.ctx!=0 || !in.ctx) return 1;
    array *a=in.ctx; *out=new_array(a->data,a->rows,a->cols); return !out->ctx;
}
API int mlx_matmul(mlx_array *out,mlx_array lhs,mlx_array rhs,mlx_stream stream)
{
    if (mlx_fail() || !stream.ctx || *(int *)stream.ctx!=1 || !lhs.ctx || !rhs.ctx) return 1;
    array *a=lhs.ctx,*b=rhs.ctx; if (a->cols!=b->rows) return 1;
    *out=new_array(NULL,a->rows,b->cols); if (!out->ctx) return 1;
    array *c=out->ctx;
    for (int i=0;i<a->rows;++i) for (int j=0;j<b->cols;++j) {
        double sum=0;
        for (int k=0;k<a->cols;++k) sum+=(double)a->data[index_at(i,a->cols,k)]*b->data[index_at(k,b->cols,j)];
        c->data[index_at(i,b->cols,j)]=(float)sum;
    }
    return 0;
}
API int mlx_get_memory_limit(size_t *bytes) { *bytes=(size_t)1024*1024*1024; return mlx_fail(); }
API int mlx_get_active_memory(size_t *bytes) { *bytes=0; return mlx_fail(); }
API int mlx_get_cache_memory(size_t *bytes) { *bytes=0; return mlx_fail(); }
