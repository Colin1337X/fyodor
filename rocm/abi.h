#ifndef NYA_ROCM_ABI_H
#define NYA_ROCM_ABI_H
#include <stddef.h>
#include "types.h"
/* Minimal 64-bit HIP/rocBLAS C ABI, checked against the upstream headers listed
   in README.md. Runtime/driver types stay private to this detachable directory.
   No HIP C++ compiler, headers or import libraries are needed by the C host. */
typedef struct nya_rocm_api {
    hipError_t (*init)(unsigned);
    hipError_t (*device_count)(int *);
    hipError_t (*get_device)(int *);
    hipError_t (*set_device)(int);
    hipError_t (*memory_info)(size_t *,size_t *);
    hipError_t (*allocate)(void **,size_t);
    hipError_t (*deallocate)(void *);
    hipError_t (*copy_async)(void *,const void *,size_t,hipMemcpyKind,hipStream_t);
    hipError_t (*stream_create)(hipStream_t *,unsigned);
    hipError_t (*stream_destroy)(hipStream_t);
    hipError_t (*synchronize)(hipStream_t);
    rocblas_status (*blas_create)(rocblas_handle *);
    rocblas_status (*blas_destroy)(rocblas_handle);
    rocblas_status (*blas_stream)(rocblas_handle,hipStream_t);
    rocblas_status (*blas_pointer_mode)(rocblas_handle,rocblas_pointer_mode);
    rocblas_status (*sgemm)(rocblas_handle,rocblas_operation,rocblas_operation,int,int,int,const float *,const float *,int,
        const float *,int,const float *,float *,int);
} nya_rocm_api;
#define NYA_HIP_H2D hipMemcpyHostToDevice
#define NYA_HIP_D2H hipMemcpyDeviceToHost
#define NYA_ROCBLAS_N rocblas_operation_none
#define NYA_ROCBLAS_T rocblas_operation_transpose
#endif
