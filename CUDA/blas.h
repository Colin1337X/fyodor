/* Minimal dynamically loaded cuBLAS C ABI. No cuBLAS headers, import library,
   host C++ compiler or SDK installation is required. These signatures and enum
   values follow NVIDIA's public cublas_api.h and cuBLAS Level-3 documentation.
   All vendor state stays inside the removable CUDA directory. */
#ifndef NYA_CUDA_BLAS_H
#define NYA_CUDA_BLAS_H
typedef struct cublasContext *nya_blas_handle;
typedef enum nya_blas_status { NYA_BLAS_SUCCESS = 0 } nya_blas_status;
typedef enum nya_blas_operation { NYA_BLAS_N = 0, NYA_BLAS_T = 1 } nya_blas_operation;
typedef enum nya_blas_math { NYA_BLAS_F32 = 0, NYA_BLAS_PEDANTIC = 2 } nya_blas_math;
typedef struct nya_cuda_blas {
    void *library;
    nya_blas_handle handle;
    size_t expansion_limit;
    int transpose;
    nya_blas_status (CUDAAPI *create)(nya_blas_handle *);
    nya_blas_status (CUDAAPI *destroy)(nya_blas_handle);
    nya_blas_status (CUDAAPI *set_stream)(nya_blas_handle, CUstream);
    nya_blas_status (CUDAAPI *set_workspace)(nya_blas_handle, void *, size_t);
    nya_blas_status (CUDAAPI *set_math)(nya_blas_handle, nya_blas_math);
    nya_blas_status (CUDAAPI *sgemm)(nya_blas_handle, nya_blas_operation, nya_blas_operation,
        int, int, int, const float *, const float *, int, const float *, int, const float *, float *, int);
} nya_cuda_blas;
/* Expansion is one bounded matrix, never a floating-point copy of the model.
   The workspace is separate and explicitly owned/accounted, including when
   cuBLAS is absent or initialization fails. */
#define NYA_BLAS_EXPANSION_MAX (64U * 1024U * 1024U)
#define NYA_BLAS_WORKSPACE (4U * 1024U * 1024U)
#endif
