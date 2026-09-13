#ifndef NYA_CUTLASS_HOST_H
#define NYA_CUTLASS_HOST_H
#include "bridge.h"
typedef struct nya_cuda_cutlass {
    void *library;
    int (*gemm)(uintptr_t,uintptr_t,uintptr_t,int,int,int,int,uintptr_t);
} nya_cuda_cutlass;
#endif
