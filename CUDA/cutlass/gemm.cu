/* C++ is confined to this removable CUTLASS directory. The implementation uses
 * the pinned upstream 3xTF32 operator: high*high + high*low + low*high with F32
 * accumulation. This reduces input rounding error relative to a single TF32
 * product; it is still tested against Fyodor's unchanged full-model bounds.
 * No quantized weight semantics are implemented here: Fyodor decodes one
 * bounded F32 chunk on the same stream before calling this kernel. */
#include "bridge.h"
#include <cuda_runtime.h>
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"

using Gemm = cutlass::gemm::device::Gemm<
    float, cutlass::layout::RowMajor,
    float, cutlass::layout::ColumnMajor,
    float, cutlass::layout::RowMajor, float,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<NYA_TILE_M,NYA_TILE_N,NYA_TILE_K>,
    cutlass::gemm::GemmShape<NYA_TILE_M/2,NYA_TILE_N/2,NYA_TILE_K>,
    cutlass::gemm::GemmShape<16,8,8>,
    cutlass::epilogue::thread::LinearCombination<float,4,float,float>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    3,4,4,false,cutlass::arch::OpMultiplyAddFastF32>;

unsigned nya_cutlass_abi_version(void) { return 1; }

int nya_cutlass_gemm(uintptr_t a, uintptr_t b, uintptr_t y,
    int batch, int rows, int k, int output_stride, uintptr_t stream) {
    /* Vectorized 128-bit accesses require aligned base pointers/strides. Reject
     * incompatible work before any launch; the caller retains native kernels. */
    if (!a || !b || !y || batch<1 || rows<1 || k<1 || output_stride<rows ||
        ((a|b|y)&15U) || (rows&3) || (k&3) || (output_stride&3)) return 1;
    Gemm::Arguments args({batch,rows,k},
        {reinterpret_cast<const float *>(a),k},
        {reinterpret_cast<const float *>(b),k},
        {reinterpret_cast<const float *>(y),output_stride},
        {reinterpret_cast<float *>(y),output_stride}, {1.0f,0.0f});
    Gemm op;
    if (op.can_implement(args) != cutlass::Status::kSuccess || op.get_workspace_size(args)) return 1;
    cudaStream_t cuda_stream=reinterpret_cast<cudaStream_t>(stream);
    cutlass::Status result=op.initialize(args,nullptr,cuda_stream);
    if (result != cutlass::Status::kSuccess) return -100-static_cast<int>(result);
    result=op(cuda_stream);
    return result == cutlass::Status::kSuccess ? 0 : -200-static_cast<int>(result);
}
