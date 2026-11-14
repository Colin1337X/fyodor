/* Original Fyodor experiment, not production dispatch.
   Native Q4_K weights, interleaved F32 input, four output columns.
   TF32 high/residual products change accumulation order; numerical admission
   must use the existing independent error bounds, not bitwise equivalence.
   Fragment coordinates follow NVIDIA PTX ISA 7.5 section 9.7.13.4.7:
   https://docs.nvidia.com/cuda/archive/11.5.2/parallel-thread-execution/index.html#warp-level-matrix-fragment-mma-1688
   Requires SM80+ and a (128,1,1) block. No CUTLASS dependency. */
static __device__ __forceinline__ unsigned nya_q4_tf32_round(float x)
{
    unsigned bits;
    asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(bits) : "f"(x));
    return bits;
}

static __device__ __forceinline__ void nya_q4_tf32_mma(float *c,
    const unsigned *a, const unsigned *b)
{
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]));
}

static __device__ __forceinline__ float nya_q4_tf32_weight(
    const unsigned char *weights, unsigned long long row,
    unsigned long long column, unsigned long long stride, unsigned rows)
{
    if (row >= rows) return 0.0f;
    const unsigned char *b = weights + row * stride + column / 256 * 144;
    unsigned g = (unsigned)(column % 256) / 32;
    unsigned lane = (unsigned)(column % 32);
    unsigned sc, mn;
    if (g < 4) {
        sc = b[4 + g] & 63U;
        mn = b[8 + g] & 63U;
    } else {
        sc = (b[8 + g] & 15U) | ((b[g] >> 6) << 4);
        mn = (b[8 + g] >> 4) | ((b[4 + g] >> 6) << 4);
    }
    unsigned q = (b[16 + g / 2 * 32 + lane] >> (g % 2 * 4)) & 15U;
    return nya_half_aligned(b) * (float)(sc * q) -
           nya_half_aligned(b + 2) * (float)mn;
}

extern "C" __global__ __launch_bounds__(128)
void nya_q4_tf32_three(const unsigned char *weights, const float *input,
    float *output, unsigned long long columns, unsigned long long stride,
    unsigned rows, unsigned batch)
{
    if (batch != 4 || !columns || columns % 256 ||
        blockDim.x != 128 || blockDim.y != 1 || blockDim.z != 1) return;
    unsigned lane = threadIdx.x & 31U;
    unsigned group = lane >> 2, thread = lane & 3U;
    unsigned long long tile = (unsigned long long)blockIdx.x * 64 +
                              (threadIdx.x >> 5) * 16;
    if (tile >= rows) return; /* Uniform over the entire participating warp. */
    float accum[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (unsigned long long k = 0; k < columns; k += 8) {
        unsigned ah[4], al[4], bh[2], bl[2];
        #pragma unroll
        for (unsigned i = 0; i < 4; ++i) {
            float a = nya_q4_tf32_weight(weights, tile + group + (i % 2) * 8,
                k + thread + (i / 2) * 4, stride, rows);
            ah[i] = nya_q4_tf32_round(a);
            al[i] = nya_q4_tf32_round(a - __uint_as_float(ah[i]));
        }
        #pragma unroll
        for (unsigned i = 0; i < 2; ++i) {
            float b = group < 4 ? input[(k + thread + i * 4) * 4 + group] : 0.0f;
            bh[i] = nya_q4_tf32_round(b);
            bl[i] = nya_q4_tf32_round(b - __uint_as_float(bh[i]));
        }
        /* Accumulate residual cross terms before the dominant product. */
        nya_q4_tf32_mma(accum, ah, bl);
        nya_q4_tf32_mma(accum, al, bh);
        nya_q4_tf32_mma(accum, ah, bh);
    }
    #pragma unroll
    for (unsigned i = 0; i < 4; ++i) {
        unsigned long long row = tile + group + (i / 2) * 8;
        unsigned column = thread * 2 + i % 2;
        if (row < rows && column < 4)
            output[(unsigned long long)column * rows + row] = accum[i];
    }
}
