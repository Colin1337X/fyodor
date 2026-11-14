/* Original Fyodor experiment: two Q6_K rows share each float4 input load.
   Native 210-byte blocks, signed integer scale product and per-row reduction
   order match interleaved_q6_integer.cu. No production dispatch. */
extern "C" __global__ __launch_bounds__(256, 3)
void nya_q6_pair_rows_four(const unsigned char *weights,
    const float *input, float *output, unsigned long long columns,
    unsigned long long stride, unsigned rows, unsigned batch)
{
    if (blockDim.x != 256 || blockDim.y != 1 || blockDim.z != 1 ||
        gridDim.y != 1 || gridDim.z != 1 || batch != 4 ||
        !columns || columns % 256) return;
    unsigned lane = threadIdx.x & 31;
    unsigned long long row = (unsigned long long)blockIdx.x * 16 +
                             (threadIdx.x >> 5) * 2;
    if (row >= rows) return;
    float sums[2][4] = {{0}, {0}};
    for (unsigned long long base = 0; base < columns; base += 256) {
        const unsigned char *blocks[2];
        float scales[2];
        #pragma unroll
        for (unsigned r = 0; r < 2; ++r) {
            bool valid = row + r < rows;
            blocks[r] = weights + (valid ? row + r : row) * stride + base / 256 * 210;
            scales[r] = nya_half_aligned(blocks[r] + 208);
        }
        #pragma unroll
        for (unsigned g = 0; g < 8; ++g) {
            float4 x = ((const float4 *)input)[base + g * 32 + lane];
            unsigned half = g / 4, quarter = g % 4;
            #pragma unroll
            for (unsigned r = 0; r < 2; ++r) {
                const unsigned char *b = blocks[r];
                unsigned lo = b[half * 64 + quarter % 2 * 32 + lane];
                unsigned hi = b[128 + half * 32 + lane];
                int q = (int)(((lo >> (quarter / 2 * 4)) & 15U) |
                            (((hi >> (quarter * 2)) & 3U) << 4)) - 32;
                int sc = ((const signed char *)(b + 192))[g * 2 + lane / 16];
                float product = __int_as_float(__float_as_int((float)(sc * q)) |
                                               ((sc ^ q) & (-2147483647 - 1)));
                float weight = scales[r] * product;
                sums[r][0] += weight * x.x;
                sums[r][1] += weight * x.y;
                sums[r][2] += weight * x.z;
                sums[r][3] += weight * x.w;
            }
        }
    }
    #pragma unroll
    for (unsigned r = 0; r < 2; ++r) {
        #pragma unroll
        for (unsigned i = 0; i < 4; ++i) {
            float sum = sums[r][i];
            for (int step = 16; step; step >>= 1)
                sum += __shfl_down_sync(0xffffffffU, sum, step);
            if (!lane && row + r < rows)
                output[(unsigned long long)i * rows + row + r] = sum;
        }
    }
}
