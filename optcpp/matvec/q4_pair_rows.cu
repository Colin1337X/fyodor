/* Original Fyodor experiment: share an interleaved input load across two
   rows per warp. Native Q4_K blocks; original per-row accumulation order.
   Requires block (256,1,1), four inputs and columns divisible by 256.
   No production dispatch. */
extern "C" __global__ __launch_bounds__(256, 3)
void nya_q4_pair_rows(const unsigned char *weights,
    const float *input, float *output, unsigned long long columns,
    unsigned long long stride, unsigned rows, unsigned batch)
{
    if (blockDim.x != 256 || blockDim.y != 1 || blockDim.z != 1 ||
        batch != 4 || !columns || columns % 256) return;
    unsigned lane = threadIdx.x & 31;
    unsigned long long row = (unsigned long long)blockIdx.x * 16 + (threadIdx.x >> 5) * 2;
    if (row >= rows || batch != 4) return;
    float sums[2][4] = {{0},{0}};
    for (unsigned long long base = 0; base < columns; base += 256) {
        const unsigned char *blocks[2];
        float d[2], minimum[2];
        unsigned scales[2][2], minima[2][2];
        #pragma unroll
        for (unsigned r = 0; r < 2; ++r) {
            bool valid = row + r < rows;
            blocks[r] = weights + (valid ? row + r : row) * stride + base / 256 * 144;
            const unsigned char *b = blocks[r];
            d[r] = valid ? nya_half_aligned(b) : 0.0f;
            minimum[r] = valid ? nya_half_aligned(b + 2) : 0.0f;
            const unsigned *metadata = (const unsigned *)(b + 4);
            unsigned a = metadata[0], c = metadata[1], e = metadata[2];
            scales[r][0] = a & 0x3f3f3f3fU;
            scales[r][1] = (e & 0x0f0f0f0fU) | ((a & 0xc0c0c0c0U) >> 2);
            minima[r][0] = c & 0x3f3f3f3fU;
            minima[r][1] = ((e >> 4) & 0x0f0f0f0fU) | ((c & 0xc0c0c0c0U) >> 2);
        }
        #pragma unroll
        for (unsigned g = 0; g < 8; ++g) {
            float4 x = ((const float4 *)input)[base + g * 32 + lane];
            #pragma unroll
            for (unsigned r = 0; r < 2; ++r) {
                unsigned shift = (g % 4) * 8;
                unsigned sc = (scales[r][g / 4] >> shift) & 255U;
                unsigned mn = (minima[r][g / 4] >> shift) & 255U;
                unsigned q = (blocks[r][16 + g / 2 * 32 + lane] >> (g % 2 * 4)) & 15U;
                float weight = d[r] * (float)(sc * q) - minimum[r] * (float)mn;
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
