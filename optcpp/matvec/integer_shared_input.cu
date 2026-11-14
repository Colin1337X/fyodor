/* Original Fyodor experiment: stage each 256-column float4 activation tile
   once per CTA for eight output rows. Native Q4_K arithmetic, accumulation
   and warp reduction order remain unchanged. No production dispatch. */
extern "C" __global__ __launch_bounds__(256, 3)
void nya_integer_shared_input_8_3(const unsigned char *weights,
    const float *input, float *output, unsigned long long columns,
    unsigned long long stride, unsigned rows, unsigned batch)
{
    unsigned lane = threadIdx.x & 31;
    unsigned long long row = (unsigned long long)blockIdx.x * 8 + (threadIdx.x >> 5);
    if (batch != 4 || blockDim.x != 256 || blockDim.y != 1 || blockDim.z != 1) return;
    int active = row < rows;
    const unsigned char *w = active ? weights + row * stride : weights;
    __shared__ float4 tile[256];
    float sums[4] = {0};
    for (unsigned long long base = 0; base < columns; base += 256) {
        tile[threadIdx.x] = ((const float4 *)input)[base + threadIdx.x];
        __syncthreads();
        if (active) {
        const unsigned char *b = w + base / 256 * 144;
        float d = nya_half_aligned(b), minimum = nya_half_aligned(b + 2);
        const unsigned *metadata = (const unsigned *)(b + 4);
        unsigned a = metadata[0], c = metadata[1], e = metadata[2];
        unsigned scales[2] = {a & 0x3f3f3f3fU,
            (e & 0x0f0f0f0fU) | ((a & 0xc0c0c0c0U) >> 2)};
        unsigned minima[2] = {c & 0x3f3f3f3fU,
            ((e >> 4) & 0x0f0f0f0fU) | ((c & 0xc0c0c0c0U) >> 2)};
        #pragma unroll
        for (unsigned g = 0; g < 8; ++g) {
            unsigned shift = (g % 4) * 8;
            unsigned sc = (scales[g / 4] >> shift) & 255U;
            unsigned mn = (minima[g / 4] >> shift) & 255U;
            unsigned q = (b[16 + g / 2 * 32 + lane] >> (g % 2 * 4)) & 15U;
            float weight = d * (float)(sc * q) - minimum * (float)mn;
            float4 x = tile[g * 32 + lane];
            sums[0] += weight * x.x;
            sums[1] += weight * x.y;
            sums[2] += weight * x.z;
            sums[3] += weight * x.w;
        }
        }
        __syncthreads(); /* Tail-row threads must also reach both barriers. */
    }
    if (!active) return;
    #pragma unroll
    for (unsigned i = 0; i < 4; ++i) {
        float sum = sums[i];
        for (int step = 16; step; step >>= 1) sum += __shfl_down_sync(0xffffffffU, sum, step);
        if (!lane) output[(unsigned long long)i * rows + row] = sum;
    }
}
