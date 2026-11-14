/* Original Fyodor experiment: batch-four integer Q4_K using the existing
   lossless 160-byte layout from CUDA/q4_byte_pack.h. Unintegrated.
   Host admission must supply complete block-aligned packed matrices and
   float4-aligned interleaved input. Arithmetic/reduction order follows
   integer_launch.cu; GPU parity and performance remain to be measured. */
extern "C" __global__ __launch_bounds__(256, 3)
void nya_interleaved_byte_integer_12_4(const unsigned char *weights,
    const float *input, float *output, unsigned long long columns,
    unsigned long long stride, unsigned rows, unsigned batch)
{
    unsigned lane = threadIdx.x & 31;
    unsigned long long row = (unsigned long long)blockIdx.x * 8 + (threadIdx.x >> 5);
    if (row >= rows || batch != 4) return;
    const unsigned char *w = weights + row * stride;
    float sums[4] = {0};
    for (unsigned long long base = 0; base < columns; base += 256) {
        const unsigned char *b = w + base / 256 * 160;
        float d = nya_half_aligned(b), minimum = nya_half_aligned(b + 2);
        #pragma unroll
        for (unsigned g = 0; g < 8; ++g) {
            unsigned sc = b[4 + g], mn = b[12 + g];
            unsigned q = (b[32 + g / 2 * 32 + lane] >> (g % 2 * 4)) & 15U;
            float weight = d * (float)(sc * q) - minimum * (float)mn;
            float4 x = ((const float4 *)input)[base + g * 32 + lane];
            sums[0] += weight * x.x;
            sums[1] += weight * x.y;
            sums[2] += weight * x.z;
            sums[3] += weight * x.w;
        }
    }
    #pragma unroll
    for (unsigned i = 0; i < 4; ++i) {
        float sum = sums[i];
        for (int step = 16; step; step >>= 1) sum += __shfl_down_sync(0xffffffffU, sum, step);
        if (!lane) output[(unsigned long long)i * rows + row] = sum;
    }
}
