/* Unintegrated launch-bound experiments derived from native weight reuse. */
/* Original Fyodor optional kernel: decode each compressed weight once per warp
   and reuse it across independent input vectors. Unlike q6_row_pair.cu this
   shares WEIGHTS across inputs, not activations across output rows. Opt-in
   runtime dispatch selects batches 2/4. Compile after CUDA/matvec.cu. */
template<unsigned TYPE, unsigned INPUTS>
__device__ __forceinline__ void nya_reuse_launch_3_impl(
    const unsigned char *weights, const float *input, float *output,
    unsigned long long columns, unsigned long long stride,
    unsigned rows, unsigned batch)
{
    unsigned lane = threadIdx.x & 31;
    unsigned long long row = (unsigned long long)blockIdx.x * 8 + (threadIdx.x >> 5);
    unsigned first = blockIdx.y * INPUTS;
    if (row >= rows || first >= batch) return;
    const unsigned char *w = weights + row * stride;
    float sums[INPUTS] = {0};
    for (unsigned long long base = 0; base < columns; base += 256) {
        const unsigned char *b = w + base / 256 * (TYPE == 12 ? 144 : 210);
        float d = nya_half_aligned(b + (TYPE == 12 ? 0 : 208));
        float minimum = 0;
        unsigned S[3] = {0};
        if (TYPE == 12) {
            minimum = nya_half_aligned(b + 2);
            const unsigned *s = (const unsigned *)(b + 4);
            S[0] = s[0]; S[1] = s[1]; S[2] = s[2];
        }
        #pragma unroll
        for (unsigned g = 0; g < 8; ++g) {
            float weight;
            if (TYPE == 12) {
                unsigned shift = (g % 4) * 8;
                unsigned sc = g < 4 ? (S[0] >> shift) & 63U : ((S[2] >> shift) & 15U) | (((S[0] >> (shift + 6)) & 3U) << 4);
                unsigned mn = g < 4 ? (S[1] >> shift) & 63U : ((S[2] >> (shift + 4)) & 15U) | (((S[1] >> (shift + 6)) & 3U) << 4);
                unsigned q = (b[16 + g / 2 * 32 + lane] >> (g % 2 * 4)) & 15U;
                weight = d * (float)sc * (float)q - minimum * (float)mn;
            } else {
                unsigned half = g / 4, quarter = g % 4;
                unsigned lo = b[half * 64 + quarter % 2 * 32 + lane], hi = b[128 + half * 32 + lane];
                int q = (int)(((lo >> (quarter / 2 * 4)) & 15U) | (((hi >> (quarter * 2)) & 3U) << 4)) - 32;
                int sc = ((const signed char *)(b + 192))[g * 2 + lane / 16];
                weight = d * (float)sc * (float)q;
            }
            #pragma unroll
            for (unsigned i = 0; i < INPUTS; ++i) if (first + i < batch)
                sums[i] += weight * input[(unsigned long long)(first + i) * columns + base + g * 32 + lane];
        }
    }
    #pragma unroll
    for (unsigned i = 0; i < INPUTS; ++i) {
        float sum = sums[i];
        for (int step = 16; step; step >>= 1) sum += __shfl_down_sync(0xffffffffU, sum, step);
        if (!lane && first + i < batch) output[(unsigned long long)(first + i) * rows + row] = sum;
    }
}
#define NYA_REUSE_LAUNCH_3(TYPE, INPUTS) \
extern "C" __global__ __launch_bounds__(256,3) void nya_reuse_launch_3_##TYPE##_##INPUTS(const unsigned char *w, \
 const float *x, float *y, unsigned long long columns, unsigned long long stride, \
 unsigned rows, unsigned batch) { \
 nya_reuse_launch_3_impl<TYPE, INPUTS>(w, x, y, columns, stride, rows, batch); }
NYA_REUSE_LAUNCH_3(12, 1) NYA_REUSE_LAUNCH_3(12, 2) NYA_REUSE_LAUNCH_3(12, 4)
NYA_REUSE_LAUNCH_3(14, 1) NYA_REUSE_LAUNCH_3(14, 2) NYA_REUSE_LAUNCH_3(14, 4)
#undef NYA_REUSE_LAUNCH_3

/* Original Fyodor optional kernel: decode each compressed weight once per warp
   and reuse it across independent input vectors. Unlike q6_row_pair.cu this
   shares WEIGHTS across inputs, not activations across output rows. Opt-in
   runtime dispatch selects batches 2/4. Compile after CUDA/matvec.cu. */
template<unsigned TYPE, unsigned INPUTS>
__device__ __forceinline__ void nya_reuse_launch_4_impl(
    const unsigned char *weights, const float *input, float *output,
    unsigned long long columns, unsigned long long stride,
    unsigned rows, unsigned batch)
{
    unsigned lane = threadIdx.x & 31;
    unsigned long long row = (unsigned long long)blockIdx.x * 8 + (threadIdx.x >> 5);
    unsigned first = blockIdx.y * INPUTS;
    if (row >= rows || first >= batch) return;
    const unsigned char *w = weights + row * stride;
    float sums[INPUTS] = {0};
    for (unsigned long long base = 0; base < columns; base += 256) {
        const unsigned char *b = w + base / 256 * (TYPE == 12 ? 144 : 210);
        float d = nya_half_aligned(b + (TYPE == 12 ? 0 : 208));
        float minimum = 0;
        unsigned S[3] = {0};
        if (TYPE == 12) {
            minimum = nya_half_aligned(b + 2);
            const unsigned *s = (const unsigned *)(b + 4);
            S[0] = s[0]; S[1] = s[1]; S[2] = s[2];
        }
        #pragma unroll
        for (unsigned g = 0; g < 8; ++g) {
            float weight;
            if (TYPE == 12) {
                unsigned shift = (g % 4) * 8;
                unsigned sc = g < 4 ? (S[0] >> shift) & 63U : ((S[2] >> shift) & 15U) | (((S[0] >> (shift + 6)) & 3U) << 4);
                unsigned mn = g < 4 ? (S[1] >> shift) & 63U : ((S[2] >> (shift + 4)) & 15U) | (((S[1] >> (shift + 6)) & 3U) << 4);
                unsigned q = (b[16 + g / 2 * 32 + lane] >> (g % 2 * 4)) & 15U;
                weight = d * (float)sc * (float)q - minimum * (float)mn;
            } else {
                unsigned half = g / 4, quarter = g % 4;
                unsigned lo = b[half * 64 + quarter % 2 * 32 + lane], hi = b[128 + half * 32 + lane];
                int q = (int)(((lo >> (quarter / 2 * 4)) & 15U) | (((hi >> (quarter * 2)) & 3U) << 4)) - 32;
                int sc = ((const signed char *)(b + 192))[g * 2 + lane / 16];
                weight = d * (float)sc * (float)q;
            }
            #pragma unroll
            for (unsigned i = 0; i < INPUTS; ++i) if (first + i < batch)
                sums[i] += weight * input[(unsigned long long)(first + i) * columns + base + g * 32 + lane];
        }
    }
    #pragma unroll
    for (unsigned i = 0; i < INPUTS; ++i) {
        float sum = sums[i];
        for (int step = 16; step; step >>= 1) sum += __shfl_down_sync(0xffffffffU, sum, step);
        if (!lane && first + i < batch) output[(unsigned long long)(first + i) * rows + row] = sum;
    }
}
#define NYA_REUSE_LAUNCH_4(TYPE, INPUTS) \
extern "C" __global__ __launch_bounds__(256,4) void nya_reuse_launch_4_##TYPE##_##INPUTS(const unsigned char *w, \
 const float *x, float *y, unsigned long long columns, unsigned long long stride, \
 unsigned rows, unsigned batch) { \
 nya_reuse_launch_4_impl<TYPE, INPUTS>(w, x, y, columns, stride, rows, batch); }
NYA_REUSE_LAUNCH_4(12, 1) NYA_REUSE_LAUNCH_4(12, 2) NYA_REUSE_LAUNCH_4(12, 4)
NYA_REUSE_LAUNCH_4(14, 1) NYA_REUSE_LAUNCH_4(14, 2) NYA_REUSE_LAUNCH_4(14, 4)
#undef NYA_REUSE_LAUNCH_4

/* Original Fyodor optional kernel: decode each compressed weight once per warp
   and reuse it across independent input vectors. Unlike q6_row_pair.cu this
   shares WEIGHTS across inputs, not activations across output rows. Opt-in
   runtime dispatch selects batches 2/4. Compile after CUDA/matvec.cu. */
template<unsigned TYPE, unsigned INPUTS>
__device__ __forceinline__ void nya_reuse_launch_6_impl(
    const unsigned char *weights, const float *input, float *output,
    unsigned long long columns, unsigned long long stride,
    unsigned rows, unsigned batch)
{
    unsigned lane = threadIdx.x & 31;
    unsigned long long row = (unsigned long long)blockIdx.x * 8 + (threadIdx.x >> 5);
    unsigned first = blockIdx.y * INPUTS;
    if (row >= rows || first >= batch) return;
    const unsigned char *w = weights + row * stride;
    float sums[INPUTS] = {0};
    for (unsigned long long base = 0; base < columns; base += 256) {
        const unsigned char *b = w + base / 256 * (TYPE == 12 ? 144 : 210);
        float d = nya_half_aligned(b + (TYPE == 12 ? 0 : 208));
        float minimum = 0;
        unsigned S[3] = {0};
        if (TYPE == 12) {
            minimum = nya_half_aligned(b + 2);
            const unsigned *s = (const unsigned *)(b + 4);
            S[0] = s[0]; S[1] = s[1]; S[2] = s[2];
        }
        #pragma unroll
        for (unsigned g = 0; g < 8; ++g) {
            float weight;
            if (TYPE == 12) {
                unsigned shift = (g % 4) * 8;
                unsigned sc = g < 4 ? (S[0] >> shift) & 63U : ((S[2] >> shift) & 15U) | (((S[0] >> (shift + 6)) & 3U) << 4);
                unsigned mn = g < 4 ? (S[1] >> shift) & 63U : ((S[2] >> (shift + 4)) & 15U) | (((S[1] >> (shift + 6)) & 3U) << 4);
                unsigned q = (b[16 + g / 2 * 32 + lane] >> (g % 2 * 4)) & 15U;
                weight = d * (float)sc * (float)q - minimum * (float)mn;
            } else {
                unsigned half = g / 4, quarter = g % 4;
                unsigned lo = b[half * 64 + quarter % 2 * 32 + lane], hi = b[128 + half * 32 + lane];
                int q = (int)(((lo >> (quarter / 2 * 4)) & 15U) | (((hi >> (quarter * 2)) & 3U) << 4)) - 32;
                int sc = ((const signed char *)(b + 192))[g * 2 + lane / 16];
                weight = d * (float)sc * (float)q;
            }
            #pragma unroll
            for (unsigned i = 0; i < INPUTS; ++i) if (first + i < batch)
                sums[i] += weight * input[(unsigned long long)(first + i) * columns + base + g * 32 + lane];
        }
    }
    #pragma unroll
    for (unsigned i = 0; i < INPUTS; ++i) {
        float sum = sums[i];
        for (int step = 16; step; step >>= 1) sum += __shfl_down_sync(0xffffffffU, sum, step);
        if (!lane && first + i < batch) output[(unsigned long long)(first + i) * rows + row] = sum;
    }
}
#define NYA_REUSE_LAUNCH_6(TYPE, INPUTS) \
extern "C" __global__ __launch_bounds__(256,6) void nya_reuse_launch_6_##TYPE##_##INPUTS(const unsigned char *w, \
 const float *x, float *y, unsigned long long columns, unsigned long long stride, \
 unsigned rows, unsigned batch) { \
 nya_reuse_launch_6_impl<TYPE, INPUTS>(w, x, y, columns, stride, rows, batch); }
NYA_REUSE_LAUNCH_6(12, 1) NYA_REUSE_LAUNCH_6(12, 2) NYA_REUSE_LAUNCH_6(12, 4)
NYA_REUSE_LAUNCH_6(14, 1) NYA_REUSE_LAUNCH_6(14, 2) NYA_REUSE_LAUNCH_6(14, 4)
#undef NYA_REUSE_LAUNCH_6
