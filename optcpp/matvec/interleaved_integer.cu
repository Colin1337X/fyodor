/* Unintegrated Q4_K batch-four experiment. Original Fyodor code derived from
   interleaved_reuse.cu. The only arithmetic change combines scale*nibble in
   unsigned integer arithmetic (maximum 945); finite F16 coefficients make
   both products exactly representable in F32. NaN/Inf model coefficients are
   outside this proof. GPU contraction, whole-model parity and timing must be
   validated before runtime integration. No Q6 specialization is emitted. */
/* Original Fyodor optional resident batch-four implementation. Batch exactly four, grid Y one,
   16-byte-aligned float4 interleaved input. Weight arithmetic and each output's
   accumulation/reduction order match small_batch_reuse.cu. */
extern "C" __global__ void nya_integer_interleave_four(const float *input,float4 *output,
    unsigned long long columns)
{
    unsigned long long j=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;
    if(j<columns)output[j]=make_float4(input[j],input[columns+j],input[2*columns+j],input[3*columns+j]);
}
template<unsigned TYPE, unsigned INPUTS>
__device__ __forceinline__ void nya_interleaved_integer_impl(
    const unsigned char *weights, const float *input, float *output,
    unsigned long long columns, unsigned long long stride,
    unsigned rows, unsigned batch)
{
    unsigned lane = threadIdx.x & 31;
    unsigned long long row = (unsigned long long)blockIdx.x * 8 + (threadIdx.x >> 5);
    unsigned first = blockIdx.y * INPUTS;
    if (row >= rows || first >= batch || batch != 4) return;
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
                weight = d * (float)(sc * q) - minimum * (float)mn;
            } else {
                unsigned half = g / 4, quarter = g % 4;
                unsigned lo = b[half * 64 + quarter % 2 * 32 + lane], hi = b[128 + half * 32 + lane];
                int q = (int)(((lo >> (quarter / 2 * 4)) & 15U) | (((hi >> (quarter * 2)) & 3U) << 4)) - 32;
                int sc = ((const signed char *)(b + 192))[g * 2 + lane / 16];
                weight = d * (float)sc * (float)q;
            }
            float4 x=((const float4 *)input)[base+g*32+lane];
            sums[0]+=weight*x.x;
            sums[1]+=weight*x.y;
            sums[2]+=weight*x.z;
            sums[3]+=weight*x.w;
        }
    }
    #pragma unroll
    for (unsigned i = 0; i < INPUTS; ++i) {
        float sum = sums[i];
        for (int step = 16; step; step >>= 1) sum += __shfl_down_sync(0xffffffffU, sum, step);
        if (!lane && first + i < batch) output[(unsigned long long)(first + i) * rows + row] = sum;
    }
}
#define NYA_SMALL_REUSE(TYPE, INPUTS) \
extern "C" __global__ void nya_interleaved_integer_##TYPE##_##INPUTS(const unsigned char *w, \
 const float *x, float *y, unsigned long long columns, unsigned long long stride, \
 unsigned rows, unsigned batch) { \
 nya_interleaved_integer_impl<TYPE, INPUTS>(w, x, y, columns, stride, rows, batch); }
NYA_SMALL_REUSE(12, 4)

#undef NYA_SMALL_REUSE
