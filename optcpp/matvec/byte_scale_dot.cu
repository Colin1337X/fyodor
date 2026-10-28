/* Original Fyodor diagnostic: aligned byte-metadata Q4 row dot. */
__device__ __forceinline__ void nya_row_dot_byte_at(const unsigned char *weights,
    const float *input, float *output, unsigned long long columns,
    unsigned long long row_bytes, unsigned long long rows, unsigned TYPE, unsigned long long row, unsigned add = 0)
{
    unsigned lane = threadIdx.x & 31;
    if (row >= rows) return;
    const unsigned char *w = weights + row * row_bytes;
    float sum = 0;
    if (TYPE == 12) {
        for (unsigned long long base = 0; base < columns; base += 256) {
            const unsigned char *b = w + base / 256 * 160;
            float d = nya_half_aligned(b), minimum = nya_half_aligned(b+2);
            const unsigned *s = (const unsigned *)(b+4);
            unsigned S[4]={s[0],s[1],s[2],s[3]};
            #pragma unroll
            for (unsigned g = 0; g < 8; ++g) {
                unsigned shift=(g%4)*8;
                unsigned sc = (S[g/4]>>shift)&255U;
                unsigned mn = (S[2+g/4]>>shift)&255U;
                unsigned q = (b[32+g/2*32+lane] >> (g%2*4)) & 15U;
                sum += (d*(float)(sc*q) - minimum*(float)mn) * input[base+g*32+lane];
            }
        }
    } else if (TYPE == 14) {
        for (unsigned long long base = 0; base < columns; base += 256) {
            const unsigned char *b = w + base / 256 * 210;
            float d = nya_half_aligned(b+208);
            #pragma unroll
            for (unsigned g = 0; g < 8; ++g) {
                unsigned half = g/4, quarter = g%4;
                unsigned lo = b[half*64+quarter%2*32+lane], hi = b[128+half*32+lane];
                int q = (int)(((lo >> (quarter/2*4)) & 15U) | (((hi >> (quarter*2)) & 3U) << 4)) - 32;
                int sc = ((const signed char *)(b+192))[g*2+lane/16];
                sum += d*(float)sc*(float)q * input[base+g*32+lane];
            }
        }
    } else {
        for (unsigned long long i = lane; i < columns; i += 32) sum += nya_weight(w, i, TYPE) * input[i];
    }
    for (int step = 16; step; step >>= 1) sum += __shfl_down_sync(0xffffffffU, sum, step);
    if (!lane) output[row] = add ? output[row]+sum : sum;
}

extern "C" __global__ void nya_dot_byte_12(const unsigned char *w, const float *x, float *y, unsigned long long c, unsigned long long s, unsigned long long r) { nya_row_dot_byte_at(w,x,y,c,s,r,12,(unsigned long long)blockIdx.x*8+(threadIdx.x>>5)); }
