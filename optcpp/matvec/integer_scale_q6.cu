/* Original Fyodor diagnostic: signed integer product, preserving zero sign. */
__device__ __forceinline__ void nya_row_dot_q6_integer_at(const unsigned char *weights,
    const float *input, float *output, unsigned long long columns,
    unsigned long long row_bytes, unsigned long long rows, unsigned TYPE, unsigned long long row, unsigned add = 0)
{
    unsigned lane = threadIdx.x & 31;
    if (row >= rows) return;
    const unsigned char *w = weights + row * row_bytes;
    float sum = 0;
    if (TYPE == 12) {
        for (unsigned long long base = 0; base < columns; base += 256) {
            const unsigned char *b = w + base / 256 * 144;
            float d = nya_half_aligned(b), minimum = nya_half_aligned(b+2);
            const unsigned *s = (const unsigned *)(b+4);
            unsigned S[3]={s[0],s[1],s[2]};
            #pragma unroll
            for (unsigned g = 0; g < 8; ++g) {
                unsigned shift=(g%4)*8;
                unsigned sc = g<4 ? (S[0]>>shift)&63U : ((S[2]>>shift)&15U) | (((S[0]>>(shift+6))&3U)<<4);
                unsigned mn = g<4 ? (S[1]>>shift)&63U : ((S[2]>>(shift+4))&15U) | (((S[1]>>(shift+6))&3U)<<4);
                unsigned q = (b[16+g/2*32+lane] >> (g%2*4)) & 15U;
                sum += (d*(float)sc*(float)q - minimum*(float)mn) * input[base+g*32+lane];
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
                float product = __int_as_float(__float_as_int((float)(sc*q)) | ((sc ^ q) & (-2147483647-1)));
                sum += d*product * input[base+g*32+lane];
            }
        }
    } else {
        for (unsigned long long i = lane; i < columns; i += 32) sum += nya_weight(w, i, TYPE) * input[i];
    }
    for (int step = 16; step; step >>= 1) sum += __shfl_down_sync(0xffffffffU, sum, step);
    if (!lane) output[row] = add ? output[row]+sum : sum;
}

extern "C" __global__ void nya_dot_q6_integer_14(const unsigned char *w, const float *x, float *y, unsigned long long c, unsigned long long s, unsigned long long r) { nya_row_dot_q6_integer_at(w,x,y,c,s,r,14,(unsigned long long)blockIdx.x*8+(threadIdx.x>>5)); }

/* Larger projections earned a runtime trial; small shapes keep native arithmetic. */
#define NYA_Q6_LARGE(NAME, ADD) extern "C" __global__ void NAME(const unsigned char *w, const float *x, float *y, unsigned long long c, unsigned long long s, unsigned long long r) {     unsigned long long row=(unsigned long long)blockIdx.x*8+(threadIdx.x>>5);     if (c>=2048 && r>=2048) nya_row_dot_q6_integer_at(w,x,y,c,s,r,14,row,ADD);     else nya_row_dot_at(w,x,y,c,s,r,14,row,ADD); }
NYA_Q6_LARGE(nya_dot_q6_large,0)
NYA_Q6_LARGE(nya_add_q6_large,1)
#undef NYA_Q6_LARGE
