/* CUDA device code only. Host allocation, module loading and error handling use
   the C driver API in cuda.c. A block reduces one matrix row; columns are read
   in parallel and quantized weights stay compressed in device memory. */
__device__ float nya_half(const unsigned char *p)
{
#ifndef NYA_CUDA_REFERENCE_MATH
    /* Byte assembly is safe for the two-byte alignment of packed GGUF scale
       fields. Hardware conversion preserves finite F16 values (including
       subnormals) exactly and avoids software exponent/normalization work. */
    unsigned short h = (unsigned short)((unsigned)p[0] | ((unsigned)p[1] << 8));
    float value;
    asm("cvt.f32.f16 %0, %1;" : "=f"(value) : "h"(h));
    return value;
#else
    unsigned int h = (unsigned int)p[0] | ((unsigned int)p[1] << 8);
    unsigned int sign = (h & 32768U) << 16, mantissa = h & 1023U;
    int exponent = (int)((h >> 10) & 31U);
    unsigned int bits;
    if (exponent == 0) {
        if (mantissa == 0) bits = sign;
        else {
            exponent = 1;
            while ((mantissa & 1024U) == 0) { mantissa <<= 1; --exponent; }
            bits = sign | ((unsigned int)(exponent + 112) << 23) | ((mantissa & 1023U) << 13);
        }
    } else if (exponent == 31) bits = sign | 0x7f800000U | (mantissa << 13);
    else bits = sign | ((unsigned int)(exponent + 112) << 23) | (mantissa << 13);
    return __uint_as_float(bits);
#endif
}

__device__ __forceinline__ float nya_weight(const unsigned char *p, unsigned long long i, unsigned int type)
{
    if (type == 0) return ((const float *)p)[i];
    if (type == 1) return nya_half(p + i * 2);
    if (type == 30) return __uint_as_float(((unsigned int)p[i * 2] | ((unsigned int)p[i * 2 + 1] << 8)) << 16);
    if (type == 2) {
        const unsigned char *b = p + (i / 32) * 18;
        unsigned int j = (unsigned int)(i % 32);
        int q = (int)((b[2 + j % 16] >> (j / 16 * 4)) & 15U) - 8;
        return nya_half(b) * (float)q;
    }
    if (type == 8) {
        const unsigned char *b = p + (i / 32) * 34;
        return nya_half(b) * (float)((const signed char *)(b + 2))[i % 32];
    }
    if (type == 12) {
        const unsigned char *b = p + (i / 256) * 144;
        unsigned int j = (unsigned int)(i % 256), group = j / 32;
        const unsigned char *s = b + 4;
        unsigned int scale = group < 4 ? s[group] & 63U : (s[group + 4] & 15U) | ((unsigned int)(s[group - 4] >> 6) << 4);
        unsigned int minimum = group < 4 ? s[group + 4] & 63U : (s[group + 4] >> 4) | ((unsigned int)(s[group] >> 6) << 4);
        unsigned int q = (b[16 + group / 2 * 32 + j % 32] >> (group % 2 * 4)) & 15U;
        return nya_half(b) * (float)scale * (float)q - nya_half(b + 2) * (float)minimum;
    }
    /* Host validation admits only Q6_K here. */
    const unsigned char *b = p + (i / 256) * 210;
    unsigned int j = (unsigned int)(i % 256), half = j / 128, quarter = j % 128 / 32, lane = j % 32;
    unsigned int lo = b[half * 64 + quarter % 2 * 32 + lane], hi = b[128 + half * 32 + lane];
    int q = (int)(((lo >> (quarter / 2 * 4)) & 15U) | (((hi >> (quarter * 2)) & 3U) << 4)) - 32;
    int scale = ((const signed char *)(b + 192))[j / 16];
    return nya_half(b + 208) * (float)scale * (float)q;
}

/* Each warp reduces one row. Constant storage IDs let NVRTC eliminate unrelated
   decoding branches for each entry point. Eight rows share a block without any
   shared-memory reduction or block-wide barrier. The scalar generic entry point
   below remains available in explicit reference mode. */
__device__ __forceinline__ void nya_row_dot(const unsigned char *weights,
    const float *input, float *output, unsigned long long columns,
    unsigned long long row_bytes, unsigned long long rows, unsigned TYPE)
{
    unsigned lane = threadIdx.x & 31;
    unsigned long long row = (unsigned long long)blockIdx.x * 8 + (threadIdx.x >> 5);
    if (row >= rows) return;
    const unsigned char *w = weights + row * row_bytes;
    float sum = 0;
    if (TYPE == 12) {
        for (unsigned long long base = 0; base < columns; base += 256) {
            const unsigned char *b = w + base / 256 * 144;
            float d = nya_half(b), minimum = nya_half(b+2);
            const unsigned char *s = b+4;
            #pragma unroll
            for (unsigned g = 0; g < 8; ++g) {
                unsigned sc = g < 4 ? s[g] & 63U : (s[g+4] & 15U) | ((unsigned)(s[g-4] >> 6) << 4);
                unsigned mn = g < 4 ? s[g+4] & 63U : (s[g+4] >> 4) | ((unsigned)(s[g] >> 6) << 4);
                unsigned q = (b[16+g/2*32+lane] >> (g%2*4)) & 15U;
                sum += (d*(float)sc*(float)q - minimum*(float)mn) * input[base+g*32+lane];
            }
        }
    } else if (TYPE == 14) {
        for (unsigned long long base = 0; base < columns; base += 256) {
            const unsigned char *b = w + base / 256 * 210;
            float d = nya_half(b+208);
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
    if (!lane) output[row] = sum;
}
#define NYA_DOT(TYPE) extern "C" __global__ void nya_dot_##TYPE(const unsigned char *w, const float *x, float *y, unsigned long long c, unsigned long long s, unsigned long long r) { nya_row_dot(w,x,y,c,s,r,TYPE); }
NYA_DOT(0) NYA_DOT(1) NYA_DOT(2) NYA_DOT(8) NYA_DOT(12) NYA_DOT(14) NYA_DOT(30)
#undef NYA_DOT


extern "C" __global__ void nya_matvec(const unsigned char *weights, const float *input,
    float *output, unsigned long long columns, unsigned long long row_bytes, unsigned int type)
{
    __shared__ float partial[256];
    unsigned int lane = threadIdx.x;
    const unsigned char *row = weights + (unsigned long long)blockIdx.x * row_bytes;
    float sum = 0.0f;
    for (unsigned long long col = lane; col < columns; col += 256) sum += nya_weight(row, col, type) * input[col];
    partial[lane] = sum;
    __syncthreads();
    for (unsigned int stride = 128; stride != 0; stride >>= 1) {
        if (lane < stride) partial[lane] += partial[lane + stride];
        __syncthreads();
    }
    if (lane == 0) output[blockIdx.x] = partial[0];
}
