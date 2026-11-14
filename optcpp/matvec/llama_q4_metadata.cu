/* Q4_K two-group metadata decoding adapted from llama.cpp, MIT license.
   Copyright (c) 2023-2026 The ggml authors
   Revision 5266f24da75dc449bd56cbed7addb9c8e4a6a73e,
   ggml/src/ggml-cuda/vecdotq.cuh, vec_dot_q4_K_q8_1.
   See LICENSE.llama-cpp. Remaining projection machinery is original Fyodor.
   Experimental: native 144-byte blocks, F32 input, no Q8 activation conversion.
   Requires columns divisible by 256, aligned native blocks and float4 input;
   launch 256 threads, ceil(rows/8) blocks, batch exactly four. */
extern "C" __global__ __launch_bounds__(256, 3)
void nya_llama_q4_metadata_12_4(const unsigned char *weights,
    const float *input, float *output, unsigned long long columns,
    unsigned long long stride, unsigned rows, unsigned batch)
{
    unsigned lane = threadIdx.x & 31;
    unsigned long long row = (unsigned long long)blockIdx.x * 8 + (threadIdx.x >> 5);
    if (row >= rows || batch != 4) return;
    const unsigned char *w = weights + row * stride;
    float sums[4] = {0};
    for (unsigned long long base = 0; base < columns; base += 256) {
        const unsigned char *b = w + base / 256 * 144;
        float d = nya_half_aligned(b), minimum = nya_half_aligned(b + 2);
        const unsigned short *scales = (const unsigned short *)(b + 4);
        #pragma unroll
        for (unsigned j = 0; j < 4; ++j) {
            unsigned short aux[2];
            if (j < 2) {
                aux[0] = scales[j+0] & 0x3f3f;
                aux[1] = scales[j+2] & 0x3f3f;
            } else {
                aux[0] = ((scales[j+2] >> 0) & 0x0f0f) | ((scales[j-2] & 0xc0c0) >> 2);
                aux[1] = ((scales[j+2] >> 4) & 0x0f0f) | ((scales[j-0] & 0xc0c0) >> 2);
            }
            #pragma unroll
            for (unsigned k = 0; k < 2; ++k) {
                unsigned g = j * 2 + k;
                unsigned sc = (aux[0] >> (k * 8)) & 255U;
                unsigned mn = (aux[1] >> (k * 8)) & 255U;
                unsigned q = (b[16 + j * 32 + lane] >> (k * 4)) & 15U;
                float weight = d * (float)(sc * q) - minimum * (float)mn;
                float4 x = ((const float4 *)input)[base + g * 32 + lane];
                sums[0] += weight * x.x;
                sums[1] += weight * x.y;
                sums[2] += weight * x.z;
                sums[3] += weight * x.w;
            }
        }
    }
    #pragma unroll
    for (unsigned i = 0; i < 4; ++i) {
        float sum = sums[i];
        for (int step = 16; step; step >>= 1) sum += __shfl_down_sync(0xffffffffU, sum, step);
        if (!lane) output[(unsigned long long)i * rows + row] = sum;
    }
}
