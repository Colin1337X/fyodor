/* Original experimental variant of q4_tf32_three.cu. Compile after that file
   for its TF32 conversion/MMA helpers. Hoist Q4_K metadata out of the K=8 loop;
   keep the same increasing-column order and three-product accumulation.
   No runtime dispatch or numerical/performance admission. */
extern "C" __global__ __launch_bounds__(128)
void nya_q4_tf32_three_hoisted(const unsigned char *weights, const float *input,
    float *output, unsigned long long columns, unsigned long long stride,
    unsigned rows, unsigned batch)
{
    if (batch != 4 || !columns || columns % 256 ||
        blockDim.x != 128 || blockDim.y != 1 || blockDim.z != 1) return;
    unsigned lane = threadIdx.x & 31U;
    unsigned group = lane >> 2, thread = lane & 3U;
    unsigned long long tile = (unsigned long long)blockIdx.x * 64 +
                              (threadIdx.x >> 5) * 16;
    if (tile >= rows) return;
    float accum[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (unsigned long long base = 0; base < columns; base += 256) {
        const unsigned char *blocks[2];
        float d[2], minimum[2];
        unsigned scales[2][2], minima[2][2];
        #pragma unroll
        for (unsigned r = 0; r < 2; ++r) {
            unsigned long long row = tile + group + r * 8;
            /* Invalid tail rows read a valid dummy block, then contribute zero.
               No lane may leave before a warp-wide MMA instruction. */
            blocks[r] = weights + (row < rows ? row : 0) * stride + base / 256 * 144;
            const unsigned char *b = blocks[r];
            d[r] = row < rows ? nya_half_aligned(b) : 0.0f;
            minimum[r] = row < rows ? nya_half_aligned(b + 2) : 0.0f;
            const unsigned *metadata = (const unsigned *)(b + 4);
            unsigned a = metadata[0], c = metadata[1], e = metadata[2];
            scales[r][0] = a & 0x3f3f3f3fU;
            scales[r][1] = (e & 0x0f0f0f0fU) | ((a & 0xc0c0c0c0U) >> 2);
            minima[r][0] = c & 0x3f3f3f3fU;
            minima[r][1] = ((e >> 4) & 0x0f0f0f0fU) | ((c & 0xc0c0c0c0U) >> 2);
        }
        #pragma unroll
        for (unsigned g = 0; g < 8; ++g) {
            unsigned sc[2], mn[2];
            #pragma unroll
            for (unsigned r = 0; r < 2; ++r) {
                sc[r] = (scales[r][g / 4] >> ((g % 4) * 8)) & 255U;
                mn[r] = (minima[r][g / 4] >> ((g % 4) * 8)) & 255U;
            }
            #pragma unroll 1
            for (unsigned step = 0; step < 4; ++step) {
                unsigned ah[4], al[4], bh[2], bl[2];
                #pragma unroll
                for (unsigned i = 0; i < 4; ++i) {
                    unsigned r = i % 2, column = step * 8 + thread + (i / 2) * 4;
                    unsigned q = (blocks[r][16 + g / 2 * 32 + column] >> (g % 2 * 4)) & 15U;
                    float a = d[r] * (float)(sc[r] * q) - minimum[r] * (float)mn[r];
                    ah[i] = nya_q4_tf32_round(a);
                    al[i] = nya_q4_tf32_round(a - __uint_as_float(ah[i]));
                }
                #pragma unroll
                for (unsigned i = 0; i < 2; ++i) {
                    unsigned long long column = base + g * 32 + step * 8 + thread + i * 4;
                    float b = group < 4 ? input[column * 4 + group] : 0.0f;
                    bh[i] = nya_q4_tf32_round(b);
                    bl[i] = nya_q4_tf32_round(b - __uint_as_float(bh[i]));
                }
                nya_q4_tf32_mma(accum, ah, bl);
                nya_q4_tf32_mma(accum, al, bh);
                nya_q4_tf32_mma(accum, ah, bh);
            }
        }
    }
    #pragma unroll
    for (unsigned i = 0; i < 4; ++i) {
        unsigned long long row = tile + group + (i / 2) * 8;
        unsigned column = thread * 2 + i % 2;
        if (row < rows && column < 4)
            output[(unsigned long long)column * rows + row] = accum[i];
    }
}
