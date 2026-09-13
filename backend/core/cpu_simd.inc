/* Instantiated for AVX2 and AVX-512. Each SIMD register dequantizes contiguous
   storage lanes and immediately consumes them in a dot product; no expanded
   matrix or heap scratch is needed. Byte loads tolerate unaligned GGUF blocks. */
NYA_TARGET static inline __attribute__((always_inline)) float NYA_DOT_IMPL(const unsigned char *w, const float *x, size_t columns, unsigned type, float *decoded)
{
    VEC sum = ZERO();
#define ACCUMULATE(v, index) do { VEC value_ = (v); if (decoded) STORE(decoded+(index),value_); else sum = ADD(sum,MUL(value_,LOAD(x+(index)))); } while (0)
    size_t i = 0;
    if (type == 0 || type == 1 || type == 30) {
        for (; i + LANES <= columns; i += LANES) {
            VEC weights;
            if (type == 0) weights = LOAD((const float *)(const void *)(w + i * 4));
            else if (type == 1) weights = HALF(w + i * 2);
            else weights = BFLOAT(w + i * 2);
            ACCUMULATE(weights, i);
        }
        float lanes[LANES]; STORE(lanes, sum);
        float out = 0; for (size_t j = 0; j < LANES; ++j) out += lanes[j];
        for (; i < columns; ++i) {
            float value;
            if (type == 0) memcpy(&value, w + i * 4, 4);
            else if (type == 1) value = cpu_half(w + i * 2);
            else { uint32_t bits = ((uint32_t)w[i*2] | ((uint32_t)w[i*2+1] << 8)) << 16; memcpy(&value, &bits, 4); }
            if (decoded) decoded[i] = value; else out += value * x[i];
        }
        return out;
    }
    size_t block = type == 2 || type == 8 ? 32 : 256;
    size_t bytes = type == 2 ? 18 : type == 8 ? 34 : type == 12 ? 144 : 210;
    for (size_t base = 0; base < columns; base += block, w += bytes) {
        if (type == 2 || type == 8) {
            VEC scale = SET(cpu_half(w));
            for (size_t j = 0; j < 32; j += LANES) {
                IVEC q;
                if (type == 8) q = SIGNED_BYTES(w + 2 + j);
                else { q = UNSIGNED_BYTES(w + 2 + j % 16); if (j >= 16) q = SHIFT(q, 4); q = ISUB(AND(q, ISET(15)), ISET(8)); }
                ACCUMULATE(MUL(scale,FLOAT(q)), base+j);
            }
        } else if (type == 12) {
            float d = cpu_half(w), minimum = cpu_half(w + 2);
            const unsigned char *scales = w + 4;
            for (size_t g = 0; g < 8; ++g) {
                unsigned sc = g < 4 ? scales[g] & 63U : (scales[g+4] & 15U) | ((unsigned)(scales[g-4] >> 6) << 4);
                unsigned mn = g < 4 ? scales[g+4] & 63U : (scales[g+4] >> 4) | ((unsigned)(scales[g] >> 6) << 4);
                VEC factor = SET(d * (float)sc), bias = SET(minimum * (float)mn);
                for (size_t j = 0; j < 32; j += LANES) {
                    IVEC q = UNSIGNED_BYTES(w + 16 + g/2*32 + j);
                    if (g % 2) q = SHIFT(q, 4);
                    VEC value = SUB(MUL(factor, FLOAT(AND(q, ISET(15)))), bias);
                    ACCUMULATE(value, base+g*32+j);
                }
            }
        } else {
            float d = cpu_half(w + 208);
            for (size_t g = 0; g < 16; ++g) {
                size_t half = g/8, quarter = g%8/2, start = g%2*16;
                int8_t sc; memcpy(&sc, w + 192 + g, 1);
                VEC factor = SET(d * (float)sc);
                for (size_t j = 0; j < 16; j += LANES) {
                    IVEC lo = UNSIGNED_BYTES(w + half*64 + quarter%2*32 + start+j);
                    IVEC hi = UNSIGNED_BYTES(w + 128 + half*32 + start+j);
                    if (quarter >= 2) lo = SHIFT(lo, 4);
                    /* Variable shifts use their scalar-count intrinsic form. */
                    hi = VSHIFT(hi, (int)(quarter * 2));
                    IVEC q = ISUB(OR(AND(lo, ISET(15)), LEFT(AND(hi, ISET(3)), 4)), ISET(32));
                    ACCUMULATE(MUL(factor,FLOAT(q)), base+g*16+j);
                }
            }
        }
    }
    float lanes[LANES]; STORE(lanes, sum);
    float out = 0; for (size_t j = 0; j < LANES; ++j) out += lanes[j];
    return out;
}

#undef ACCUMULATE
NYA_TARGET static float NYA_DOT_NAME(const unsigned char *w, const float *x, size_t n, unsigned type)
{ return NYA_DOT_IMPL(w,x,n,type,NULL); }
NYA_TARGET static void NYA_DECODE_NAME(const unsigned char *w, float *out, size_t n, unsigned type)
{ (void)NYA_DOT_IMPL(w,NULL,n,type,out); }

/* One independent query/head. Scores retain the reference stable softmax;
   contiguous head lanes use SIMD for both Q*K and the weighted value sum.
   Keeping each output vector in registers avoids repeatedly storing it for
   every context position. The caller supplies private per-worker score scratch. */
NYA_TARGET static int NYA_ATTENTION_NAME(const float *q, const float *keys, const float *values,
    float *out, float *scores, size_t width, size_t stride, size_t length, float scale)
{
    float maximum = -INFINITY;
    for (size_t t = 0; t < length; ++t) {
        scores[t] = NYA_DOT_NAME((const unsigned char *)(keys+t*stride),q,width,0)*scale;
        if (scores[t] > maximum) maximum = scores[t];
    }
    float sum = 0;
    for (size_t t = 0; t < length; ++t) { scores[t] = expf(scores[t]-maximum); sum += scores[t]; }
    if (!(sum > 0) || !isfinite(sum)) return -1;
    for (size_t t = 0; t < length; ++t) scores[t] /= sum;
    size_t j = 0;
    for (; j+LANES <= width; j += LANES) {
        VEC value = ZERO();
        for (size_t t = 0; t < length; ++t) value = ADD(value,MUL(SET(scores[t]),LOAD(values+t*stride+j)));
        STORE(out+j,value);
    }
    for (; j < width; ++j) {
        float value = 0;
        for (size_t t = 0; t < length; ++t) value += scores[t]*values[t*stride+j];
        out[j] = value;
    }
    return 0;
}

/* Two output rows share each vector of prompt activations. SIMD lanes are
   tokens, so input is packed once per projection into K-major order. Two
   pairs of independent accumulation chains improve instruction overlap and limit the
   rounding depth. No quantized model-wide F32 expansion is retained. */
NYA_TARGET static void NYA_GEMM_NAME(const float *w, const float *x, float *out,
    size_t columns, size_t rows, size_t batch, size_t packed_stride, size_t output_stride)
{
    for (size_t token = 0; token < batch; token += LANES) {
        VEC sums[2][4];
        for (size_t r = 0; r < 2; ++r) for (size_t p = 0; p < 4; ++p) sums[r][p] = ZERO();
        size_t k = 0;
        for (; k+3 < columns; k += 4) {
            for (size_t p = 0; p < 4; ++p) {
                VEC a = LOAD(x+(k+p)*packed_stride+token);
                for (size_t r = 0; r < 2; ++r) sums[r][p] = ADD(sums[r][p],MUL(SET(w[r*columns+k+p]),a));
            }
        }
        for (; k < columns; ++k) {
            VEC a = LOAD(x+k*packed_stride+token);
            for (size_t r = 0; r < 2; ++r) sums[r][0] = ADD(sums[r][0],MUL(SET(w[r*columns+k]),a));
        }
        for (size_t r = 0; r < rows; ++r) {
            float lanes[LANES]; STORE(lanes,ADD(ADD(sums[r][0],sums[r][1]),ADD(sums[r][2],sums[r][3])));
            for (size_t t = 0; t < LANES && token+t < batch; ++t) out[(token+t)*output_stride+r] = lanes[t];
        }
    }
}
