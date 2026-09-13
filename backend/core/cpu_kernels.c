#include "cpu_kernels.h"
#include <stdint.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

/* Target attributes keep advanced instructions out of baseline code. Unsupported
   compilers/architectures return NULL and retain the portable scalar kernels. */
#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
static float cpu_half(const unsigned char *p)
{
    uint32_t h = (uint32_t)p[0] | ((uint32_t)p[1] << 8), mantissa = h & 1023U;
    uint32_t bits, sign = (h & 32768U) << 16;
    int exponent = (int)((h >> 10) & 31U);
    if (!exponent && mantissa) { exponent = 1; while (!(mantissa & 1024U)) { mantissa <<= 1; --exponent; }
        bits = sign | ((uint32_t)(exponent + 112) << 23) | ((mantissa & 1023U) << 13); }
    else if (!exponent) bits = sign;
    else if (exponent == 31) bits = sign | 0x7f800000U | (mantissa << 13);
    else bits = sign | ((uint32_t)(exponent + 112) << 23) | (mantissa << 13);
    float value; memcpy(&value, &bits, 4); return value;
}
#define NYA_TARGET __attribute__((target("avx2,f16c")))
#define NYA_DOT_NAME cpu_avx2
#define NYA_DOT_IMPL cpu_avx2_impl
#define NYA_DECODE_NAME cpu_decode_avx2
#define NYA_GEMM_NAME cpu_gemm_avx2
#define NYA_ATTENTION_NAME cpu_attention_avx2
#define LANES 8
#define VEC __m256
#define IVEC __m256i
#define ZERO() _mm256_setzero_ps()
#define LOAD(p) _mm256_loadu_ps(p)
#define STORE(p,v) _mm256_storeu_ps(p,v)
#define SET(v) _mm256_set1_ps(v)
#define ADD(a,b) _mm256_add_ps(a,b)
#define SUB(a,b) _mm256_sub_ps(a,b)
#define MUL(a,b) _mm256_mul_ps(a,b)
#define ISET(v) _mm256_set1_epi32(v)
#define ISUB(a,b) _mm256_sub_epi32(a,b)
#define AND(a,b) _mm256_and_si256(a,b)
#define OR(a,b) _mm256_or_si256(a,b)
#define SHIFT(a,n) _mm256_srli_epi32(a,n)
#define VSHIFT(a,n) _mm256_srl_epi32(a,_mm_cvtsi32_si128(n))
#define LEFT(a,n) _mm256_slli_epi32(a,n)
#define FLOAT(v) _mm256_cvtepi32_ps(v)
#define UNSIGNED_BYTES(p) _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i *)(const void *)(p)))
#define SIGNED_BYTES(p) _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i *)(const void *)(p)))
#define HALF(p) _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(const void *)(p)))
#define BFLOAT(p) _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i *)(const void *)(p))),16))
#include "cpu_simd.inc"
#include "cpu_simd_undef.inc"
#define NYA_TARGET __attribute__((target("avx512f,avx512bw,avx2,f16c")))
#define NYA_DOT_NAME cpu_avx512
#define NYA_DOT_IMPL cpu_avx512_impl
#define NYA_DECODE_NAME cpu_decode_avx512
#define NYA_GEMM_NAME cpu_gemm_avx512
#define NYA_ATTENTION_NAME cpu_attention_avx512
#define LANES 16
#define VEC __m512
#define IVEC __m512i
#define ZERO() _mm512_setzero_ps()
#define LOAD(p) _mm512_loadu_ps(p)
#define STORE(p,v) _mm512_storeu_ps(p,v)
#define SET(v) _mm512_set1_ps(v)
#define ADD(a,b) _mm512_add_ps(a,b)
#define SUB(a,b) _mm512_sub_ps(a,b)
#define MUL(a,b) _mm512_mul_ps(a,b)
#define ISET(v) _mm512_set1_epi32(v)
#define ISUB(a,b) _mm512_sub_epi32(a,b)
#define AND(a,b) _mm512_and_si512(a,b)
#define OR(a,b) _mm512_or_si512(a,b)
#define SHIFT(a,n) _mm512_srli_epi32(a,n)
#define VSHIFT(a,n) _mm512_srl_epi32(a,_mm_cvtsi32_si128(n))
#define LEFT(a,n) _mm512_slli_epi32(a,n)
#define FLOAT(v) _mm512_cvtepi32_ps(v)
#define UNSIGNED_BYTES(p) _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i *)(const void *)(p)))
#define SIGNED_BYTES(p) _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i *)(const void *)(p)))
#define HALF(p) _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)(const void *)(p)))
#define BFLOAT(p) _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256((const __m256i *)(const void *)(p))),16))
#include "cpu_simd.inc"
#include "cpu_simd_undef.inc"
#endif

nya_cpu_dot_function nya_cpu_select_dot(const char **name)
{
    const char *selection = getenv("NYA_CPU_ISA");
    *name = "scalar";
    if (selection && !strcmp(selection, "scalar")) return NULL;
#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
    __builtin_cpu_init();
    if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("f16c")) {
        if ((!selection || !strcmp(selection, "avx512")) && __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw")) {
            *name = "avx512"; return cpu_avx512;
        }
        if (!selection || !strcmp(selection, "avx2")) { *name = "avx2"; return cpu_avx2; }
    }
#endif
    return NULL;
}

nya_cpu_decode_function nya_cpu_select_decode(void)
{
#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
    const char *name;
    nya_cpu_dot_function dot = nya_cpu_select_dot(&name);
    if (dot == cpu_avx2) return cpu_decode_avx2;
    if (dot == cpu_avx512) return cpu_decode_avx512;
#endif
    return NULL;
}

nya_cpu_gemm_function nya_cpu_select_gemm(void)
{
#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
    const char *name;
    nya_cpu_dot_function dot = nya_cpu_select_dot(&name);
    if (dot == cpu_avx2) return cpu_gemm_avx2;
    if (dot == cpu_avx512) return cpu_gemm_avx512;
#endif
    return NULL;
}
nya_cpu_attention_function nya_cpu_select_attention(void)
{
#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
    const char *name;
    nya_cpu_dot_function dot = nya_cpu_select_dot(&name);
    if (dot == cpu_avx2) return cpu_attention_avx2;
    if (dot == cpu_avx512) return cpu_attention_avx512;
#endif
    return NULL;
}
