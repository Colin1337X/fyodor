#include "llm_internal.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

/* Arithmetic oracle, independent of the production bit-shift conversion. All
   finite binary16 numbers are exactly representable as binary32. Exercise every
   encoding and every adjacent positive pair's tie and neighboring float values. */
static float half_value(unsigned h)
{
    unsigned exponent = (h >> 10) & 31U, fraction = h & 1023U;
    float value = exponent ? ldexpf((float)(1024U+fraction), (int)exponent-25) : ldexpf((float)fraction,-24);
    return h & 32768U ? -value : value;
}

static int same(float a, float b)
{
    uint32_t aa, bb;
    memcpy(&aa,&a,sizeof(aa)); memcpy(&bb,&b,sizeof(bb));
    return aa == bb;
}

int main(void)
{
    for (unsigned h = 0; h < 65536; ++h) {
        if (((h >> 10) & 31U) == 31U) continue;
        float value = half_value(h);
        if (!same(nya_llm_round_f16(value),value)) { fprintf(stderr,"half encoding %u\n",h); return 1; }
    }
    for (unsigned h = 0; h < 0x7bffU; ++h) {
        float a = half_value(h), b = half_value(h+1), midpoint = (a+b)*0.5f;
        float expected = h & 1U ? b : a;
        for (unsigned negative = 0; negative < 2; ++negative) {
            float sign = negative ? -1.0f : 1.0f;
            if (!same(nya_llm_round_f16(sign*midpoint),sign*expected) ||
                !same(nya_llm_round_f16(sign*nextafterf(midpoint,0)),sign*a) ||
                !same(nya_llm_round_f16(sign*nextafterf(midpoint,INFINITY)),sign*b)) {
                fprintf(stderr,"half rounding boundary %u sign %u\n",h,negative); return 1;
            }
        }
    }
    if (!isinf(nya_llm_round_f16(65520.0f)) || nya_llm_round_f16(nextafterf(65520.0f,0)) != 65504.0f ||
        !same(nya_llm_round_f16(-0.0f),-0.0f) || !isnan(nya_llm_round_f16(NAN)) ||
        nya_llm_round_f16(INFINITY) != INFINITY || nya_llm_round_f16(-INFINITY) != -INFINITY) return 1;
    return 0;
}
