/* Exhaustive finite-F16 coefficient-product check, original Fyodor diagnostic. */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
int main(void) {
    unsigned long long checked=0;
    for (unsigned h=0;h<65536;h++) {
        unsigned exponent=(h>>10)&31, mantissa=h&1023;
        if (exponent==31) continue;
        float d=exponent?ldexpf((float)(1024+mantissa),(int)exponent-25):ldexpf((float)mantissa,-24);
        if (h&32768) d=-d;
        for (int s=-128;s<128;s++) for (int q=-32;q<32;q++) {
            volatile float intermediate=d*(float)s;
            float product=(float)(s*q); uint32_t bits; memcpy(&bits,&product,4);
            bits |= ((uint32_t)s ^ (uint32_t)q) & UINT32_C(0x80000000); memcpy(&product,&bits,4);
            float before=intermediate*(float)q,after=d*product;
            if (memcmp(&before,&after,sizeof(float))) {printf("mismatch %u %d %d\n",h,s,q);return 1;}
            checked++;
        }
    }
    printf("{\"finite_coefficients_checked\":%llu,\"bit_exact\":true}\n",checked);
    return 0;
}
