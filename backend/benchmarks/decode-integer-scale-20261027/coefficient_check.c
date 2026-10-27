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
        for (unsigned s=0;s<64;s++) for (unsigned q=0;q<16;q++) {
            volatile float intermediate=d*(float)s;
            float before=intermediate*(float)q,after=d*(float)(s*q);
            if (memcmp(&before,&after,sizeof(float))) {printf("mismatch %u %u %u\n",h,s,q);return 1;}
            checked++;
        }
    }
    printf("{\"finite_coefficients_checked\":%llu,\"bit_exact\":true}\n",checked);
    return 0;
}
