#include <immintrin.h>
#include <stdio.h>
__attribute__((target("avx512f"),noinline)) static float sum(const float *p) {
#ifdef EXPLICIT_ALIGN
    _Alignas(64)
#endif
    float lanes[16];
    _mm512_storeu_ps(lanes, _mm512_loadu_ps(p));
#ifdef CHECK_REDZONE
    volatile int beyond = 16;
    lanes[beyond] = 1;
#endif
    float s=0; for(int i=0;i<16;i++) s+=lanes[i];
    return s;
}
int main(void) {float x[16]; for(int i=0;i<16;i++)x[i]=(float)i; printf("%g\n",sum(x)); return 0;}
