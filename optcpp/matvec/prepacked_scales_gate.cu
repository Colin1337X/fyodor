/* Original Fyodor diagnostic. 192-byte blocks: 8 float scales, 8 float minima, 128 payload bytes. */
extern "C" __global__ __launch_bounds__(256,4) void nya_gate_up_q4_w2(const unsigned char *gate,
    const unsigned char *up, const float *input, float *output,
    unsigned long long columns, unsigned long long stride, unsigned long long rows)
{
    unsigned lane = threadIdx.x & 31;
    unsigned long long row = (unsigned long long)blockIdx.x * 8 + (threadIdx.x >> 5);
    if (row >= rows) return;
    float a = 0, z = 0;
    #pragma unroll 8
    for (unsigned long long base = 0; base < columns; base += 256) {
        const unsigned char *g = gate + row*(stride/144*192) + base/256*192;
        const unsigned char *u = up + row*(stride/144*192) + base/256*192;
        #pragma unroll
        for (unsigned j=0; j<8; ++j) {
            unsigned qg=(g[64+j/2*32+lane]>>(j%2*4))&15U;
            unsigned qu=(u[64+j/2*32+lane]>>(j%2*4))&15U;
            float x=input[base+j*32+lane];
            a += (((const float *)g)[j]*(float)qg-((const float *)g)[8+j])*x;
            z += (((const float *)u)[j]*(float)qu-((const float *)u)[8+j])*x;
        }
    }
    for (int step=16;step;step>>=1) {
        a += __shfl_down_sync(0xffffffffU,a,step);
        z += __shfl_down_sync(0xffffffffU,z,step);
    }
    if (!lane) output[row]=a/(1+expf(-a))*z;
}
