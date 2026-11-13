/* Original Fyodor experimental packed Q4_K paired projection.
   Gemma tanh GELU uses the same expression as native vector kind 3.
   Exactly 3840 columns, 2400 bytes/row, 256 threads; opt-in runtime dispatch. */
extern "C" __global__ __launch_bounds__(256,4) void nya_gate_up_gemma_byte_3840(const unsigned char *gate,
    const unsigned char *up, const float *input, float *output,
    unsigned long long ignored_columns, unsigned long long ignored_stride, unsigned long long rows)
{
    const unsigned long long columns=3840, stride=2400;
    unsigned lane = threadIdx.x & 31;
    unsigned long long row = (unsigned long long)blockIdx.x * 8 + (threadIdx.x >> 5);
    if (row >= rows) return;
    float a = 0, z = 0;
    #pragma unroll 8
    for (unsigned long long base = 0; base < columns; base += 256) {
        const unsigned char *g = gate + row*stride + base/256*160;
        const unsigned char *u = up + row*stride + base/256*160;
        float gd=nya_half_aligned(g), gm=nya_half_aligned(g+2), ud=nya_half_aligned(u), um=nya_half_aligned(u+2);
        const unsigned *gs32=(const unsigned *)(g+4), *us32=(const unsigned *)(u+4);
        unsigned G[4]={gs32[0],gs32[1],gs32[2],gs32[3]}, U[4]={us32[0],us32[1],us32[2],us32[3]};
        #pragma unroll
        for (unsigned j=0; j<8; ++j) {
            unsigned scg=(G[j/4]>>((j%4)*8))&255U;
            unsigned mng=(G[2+j/4]>>((j%4)*8))&255U;
            unsigned scu=(U[j/4]>>((j%4)*8))&255U;
            unsigned mnu=(U[2+j/4]>>((j%4)*8))&255U;
            unsigned qg=(g[32+j/2*32+lane]>>(j%2*4))&15U;
            unsigned qu=(u[32+j/2*32+lane]>>(j%2*4))&15U;
            float x=input[base+j*32+lane];
            a += (gd*(float)(scg*qg)-gm*(float)mng)*x;
            z += (ud*(float)(scu*qu)-um*(float)mnu)*x;
        }
    }
    for (int step=16;step;step>>=1) {
        a += __shfl_down_sync(0xffffffffU,a,step);
        z += __shfl_down_sync(0xffffffffU,z,step);
    }
    if (!lane) output[row]=0.5f * a * (1 + tanhf(0.7978845608028654f * (a + 0.044715f * a*a*a))) * z;
}
