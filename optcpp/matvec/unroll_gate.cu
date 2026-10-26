/* Original Fyodor diagnostic: preserve ordered arithmetic, vary outer unrolling. */
extern "C" __global__ __launch_bounds__(256,4) void nya_gate_up_q4_w1(const unsigned char *gate,
    const unsigned char *up, const float *input, float *output,
    unsigned long long columns, unsigned long long stride, unsigned long long rows)
{
    unsigned lane = threadIdx.x & 31;
    unsigned long long row = (unsigned long long)blockIdx.x * 8 + (threadIdx.x >> 5);
    if (row >= rows) return;
    float a = 0, z = 0;
    #pragma unroll 1
    for (unsigned long long base = 0; base < columns; base += 256) {
        const unsigned char *g = gate + row*stride + base/256*144;
        const unsigned char *u = up + row*stride + base/256*144;
        float gd=nya_half_aligned(g), gm=nya_half_aligned(g+2), ud=nya_half_aligned(u), um=nya_half_aligned(u+2);
        /* Device allocations and 144-byte Q4_K blocks are word-aligned.
           Load each metadata word once; unrolling makes extraction constant. */
        const unsigned *gs32=(const unsigned *)(g+4), *us32=(const unsigned *)(u+4);
        unsigned G[3]={gs32[0],gs32[1],gs32[2]}, U[3]={us32[0],us32[1],us32[2]};
        #pragma unroll
        for (unsigned j=0; j<8; ++j) {
            unsigned shift=(j%4)*8;
            unsigned scg=j<4 ? (G[0]>>shift)&63U : ((G[2]>>shift)&15U) | (((G[0]>>(shift+6))&3U)<<4);
            unsigned mng=j<4 ? (G[1]>>shift)&63U : ((G[2]>>(shift+4))&15U) | (((G[1]>>(shift+6))&3U)<<4);
            unsigned scu=j<4 ? (U[0]>>shift)&63U : ((U[2]>>shift)&15U) | (((U[0]>>(shift+6))&3U)<<4);
            unsigned mnu=j<4 ? (U[1]>>shift)&63U : ((U[2]>>(shift+4))&15U) | (((U[1]>>(shift+6))&3U)<<4);
            unsigned qg=(g[16+j/2*32+lane]>>(j%2*4))&15U;
            unsigned qu=(u[16+j/2*32+lane]>>(j%2*4))&15U;
            float x=input[base+j*32+lane];
            a += (gd*(float)scg*(float)qg-gm*(float)mng)*x;
            z += (ud*(float)scu*(float)qu-um*(float)mnu)*x;
        }
    }
    for (int step=16;step;step>>=1) {
        a += __shfl_down_sync(0xffffffffU,a,step);
        z += __shfl_down_sync(0xffffffffU,z,step);
    }
    if (!lane) output[row]=a/(1+expf(-a))*z;
}


extern "C" __global__ __launch_bounds__(256,4) void nya_gate_up_q4_w2(const unsigned char *gate,
    const unsigned char *up, const float *input, float *output,
    unsigned long long columns, unsigned long long stride, unsigned long long rows)
{
    unsigned lane = threadIdx.x & 31;
    unsigned long long row = (unsigned long long)blockIdx.x * 8 + (threadIdx.x >> 5);
    if (row >= rows) return;
    float a = 0, z = 0;
    #pragma unroll 2
    for (unsigned long long base = 0; base < columns; base += 256) {
        const unsigned char *g = gate + row*stride + base/256*144;
        const unsigned char *u = up + row*stride + base/256*144;
        float gd=nya_half_aligned(g), gm=nya_half_aligned(g+2), ud=nya_half_aligned(u), um=nya_half_aligned(u+2);
        /* Device allocations and 144-byte Q4_K blocks are word-aligned.
           Load each metadata word once; unrolling makes extraction constant. */
        const unsigned *gs32=(const unsigned *)(g+4), *us32=(const unsigned *)(u+4);
        unsigned G[3]={gs32[0],gs32[1],gs32[2]}, U[3]={us32[0],us32[1],us32[2]};
        #pragma unroll
        for (unsigned j=0; j<8; ++j) {
            unsigned shift=(j%4)*8;
            unsigned scg=j<4 ? (G[0]>>shift)&63U : ((G[2]>>shift)&15U) | (((G[0]>>(shift+6))&3U)<<4);
            unsigned mng=j<4 ? (G[1]>>shift)&63U : ((G[2]>>(shift+4))&15U) | (((G[1]>>(shift+6))&3U)<<4);
            unsigned scu=j<4 ? (U[0]>>shift)&63U : ((U[2]>>shift)&15U) | (((U[0]>>(shift+6))&3U)<<4);
            unsigned mnu=j<4 ? (U[1]>>shift)&63U : ((U[2]>>(shift+4))&15U) | (((U[1]>>(shift+6))&3U)<<4);
            unsigned qg=(g[16+j/2*32+lane]>>(j%2*4))&15U;
            unsigned qu=(u[16+j/2*32+lane]>>(j%2*4))&15U;
            float x=input[base+j*32+lane];
            a += (gd*(float)scg*(float)qg-gm*(float)mng)*x;
            z += (ud*(float)scu*(float)qu-um*(float)mnu)*x;
        }
    }
    for (int step=16;step;step>>=1) {
        a += __shfl_down_sync(0xffffffffU,a,step);
        z += __shfl_down_sync(0xffffffffU,z,step);
    }
    if (!lane) output[row]=a/(1+expf(-a))*z;
}


extern "C" __global__ __launch_bounds__(256,4) void nya_gate_up_q4_w4(const unsigned char *gate,
    const unsigned char *up, const float *input, float *output,
    unsigned long long columns, unsigned long long stride, unsigned long long rows)
{
    unsigned lane = threadIdx.x & 31;
    unsigned long long row = (unsigned long long)blockIdx.x * 8 + (threadIdx.x >> 5);
    if (row >= rows) return;
    float a = 0, z = 0;
    #pragma unroll 4
    for (unsigned long long base = 0; base < columns; base += 256) {
        const unsigned char *g = gate + row*stride + base/256*144;
        const unsigned char *u = up + row*stride + base/256*144;
        float gd=nya_half_aligned(g), gm=nya_half_aligned(g+2), ud=nya_half_aligned(u), um=nya_half_aligned(u+2);
        /* Device allocations and 144-byte Q4_K blocks are word-aligned.
           Load each metadata word once; unrolling makes extraction constant. */
        const unsigned *gs32=(const unsigned *)(g+4), *us32=(const unsigned *)(u+4);
        unsigned G[3]={gs32[0],gs32[1],gs32[2]}, U[3]={us32[0],us32[1],us32[2]};
        #pragma unroll
        for (unsigned j=0; j<8; ++j) {
            unsigned shift=(j%4)*8;
            unsigned scg=j<4 ? (G[0]>>shift)&63U : ((G[2]>>shift)&15U) | (((G[0]>>(shift+6))&3U)<<4);
            unsigned mng=j<4 ? (G[1]>>shift)&63U : ((G[2]>>(shift+4))&15U) | (((G[1]>>(shift+6))&3U)<<4);
            unsigned scu=j<4 ? (U[0]>>shift)&63U : ((U[2]>>shift)&15U) | (((U[0]>>(shift+6))&3U)<<4);
            unsigned mnu=j<4 ? (U[1]>>shift)&63U : ((U[2]>>(shift+4))&15U) | (((U[1]>>(shift+6))&3U)<<4);
            unsigned qg=(g[16+j/2*32+lane]>>(j%2*4))&15U;
            unsigned qu=(u[16+j/2*32+lane]>>(j%2*4))&15U;
            float x=input[base+j*32+lane];
            a += (gd*(float)scg*(float)qg-gm*(float)mng)*x;
            z += (ud*(float)scu*(float)qu-um*(float)mnu)*x;
        }
    }
    for (int step=16;step;step>>=1) {
        a += __shfl_down_sync(0xffffffffU,a,step);
        z += __shfl_down_sync(0xffffffffU,z,step);
    }
    if (!lane) output[row]=a/(1+expf(-a))*z;
}


extern "C" __global__ __launch_bounds__(256,4) void nya_gate_up_q4_w8(const unsigned char *gate,
    const unsigned char *up, const float *input, float *output,
    unsigned long long columns, unsigned long long stride, unsigned long long rows)
{
    unsigned lane = threadIdx.x & 31;
    unsigned long long row = (unsigned long long)blockIdx.x * 8 + (threadIdx.x >> 5);
    if (row >= rows) return;
    float a = 0, z = 0;
    #pragma unroll 8
    for (unsigned long long base = 0; base < columns; base += 256) {
        const unsigned char *g = gate + row*stride + base/256*144;
        const unsigned char *u = up + row*stride + base/256*144;
        float gd=nya_half_aligned(g), gm=nya_half_aligned(g+2), ud=nya_half_aligned(u), um=nya_half_aligned(u+2);
        /* Device allocations and 144-byte Q4_K blocks are word-aligned.
           Load each metadata word once; unrolling makes extraction constant. */
        const unsigned *gs32=(const unsigned *)(g+4), *us32=(const unsigned *)(u+4);
        unsigned G[3]={gs32[0],gs32[1],gs32[2]}, U[3]={us32[0],us32[1],us32[2]};
        #pragma unroll
        for (unsigned j=0; j<8; ++j) {
            unsigned shift=(j%4)*8;
            unsigned scg=j<4 ? (G[0]>>shift)&63U : ((G[2]>>shift)&15U) | (((G[0]>>(shift+6))&3U)<<4);
            unsigned mng=j<4 ? (G[1]>>shift)&63U : ((G[2]>>(shift+4))&15U) | (((G[1]>>(shift+6))&3U)<<4);
            unsigned scu=j<4 ? (U[0]>>shift)&63U : ((U[2]>>shift)&15U) | (((U[0]>>(shift+6))&3U)<<4);
            unsigned mnu=j<4 ? (U[1]>>shift)&63U : ((U[2]>>(shift+4))&15U) | (((U[1]>>(shift+6))&3U)<<4);
            unsigned qg=(g[16+j/2*32+lane]>>(j%2*4))&15U;
            unsigned qu=(u[16+j/2*32+lane]>>(j%2*4))&15U;
            float x=input[base+j*32+lane];
            a += (gd*(float)scg*(float)qg-gm*(float)mng)*x;
            z += (ud*(float)scu*(float)qu-um*(float)mnu)*x;
        }
    }
    for (int step=16;step;step>>=1) {
        a += __shfl_down_sync(0xffffffffU,a,step);
        z += __shfl_down_sync(0xffffffffU,z,step);
    }
    if (!lane) output[row]=a/(1+expf(-a))*z;
}
