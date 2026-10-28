/* Original Fyodor diagnostic: 160-byte aligned Q4 blocks with byte scales/minima. */
extern "C" __global__ __launch_bounds__(256,4) void nya_gate_up_byte_scales_256(const unsigned char *gate,
    const unsigned char *up, const float *input, float *output,
    unsigned long long ignored_columns, unsigned long long ignored_stride, unsigned long long rows)
{
    const unsigned long long columns=256, stride=160;
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
    if (!lane) output[row]=a/(1+expf(-a))*z;
}
extern "C" __global__ __launch_bounds__(256,4) void nya_gate_up_byte_scales_768(const unsigned char *gate,
    const unsigned char *up, const float *input, float *output,
    unsigned long long ignored_columns, unsigned long long ignored_stride, unsigned long long rows)
{
    const unsigned long long columns=768, stride=480;
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
    if (!lane) output[row]=a/(1+expf(-a))*z;
}
extern "C" __global__ __launch_bounds__(256,4) void nya_gate_up_byte_scales_2048(const unsigned char *gate,
    const unsigned char *up, const float *input, float *output,
    unsigned long long ignored_columns, unsigned long long ignored_stride, unsigned long long rows)
{
    const unsigned long long columns=2048, stride=1280;
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
    if (!lane) output[row]=a/(1+expf(-a))*z;
}
extern "C" __global__ __launch_bounds__(256,4) void nya_gate_up_byte_scales_2304(const unsigned char *gate,
    const unsigned char *up, const float *input, float *output,
    unsigned long long ignored_columns, unsigned long long ignored_stride, unsigned long long rows)
{
    const unsigned long long columns=2304, stride=1440;
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
    if (!lane) output[row]=a/(1+expf(-a))*z;
}
extern "C" __global__ __launch_bounds__(256,4) void nya_gate_up_byte_scales_4096(const unsigned char *gate,
    const unsigned char *up, const float *input, float *output,
    unsigned long long ignored_columns, unsigned long long ignored_stride, unsigned long long rows)
{
    const unsigned long long columns=4096, stride=2560;
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
    if (!lane) output[row]=a/(1+expf(-a))*z;
}
extern "C" __global__ __launch_bounds__(256,4) void nya_gate_up_byte_scales_8192(const unsigned char *gate,
    const unsigned char *up, const float *input, float *output,
    unsigned long long ignored_columns, unsigned long long ignored_stride, unsigned long long rows)
{
    const unsigned long long columns=8192, stride=5120;
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
    if (!lane) output[row]=a/(1+expf(-a))*z;
}
