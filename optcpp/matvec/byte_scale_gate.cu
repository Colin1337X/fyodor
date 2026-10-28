/* Original Fyodor diagnostic: 148-byte Q4 blocks with byte scales/minima. */
extern "C" __global__ __launch_bounds__(256,4) void nya_gate_up_byte_scales_256(const unsigned char *gate,
    const unsigned char *up, const float *input, float *output,
    unsigned long long ignored_columns, unsigned long long ignored_stride, unsigned long long rows)
{
    const unsigned long long columns=256, stride=148;
    unsigned lane = threadIdx.x & 31;
    unsigned long long row = (unsigned long long)blockIdx.x * 8 + (threadIdx.x >> 5);
    if (row >= rows) return;
    float a = 0, z = 0;
    #pragma unroll 8
    for (unsigned long long base = 0; base < columns; base += 256) {
        const unsigned char *g = gate + row*stride + base/256*148;
        const unsigned char *u = up + row*stride + base/256*148;
        float gd=nya_half_aligned(g), gm=nya_half_aligned(g+2), ud=nya_half_aligned(u), um=nya_half_aligned(u+2);
        #pragma unroll
        for (unsigned j=0; j<8; ++j) {
            unsigned scg=g[4+j];
            unsigned mng=g[12+j];
            unsigned scu=u[4+j];
            unsigned mnu=u[12+j];
            unsigned qg=(g[20+j/2*32+lane]>>(j%2*4))&15U;
            unsigned qu=(u[20+j/2*32+lane]>>(j%2*4))&15U;
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
    const unsigned long long columns=768, stride=444;
    unsigned lane = threadIdx.x & 31;
    unsigned long long row = (unsigned long long)blockIdx.x * 8 + (threadIdx.x >> 5);
    if (row >= rows) return;
    float a = 0, z = 0;
    #pragma unroll 8
    for (unsigned long long base = 0; base < columns; base += 256) {
        const unsigned char *g = gate + row*stride + base/256*148;
        const unsigned char *u = up + row*stride + base/256*148;
        float gd=nya_half_aligned(g), gm=nya_half_aligned(g+2), ud=nya_half_aligned(u), um=nya_half_aligned(u+2);
        #pragma unroll
        for (unsigned j=0; j<8; ++j) {
            unsigned scg=g[4+j];
            unsigned mng=g[12+j];
            unsigned scu=u[4+j];
            unsigned mnu=u[12+j];
            unsigned qg=(g[20+j/2*32+lane]>>(j%2*4))&15U;
            unsigned qu=(u[20+j/2*32+lane]>>(j%2*4))&15U;
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
    const unsigned long long columns=2048, stride=1184;
    unsigned lane = threadIdx.x & 31;
    unsigned long long row = (unsigned long long)blockIdx.x * 8 + (threadIdx.x >> 5);
    if (row >= rows) return;
    float a = 0, z = 0;
    #pragma unroll 8
    for (unsigned long long base = 0; base < columns; base += 256) {
        const unsigned char *g = gate + row*stride + base/256*148;
        const unsigned char *u = up + row*stride + base/256*148;
        float gd=nya_half_aligned(g), gm=nya_half_aligned(g+2), ud=nya_half_aligned(u), um=nya_half_aligned(u+2);
        #pragma unroll
        for (unsigned j=0; j<8; ++j) {
            unsigned scg=g[4+j];
            unsigned mng=g[12+j];
            unsigned scu=u[4+j];
            unsigned mnu=u[12+j];
            unsigned qg=(g[20+j/2*32+lane]>>(j%2*4))&15U;
            unsigned qu=(u[20+j/2*32+lane]>>(j%2*4))&15U;
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
    const unsigned long long columns=2304, stride=1332;
    unsigned lane = threadIdx.x & 31;
    unsigned long long row = (unsigned long long)blockIdx.x * 8 + (threadIdx.x >> 5);
    if (row >= rows) return;
    float a = 0, z = 0;
    #pragma unroll 8
    for (unsigned long long base = 0; base < columns; base += 256) {
        const unsigned char *g = gate + row*stride + base/256*148;
        const unsigned char *u = up + row*stride + base/256*148;
        float gd=nya_half_aligned(g), gm=nya_half_aligned(g+2), ud=nya_half_aligned(u), um=nya_half_aligned(u+2);
        #pragma unroll
        for (unsigned j=0; j<8; ++j) {
            unsigned scg=g[4+j];
            unsigned mng=g[12+j];
            unsigned scu=u[4+j];
            unsigned mnu=u[12+j];
            unsigned qg=(g[20+j/2*32+lane]>>(j%2*4))&15U;
            unsigned qu=(u[20+j/2*32+lane]>>(j%2*4))&15U;
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
    const unsigned long long columns=4096, stride=2368;
    unsigned lane = threadIdx.x & 31;
    unsigned long long row = (unsigned long long)blockIdx.x * 8 + (threadIdx.x >> 5);
    if (row >= rows) return;
    float a = 0, z = 0;
    #pragma unroll 8
    for (unsigned long long base = 0; base < columns; base += 256) {
        const unsigned char *g = gate + row*stride + base/256*148;
        const unsigned char *u = up + row*stride + base/256*148;
        float gd=nya_half_aligned(g), gm=nya_half_aligned(g+2), ud=nya_half_aligned(u), um=nya_half_aligned(u+2);
        #pragma unroll
        for (unsigned j=0; j<8; ++j) {
            unsigned scg=g[4+j];
            unsigned mng=g[12+j];
            unsigned scu=u[4+j];
            unsigned mnu=u[12+j];
            unsigned qg=(g[20+j/2*32+lane]>>(j%2*4))&15U;
            unsigned qu=(u[20+j/2*32+lane]>>(j%2*4))&15U;
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
    const unsigned long long columns=8192, stride=4736;
    unsigned lane = threadIdx.x & 31;
    unsigned long long row = (unsigned long long)blockIdx.x * 8 + (threadIdx.x >> 5);
    if (row >= rows) return;
    float a = 0, z = 0;
    #pragma unroll 8
    for (unsigned long long base = 0; base < columns; base += 256) {
        const unsigned char *g = gate + row*stride + base/256*148;
        const unsigned char *u = up + row*stride + base/256*148;
        float gd=nya_half_aligned(g), gm=nya_half_aligned(g+2), ud=nya_half_aligned(u), um=nya_half_aligned(u+2);
        #pragma unroll
        for (unsigned j=0; j<8; ++j) {
            unsigned scg=g[4+j];
            unsigned mng=g[12+j];
            unsigned scu=u[4+j];
            unsigned mnu=u[12+j];
            unsigned qg=(g[20+j/2*32+lane]>>(j%2*4))&15U;
            unsigned qu=(u[20+j/2*32+lane]>>(j%2*4))&15U;
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
