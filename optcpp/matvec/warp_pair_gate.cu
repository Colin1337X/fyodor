/* Original Fyodor diagnostic: separate gate/up warps, one block-local combine. */
extern "C" __global__ __launch_bounds__(256,4) void nya_gate_up_q4_w2(
    const unsigned char *gate, const unsigned char *up, const float *input,
    float *output, unsigned long long columns, unsigned long long stride,
    unsigned long long rows)
{
    unsigned lane=threadIdx.x&31, warp=threadIdx.x>>5;
    unsigned long long row=(unsigned long long)blockIdx.x*4+warp/2;
    const unsigned char *weights=(warp&1)?up:gate;
    float sum=0;
    __shared__ float partial[8];
    #pragma unroll 8
    for (unsigned long long base=0; row<rows && base<columns; base+=256) {
        const unsigned char *g=weights+row*stride+base/256*144;
        float d=nya_half_aligned(g), m=nya_half_aligned(g+2);
        const unsigned *s=(const unsigned *)(g+4);
        unsigned S[3]={s[0],s[1],s[2]};
        #pragma unroll
        for (unsigned j=0;j<8;++j) {
            unsigned shift=(j%4)*8;
            unsigned sc=j<4?(S[0]>>shift)&63U:((S[2]>>shift)&15U)|(((S[0]>>(shift+6))&3U)<<4);
            unsigned mn=j<4?(S[1]>>shift)&63U:((S[2]>>(shift+4))&15U)|(((S[1]>>(shift+6))&3U)<<4);
            unsigned q=(g[16+j/2*32+lane]>>(j%2*4))&15U;
            float x=input[base+j*32+lane];
            sum+=(d*(float)sc*(float)q-m*(float)mn)*x;
        }
    }
    for (int step=16;step;step>>=1) sum+=__shfl_down_sync(0xffffffffU,sum,step);
    if (!lane) partial[warp]=sum;
    __syncthreads();
    if (!lane && !(warp&1) && row<rows) output[row]=sum/(1+expf(-sum))*partial[warp+1];
}
