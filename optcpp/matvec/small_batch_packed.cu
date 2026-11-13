/* Original Fyodor experiment: reuse expanded byte scales and Q4 payloads
   across narrow verification batches. Compile after CUDA/matvec.cu. Input
   blocks use q4_byte_pack.h's 160-byte layout. Arithmetic matches the native
   small_batch_reuse Q4 path, including multiplication and reduction order. */
template<unsigned INPUTS>
__device__ __forceinline__ void nya_packed_reuse_impl(
    const unsigned char *weights, const float *input, float *output,
    unsigned long long columns, unsigned long long stride,
    unsigned rows, unsigned batch)
{
    unsigned lane=threadIdx.x&31;
    unsigned long long row=(unsigned long long)blockIdx.x*8+(threadIdx.x>>5);
    unsigned first=blockIdx.y*INPUTS;
    if(row>=rows || first>=batch)return;
    float sums[INPUTS]={0};
    for(unsigned long long base=0;base<columns;base+=256) {
        const unsigned char *b=weights+row*stride+base/256*160;
        float d=nya_half_aligned(b),minimum=nya_half_aligned(b+2);
        const unsigned *s=(const unsigned *)(b+4);
        unsigned S[4]={s[0],s[1],s[2],s[3]};
        #pragma unroll
        for(unsigned g=0;g<8;++g) {
            unsigned shift=(g%4)*8;
            unsigned sc=(S[g/4]>>shift)&255U,mn=(S[2+g/4]>>shift)&255U;
            unsigned q=(b[32+g/2*32+lane]>>(g%2*4))&15U;
            float weight=d*(float)sc*(float)q-minimum*(float)mn;
            #pragma unroll
            for(unsigned i=0;i<INPUTS;++i)if(first+i<batch)
                sums[i]+=weight*input[(unsigned long long)(first+i)*columns+base+g*32+lane];
        }
    }
    #pragma unroll
    for(unsigned i=0;i<INPUTS;++i) {
        float sum=sums[i];
        for(int step=16;step;step>>=1)sum+=__shfl_down_sync(0xffffffffU,sum,step);
        if(!lane && first+i<batch)output[(unsigned long long)(first+i)*rows+row]=sum;
    }
}
#define NYA_PACKED_REUSE(N) \
extern "C" __global__ void nya_packed_reuse_12_##N(const unsigned char *w, \
 const float *x,float *y,unsigned long long columns,unsigned long long stride, \
 unsigned rows,unsigned batch) {nya_packed_reuse_impl<N>(w,x,y,columns,stride,rows,batch);}
NYA_PACKED_REUSE(2) NYA_PACKED_REUSE(4)
#undef NYA_PACKED_REUSE
