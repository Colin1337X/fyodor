/* Original Fyodor experiment: reuse each input value across adjacent Q6 rows.
   Preserves the signed scale-product and per-output accumulation order. */
template<unsigned R> __device__ __forceinline__ void nya_q6_rows(
    const unsigned char *weights,const float *input,float *output,
    unsigned long long columns,unsigned long long stride,unsigned long long rows)
{
    unsigned lane=threadIdx.x&31;
    unsigned long long row=((unsigned long long)blockIdx.x*8+(threadIdx.x>>5))*R;
    if(row>=rows) return;
    float sums[R]={0};
    for(unsigned long long base=0;base<columns;base+=256) {
        const unsigned char *blocks[R];float scales[R];
        #pragma unroll
        for(unsigned r=0;r<R;++r) {
            blocks[r]=row+r<rows ? weights+(row+r)*stride+base/256*210 : weights;
            scales[r]=row+r<rows ? nya_half_aligned(blocks[r]+208) : 0;
        }
        #pragma unroll
        for(unsigned g=0;g<8;++g) {
            float x=input[base+g*32+lane];
            unsigned half=g/4,quarter=g%4;
            #pragma unroll
            for(unsigned r=0;r<R;++r) if(row+r<rows) {
                const unsigned char *b=blocks[r];
                unsigned lo=b[half*64+quarter%2*32+lane],hi=b[128+half*32+lane];
                int q=(int)(((lo>>(quarter/2*4))&15U)|(((hi>>(quarter*2))&3U)<<4))-32;
                int sc=((const signed char *)(b+192))[g*2+lane/16];
                float product=__int_as_float(__float_as_int((float)(sc*q))|((sc^q)&(-2147483647-1)));
                sums[r]+=scales[r]*product*x;
            }
        }
    }
    #pragma unroll
    for(unsigned r=0;r<R;++r) {
        float sum=sums[r];
        for(int step=16;step;step>>=1) sum+=__shfl_down_sync(0xffffffffU,sum,step);
        if(!lane && row+r<rows) output[row+r]=sum;
    }
}
#define NYA_Q6_ROWS(R) extern "C" __global__ void nya_q6_rows_##R(const unsigned char *w,const float *x,float *y,unsigned long long c,unsigned long long s,unsigned long long r) { nya_q6_rows<R>(w,x,y,c,s,r); }
NYA_Q6_ROWS(1)
NYA_Q6_ROWS(2)
NYA_Q6_ROWS(4)
NYA_Q6_ROWS(8)
#undef NYA_Q6_ROWS
