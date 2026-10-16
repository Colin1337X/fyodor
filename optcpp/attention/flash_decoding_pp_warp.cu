/* Original CUDA specialization of unified-scale attention. Eight independent
   warp partitions stream coalesced K/V, retaining unnormalized sums in registers.
   Experimental: not selected by the runtime until separately validated. */
extern "C" __global__ void nya_fdpp_partial_warp(const float *q,
    const float *keys, const float *values, float *parts, unsigned width,
    unsigned heads, unsigned kvheads, unsigned window, unsigned position,
    float scale, const unsigned *request, unsigned partitions)
{
    unsigned h=blockIdx.x, part=blockIdx.y, lane=threadIdx.x&31, warp=threadIdx.x>>5;
    unsigned end=request?request[1]:position;
    unsigned first=window && end>=window ? end-window+1 : 0;
    unsigned long long length=(unsigned long long)end-first+1;
    unsigned long long begin=first+length*part/partitions;
    unsigned long long stop=first+length*(part+1)/partitions;
    unsigned kh=h/(heads/kvheads);
    unsigned long long stride=(unsigned long long)kvheads*width;
    q+=(unsigned long long)h*width;
    parts+=((unsigned long long)h*partitions+part)*(width+2);
    float query[8], numerator[8];
    #pragma unroll
    for (unsigned j=0;j<8;++j) {
        query[j]=lane+32*j<width?q[lane+32*j]:0.0f;
        numerator[j]=0.0f;
    }
    float denominator=0.0f;
    unsigned bad=0;
    for (unsigned long long token=begin+warp;token<stop;token+=8) {
        float score=0.0f;
        #pragma unroll
        for (unsigned j=0;j<8;++j)
            if (lane+32*j<width) score+=query[j]*keys[token*stride+(unsigned long long)kh*width+lane+32*j];
        for (unsigned step=16;step;step>>=1) score+=__shfl_down_sync(0xffffffffU,score,step);
        score=__shfl_sync(0xffffffffU,score,0)*scale;
        if (!(score>=-16.0f && score<=16.0f)) {bad=1;break;}
        float probability=expf(score);
        denominator+=probability;
        #pragma unroll
        for (unsigned j=0;j<8;++j) if (lane+32*j<width) {
            float v=values[token*stride+(unsigned long long)kh*width+lane+32*j];
            bad|=!(fabsf(v)<=1.0e12f);
            numerator[j]+=probability*v;
        }
        if (__any_sync(0xffffffffU,bad)) {bad=1;break;}
    }
    __shared__ float sums[8][257];
    __shared__ unsigned failures[8];
    #pragma unroll
    for (unsigned j=0;j<8;++j) if (lane+32*j<width) sums[warp][lane+32*j]=numerator[j];
    if (!lane) {sums[warp][256]=denominator;failures[warp]=bad;}
    __syncthreads();
    if (threadIdx.x<width) {
        float sum=0.0f;
        #pragma unroll
        for (unsigned i=0;i<8;++i) sum+=sums[i][threadIdx.x];
        parts[threadIdx.x]=sum;
    }
    if (!threadIdx.x) {
        float sum=0.0f;unsigned retry=0;
        #pragma unroll
        for (unsigned i=0;i<8;++i) {sum+=sums[i][256];retry|=failures[i];}
        parts[width]=sum;parts[width+1]=(float)retry;
    }
}

/* Same unified-scale reduction, but unsafe heads use Fyodor's coalesced
   stable CUDA implementation and its already budgeted per-head score scratch.
   The generic optional implementation remains independent of this helper. */
extern "C" __global__ void nya_fdpp_combine_warp(const float *q,
    const float *keys, const float *values, float *out, const float *parts,
    unsigned width, unsigned heads, unsigned kvheads, unsigned window,
    unsigned position, float scale, const unsigned *request, unsigned partitions,
    float *scores, unsigned capacity)
{
    unsigned h=blockIdx.x, tid=threadIdx.x;
    const float *p=parts+(unsigned long long)h*partitions*(width+2);
    __shared__ unsigned retry;
    __shared__ float denominator;
    if (!tid) {
        retry=0; denominator=0.0f;
        for (unsigned i=0;i<partitions;++i) {
            retry|=p[i*(width+2)+width+1]!=0.0f;
            denominator+=p[i*(width+2)+width];
        }
        if (!(denominator>0.0f) || !isfinite(denominator)) retry=1;
    }
    __syncthreads();
    if (retry) {
        nya_attention_partition(q,keys,values,out,scores,width,heads,kvheads,
            window,position,capacity,scale,request,1);
    } else if (tid<width) {
        float sum=0.0f;
        for (unsigned i=0;i<partitions;++i) sum+=p[i*(width+2)+tid];
        out[(unsigned long long)h*width+tid]=sum/denominator;
    }
}
