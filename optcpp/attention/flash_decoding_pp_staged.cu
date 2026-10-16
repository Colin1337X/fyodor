/* Original FDPP unified-scale variant using Fyodor's existing score workspace.
   Parallel exponent evaluation avoids the per-warp streaming exp dependency.
   Every KV partition remains independent; no cross-block maximum is required.
   Diagnostic only until numerical and end-to-end performance acceptance. */
extern "C" __global__ void nya_fdpp_partial_staged(const float *q,
    const float *keys, const float *values, float *parts, unsigned width,
    unsigned heads, unsigned kvheads, unsigned window, unsigned position,
    float scale, const unsigned *request, unsigned partitions,
    float *scores, unsigned capacity)
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
    float *row=scores+(unsigned long long)h*capacity;
    __shared__ unsigned failures[8], bad;
    __shared__ float partial[8][32], denominator;
    unsigned local_bad=0;
    for (unsigned long long t=begin+warp;t<stop;t+=8) {
        float score=0.0f;
        for (unsigned j=lane;j<width;j+=32)
            score+=q[j]*keys[t*stride+(unsigned long long)kh*width+j];
        for (unsigned step=16;step;step>>=1) score+=__shfl_down_sync(0xffffffffU,score,step);
        if (!lane) {
            score*=scale; row[t]=score;
            local_bad|=!(score>=-16.0f && score<=16.0f);
        }
    }
    if (!lane) failures[warp]=local_bad;
    __syncthreads();
    if (!threadIdx.x) {
        unsigned failed=0;
        for (unsigned i=0;i<8;++i) failed|=failures[i];
        bad=failed;
    }
    __syncthreads();
    if (bad) {
        if (threadIdx.x<width) parts[threadIdx.x]=0.0f;
        if (!threadIdx.x) {parts[width]=0.0f;parts[width+1]=1.0f;}
        return;
    }
    float sum=0.0f;
    for (unsigned long long t=begin+threadIdx.x;t<stop;t+=256) {
        float probability=expf(row[t]);row[t]=probability;sum+=probability;
    }
    sum=nya_sum(sum);
    if (!threadIdx.x) denominator=sum;
    __syncthreads();
    for (unsigned base=0;base<width;base+=32) {
        unsigned j=base+lane;
        float value=0.0f;
        if (j<width) for (unsigned long long t=begin+warp;t<stop;t+=8) {
            float v=values[t*stride+(unsigned long long)kh*width+j];
            local_bad|=!(fabsf(v)<=1.0e12f);
            value+=row[t]*v;
        }
        partial[warp][lane]=value;
        __syncthreads();
        if (!warp && j<width) {
            for (unsigned i=1;i<8;++i) value+=partial[i][lane];
            parts[j]=value;
        }
        __syncthreads();
    }
    if (local_bad) atomicOr(&bad,1U);
    __syncthreads();
    if (!threadIdx.x) {parts[width]=denominator;parts[width+1]=(float)bad;}
}
