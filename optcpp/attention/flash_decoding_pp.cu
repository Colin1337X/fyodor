/* Original implementation of the unified-scale split attention technique in
   FlashDecoding++ (Hong et al., arXiv:2311.01282, section 3). No artifact code
   copied. F32 KV, decode only, width <= 256, exactly 256 threads per block.
   All partitions use phi=0. Out-of-range inputs trigger whole-head stable
   recomputation; no clamping, precision downcast, or dropped probability mass.
   No warp-size, tensor-core, TMA or TMEM dependencies. */
extern "C" __global__ void nya_fdpp_partial(const float *q,
    const float *keys, const float *values, float *parts, unsigned width,
    unsigned heads, unsigned kvheads, unsigned window, unsigned position,
    float scale, const unsigned *request, unsigned partitions)
{
    unsigned h=blockIdx.x, part=blockIdx.y, tid=threadIdx.x;
    unsigned end=request?request[1]:position;
    unsigned first=window && end>=window ? end-window+1 : 0;
    unsigned long long length=(unsigned long long)end-first+1;
    unsigned long long begin=first+length*part/partitions;
    unsigned long long stop=first+length*(part+1)/partitions;
    unsigned kh=h/(heads/kvheads);
    unsigned long long stride=(unsigned long long)kvheads*width;
    q+=(unsigned long long)h*width;
    parts+=((unsigned long long)h*partitions+part)*(width+2);
    __shared__ float probability[256];
    __shared__ unsigned bad;
    float numerator=0.0f, denominator=0.0f;
    if (!tid) bad=0;
    __syncthreads();
    for (unsigned long long base=begin;base<stop;base+=256) {
        unsigned long long token=base+tid;
        float score=0.0f;
        if (token<stop) {
            for (unsigned j=0;j<width;++j)
                score+=q[j]*keys[token*stride+(unsigned long long)kh*width+j];
            score*=scale;
            /* Also rejects NaN. This conservative range leaves headroom for
               summation at any uint32 context length and bounded values. */
            if (!(score>=-16.0f && score<=16.0f)) atomicExch(&bad,1U);
        }
        probability[tid]=token<stop && score>=-16.0f && score<=16.0f ? expf(score) : 0.0f;
        __syncthreads();
        unsigned abort_tile=bad;
        __syncthreads();
        if (abort_tile) break;
        if (tid<width) {
            for (unsigned i=0;i<256 && base+i<stop;++i) {
                float v=values[(base+i)*stride+(unsigned long long)kh*width+tid];
                if (!(fabsf(v)<=1.0e12f)) atomicExch(&bad,1U);
                numerator+=probability[i]*v;
            }
        }
        if (!tid) for (unsigned i=0;i<256;++i) denominator+=probability[i];
        __syncthreads();
        abort_tile=bad;
        __syncthreads();
        if (abort_tile) break;
    }
    if (tid<width) parts[tid]=numerator;
    if (!tid) { parts[width]=denominator; parts[width+1]=(float)bad; }
}

extern "C" __global__ void nya_fdpp_combine(const float *q,
    const float *keys, const float *values, float *out, const float *parts,
    unsigned width, unsigned heads, unsigned kvheads, unsigned window,
    unsigned position, float scale, const unsigned *request, unsigned partitions)
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
        nya_attention_portable_body(q,keys,values,out,width,heads,kvheads,
            window,position,1,scale,request,1);
    } else if (tid<width) {
        float sum=0.0f;
        for (unsigned i=0;i<partitions;++i) sum+=p[i*(width+2)+tid];
        out[(unsigned long long)h*width+tid]=sum/denominator;
    }
}
