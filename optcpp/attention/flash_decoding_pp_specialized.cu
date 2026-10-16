/* Original optional width/unroll specializations. Runtime uses the r32 entries;
   other entries remain available for comparative diagnostics. */
template<unsigned WIDTH, unsigned UNROLL, unsigned BOUND=16>
__device__ __forceinline__ void nya_fdpp_width_body(const float *q,
    const float *keys, const float *values, float *parts, unsigned runtime_width,
    unsigned heads, unsigned kvheads, unsigned window, unsigned position,
    float scale, const unsigned *request, unsigned partitions)
{
    const unsigned width=WIDTH;
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
    float query[WIDTH/32], numerator[WIDTH/32];
    #pragma unroll
    for (unsigned j=0;j<WIDTH/32;++j) {
        query[j]=lane+32*j<width?q[lane+32*j]:0.0f;
        numerator[j]=0.0f;
    }
    float denominator=0.0f;
    unsigned bad=0;
    #pragma unroll UNROLL
    for (unsigned long long token=begin+warp;token<stop;token+=8) {
        float score=0.0f;
        #pragma unroll
        for (unsigned j=0;j<WIDTH/32;++j)
            if (lane+32*j<width) score+=query[j]*keys[token*stride+(unsigned long long)kh*width+lane+32*j];
        for (unsigned step=16;step;step>>=1) score+=__shfl_down_sync(0xffffffffU,score,step);
        score=__shfl_sync(0xffffffffU,score,0)*scale;
        if (!(score>=-(float)BOUND && score<=(float)BOUND)) {bad=1;break;}
        float probability=expf(score);
        denominator+=probability;
        #pragma unroll
        for (unsigned j=0;j<WIDTH/32;++j) if (lane+32*j<width) {
            float v=values[token*stride+(unsigned long long)kh*width+lane+32*j];
            bad|=!(fabsf(v)<=1.0e12f);
            numerator[j]+=probability*v;
        }
        if (__any_sync(0xffffffffU,bad)) {bad=1;break;}
    }
    __shared__ float sums[8][WIDTH+1];
    __shared__ unsigned failures[8];
    #pragma unroll
    for (unsigned j=0;j<WIDTH/32;++j) if (lane+32*j<width) sums[warp][lane+32*j]=numerator[j];
    if (!lane) {sums[warp][WIDTH]=denominator;failures[warp]=bad;}
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
        for (unsigned i=0;i<8;++i) {sum+=sums[i][WIDTH];retry|=failures[i];}
        parts[width]=sum;parts[width+1]=(float)retry;
    }
}

extern "C" __global__ void nya_fdpp_w64_u1(const float *q,
    const float *keys, const float *values, float *parts, unsigned width,
    unsigned heads, unsigned kvheads, unsigned window, unsigned position,
    float scale, const unsigned *request, unsigned partitions)
{ nya_fdpp_width_body<64,1>(q,keys,values,parts,width,heads,kvheads,
    window,position,scale,request,partitions); }

extern "C" __global__ void nya_fdpp_w64_u2(const float *q,
    const float *keys, const float *values, float *parts, unsigned width,
    unsigned heads, unsigned kvheads, unsigned window, unsigned position,
    float scale, const unsigned *request, unsigned partitions)
{ nya_fdpp_width_body<64,2>(q,keys,values,parts,width,heads,kvheads,
    window,position,scale,request,partitions); }

extern "C" __global__ void nya_fdpp_w64_u4(const float *q,
    const float *keys, const float *values, float *parts, unsigned width,
    unsigned heads, unsigned kvheads, unsigned window, unsigned position,
    float scale, const unsigned *request, unsigned partitions)
{ nya_fdpp_width_body<64,4>(q,keys,values,parts,width,heads,kvheads,
    window,position,scale,request,partitions); }

extern "C" __global__ void nya_fdpp_w128_u1(const float *q,
    const float *keys, const float *values, float *parts, unsigned width,
    unsigned heads, unsigned kvheads, unsigned window, unsigned position,
    float scale, const unsigned *request, unsigned partitions)
{ nya_fdpp_width_body<128,1>(q,keys,values,parts,width,heads,kvheads,
    window,position,scale,request,partitions); }

extern "C" __global__ void nya_fdpp_w128_u2(const float *q,
    const float *keys, const float *values, float *parts, unsigned width,
    unsigned heads, unsigned kvheads, unsigned window, unsigned position,
    float scale, const unsigned *request, unsigned partitions)
{ nya_fdpp_width_body<128,2>(q,keys,values,parts,width,heads,kvheads,
    window,position,scale,request,partitions); }

extern "C" __global__ void nya_fdpp_w128_u4(const float *q,
    const float *keys, const float *values, float *parts, unsigned width,
    unsigned heads, unsigned kvheads, unsigned window, unsigned position,
    float scale, const unsigned *request, unsigned partitions)
{ nya_fdpp_width_body<128,4>(q,keys,values,parts,width,heads,kvheads,
    window,position,scale,request,partitions); }

extern "C" __global__ void nya_fdpp_w64_u4_r32(const float *q,
    const float *keys, const float *values, float *parts, unsigned width,
    unsigned heads, unsigned kvheads, unsigned window, unsigned position,
    float scale, const unsigned *request, unsigned partitions)
{ nya_fdpp_width_body<64,4,32>(q,keys,values,parts,width,heads,kvheads,
    window,position,scale,request,partitions); }

extern "C" __global__ void nya_fdpp_w128_u1_r32(const float *q,
    const float *keys, const float *values, float *parts, unsigned width,
    unsigned heads, unsigned kvheads, unsigned window, unsigned position,
    float scale, const unsigned *request, unsigned partitions)
{ nya_fdpp_width_body<128,1,32>(q,keys,values,parts,width,heads,kvheads,
    window,position,scale,request,partitions); }
