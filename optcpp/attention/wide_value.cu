/* Original Fyodor experimental four-column attention value reduction. */
__device__ __forceinline__ void nya_attention_partition_vec4(const float *q, const float *keys,
    const float *values, float *out, float *scores, unsigned width, unsigned heads,
    unsigned kvheads, unsigned window, unsigned end, unsigned capacity, float scale,
    const unsigned *request, unsigned partitions, unsigned ring_capacity)
{
    if (request) end = request[1];
    unsigned query = blockIdx.y/partitions, partition = blockIdx.y%partitions;
    end += query;
    unsigned first = window && end >= window ? end-window+1 : 0;
    if (partitions > 1) {
        unsigned length = end-first+1;
        end = first+(unsigned)((unsigned long long)length*(partition+1)/partitions)-1;
        first += (unsigned)((unsigned long long)length*partition/partitions);
    }
    unsigned head = blockIdx.x, kh = head/(heads/kvheads), stride = kvheads*width;
    unsigned lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    q += ((unsigned long long)query*heads+head)*width;
    out += partitions == 1 ? ((unsigned long long)query*heads+head)*width :
        ((unsigned long long)head*partitions+partition)*(width+2);
    float *row = scores+((unsigned long long)query*heads+head)*capacity;
    __shared__ float partial[8][128];
    __shared__ float maxima[8], maximum, denominator;
    float local_max = -__int_as_float(0x7f800000);
    for (unsigned t = first+warp; t <= end; t += 8) {
        float value = 0;
        for (unsigned j = lane; j < width; j += 32)
            value += q[j]*keys[(unsigned long long)(ring_capacity ? t%ring_capacity : t)*stride+kh*width+j];
        for (int step = 16; step; step >>= 1) value += __shfl_down_sync(0xffffffffU,value,step);
        if (!lane) { row[t] = value*scale; local_max = fmaxf(local_max,row[t]); }
    }
    if (!lane) maxima[warp] = local_max;
    __syncthreads();
    if (!threadIdx.x) {
        float value = maxima[0];
        for (unsigned i = 1; i < 8; ++i) value = fmaxf(value,maxima[i]);
        maximum = value;
    }
    __syncthreads();
    float sum = 0;
    for (unsigned t = first+threadIdx.x; t <= end; t += 256) {
        row[t] = expf(row[t]-maximum); sum += row[t];
    }
    sum = nya_sum(sum);
    if (!threadIdx.x) denominator = sum;
    __syncthreads();
    /* Four adjacent output columns share the score load and synchronization.
       Every output keeps the original t/warp accumulation order. */
    for (unsigned base = 0; base < width; base += 128) {
        unsigned j = base+lane*4;
        float value[4] = {0,0,0,0};
        if (j < width) for (unsigned t = first+warp; t <= end; t += 8) {
            unsigned long long offset=(unsigned long long)(ring_capacity ? t%ring_capacity : t)*stride+kh*width+j;
            float score=row[t];
            if (!(width&3U)) {
                float4 v=*((const float4 *)(values+offset));
                value[0]+=score*v.x;value[1]+=score*v.y;
                value[2]+=score*v.z;value[3]+=score*v.w;
            } else {
                #pragma unroll
                for(unsigned c=0;c<4;++c) if(j+c<width) value[c]+=score*values[offset+c];
            }
        }
        #pragma unroll
        for(unsigned c=0;c<4;++c) partial[warp][lane*4+c]=value[c];
        __syncthreads();
        if (!warp) {
            #pragma unroll
            for(unsigned c=0;c<4;++c) if(j+c<width) {
                for(unsigned i=1;i<8;++i) value[c]+=partial[i][lane*4+c];
                out[j+c]=partitions==1 ? value[c]/denominator : value[c];
            }
        }
        __syncthreads();
    }
    if (partitions > 1 && !threadIdx.x) { out[width] = maximum; out[width+1] = denominator; }
}

extern "C" __global__ void nya_attention_vec4(const float *q,const float *keys,const float *values,float *out,float *scores,unsigned width,unsigned heads,unsigned kvheads,unsigned window,unsigned end,unsigned capacity,float scale,const unsigned *request) { nya_attention_partition_vec4(q,keys,values,out,scores,width,heads,kvheads,window,end,capacity,scale,request,1,0); }
extern "C" __global__ void nya_attention_vec4_split(const float *q,const float *keys,const float *values,float *out,float *scores,unsigned width,unsigned heads,unsigned kvheads,unsigned window,unsigned end,unsigned capacity,float scale,const unsigned *request) { nya_attention_partition_vec4(q,keys,values,out,scores,width,heads,kvheads,window,end,capacity,scale,request,4,0); }
extern "C" __global__ void nya_attention_vec4_ring(const float *q,const float *keys,const float *values,float *out,float *scores,unsigned width,unsigned heads,unsigned kvheads,unsigned window,unsigned end,unsigned capacity,float scale,const unsigned *request,unsigned ring_capacity) { nya_attention_partition_vec4(q,keys,values,out,scores,width,heads,kvheads,window,end,capacity,scale,request,1,ring_capacity); }
extern "C" __global__ void nya_attention_vec4_ring_split(const float *q,const float *keys,const float *values,float *out,float *scores,unsigned width,unsigned heads,unsigned kvheads,unsigned window,unsigned end,unsigned capacity,float scale,const unsigned *request,unsigned ring_capacity) { nya_attention_partition_vec4(q,keys,values,out,scores,width,heads,kvheads,window,end,capacity,scale,request,4,ring_capacity); }
