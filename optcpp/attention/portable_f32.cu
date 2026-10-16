/* Original Fyodor portable online-softmax attention. No warp-size assumptions,
   tensor-core instructions, TMEM, TMA, or host C++ ABI. CUDA/HIP device dialect.
   This is a compatibility implementation, not the upstream FA4 kernel. */
__device__ void nya_attention_portable_body(const float *q,
    const float *keys, const float *values, float *out, unsigned width,
    unsigned heads, unsigned kvheads, unsigned window, unsigned position,
    unsigned count, float scale, const unsigned *request)
{
    unsigned head=blockIdx.x, query=blockIdx.y, tid=threadIdx.x;
    if (query>=count) return;
    unsigned end=(request?request[1]:position)+query;
    unsigned first=window && end>=window ? end-window+1 : 0;
    unsigned kh=head/(heads/kvheads);
    unsigned long long stride=(unsigned long long)kvheads*width;
    q+=((unsigned long long)query*heads+head)*width;
    out+=((unsigned long long)query*heads+head)*width;
    __shared__ float dots[256], probabilities[64], correction, denominator;
    __shared__ float maximum;
    float numerator=0.0f;
    if (!tid) { maximum=-__int_as_float(0x7f800000); denominator=0.0f; }
    __syncthreads();
    for (unsigned long long base=first; base<=end; base+=64) {
        unsigned key_index=tid/4, component=tid%4;
        unsigned long long token=base+key_index;
        float dot=0.0f;
        if (token<=end) for (unsigned j=component;j<width;j+=4)
            dot+=q[j]*keys[token*stride+(unsigned long long)kh*width+j];
        dots[tid]=dot;
        __syncthreads();
        if (tid<64) {
            float score=((dots[tid*4]+dots[tid*4+1])+dots[tid*4+2])+dots[tid*4+3];
            probabilities[tid]=base+tid<=end ? score*scale : -__int_as_float(0x7f800000);
        }
        __syncthreads();
        if (!tid) {
            float next=maximum;
            for (unsigned i=0;i<64 && base+i<=end;++i) next=fmaxf(next,probabilities[i]);
            correction=expf(maximum-next);
            maximum=next;
        }
        __syncthreads();
        if (tid<64) probabilities[tid]=base+tid<=end ? expf(probabilities[tid]-maximum) : 0.0f;
        __syncthreads();
        if (!tid) {
            float sum=0.0f;
            for (unsigned i=0;i<64;++i) sum+=probabilities[i];
            denominator=denominator*correction+sum;
        }
        if (tid<width) {
            numerator*=correction;
            for (unsigned i=0;i<64 && base+i<=end;++i)
                numerator+=probabilities[i]*values[(base+i)*stride+(unsigned long long)kh*width+tid];
        }
        __syncthreads();
    }
    if (tid<width) out[tid]=numerator/denominator;
}

extern "C" __global__ void nya_attention_portable_f32(const float *q,
    const float *keys, const float *values, float *out, unsigned width,
    unsigned heads, unsigned kvheads, unsigned window, unsigned position,
    unsigned count, float scale, const unsigned *request)
{
    nya_attention_portable_body(q,keys,values,out,width,heads,kvheads,window,
        position,count,scale,request);
}
