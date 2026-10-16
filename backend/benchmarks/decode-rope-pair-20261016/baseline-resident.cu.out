/* Resident transformer primitives. All launches use the same stream: dependencies
   are ordered on device and only the final logits require a host fence. */
__device__ float nya_sum(float value)
{
    __shared__ float warps[8];
    unsigned lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    for (int step = 16; step; step >>= 1) value += __shfl_down_sync(0xffffffffU, value, step);
    if (!lane) warps[warp] = value;
    __syncthreads();
    value = threadIdx.x < 8 ? warps[lane] : 0;
    if (!warp) for (int step = 16; step; step >>= 1) value += __shfl_down_sync(0xffffffffU, value, step);
    return value;
}

/* A request owns this two-word packet. One stream-ordered launch updates it
   before replay, so a captured graph never embeds a stale token or position.
   Kernel arguments carry the values; no host buffer lifetime or copy is needed. */
extern "C" __global__ void nya_decode_request(unsigned *request, unsigned token, unsigned position)
{
    if (!threadIdx.x) { request[0] = token; request[1] = position; }
}

extern "C" __global__ void nya_gather(const unsigned char *w, float *out,
    unsigned long long n, unsigned long long row_bytes, unsigned type, unsigned token, float scale,
    const unsigned *request)
{
    if (request) token = request[0];
    unsigned long long i = (unsigned long long)blockIdx.x * 256 + threadIdx.x;
    if (i < n) out[i] = nya_weight(w + token * row_bytes, i, type) * scale;
}

/* A bounded by-value token packet fits within CUDA's 4 KiB legacy parameter
   budget, avoiding a separate token-buffer upload or 512 tiny gather launches.
   Decode keeps the compact single-token entry point above. Host admission caps
   prefill at 512 and validates every token before submitting this kernel. */
struct nya_gather_tokens { unsigned values[512]; };
extern "C" __global__ void nya_gather_batch(const unsigned char *w, float *out,
    unsigned long long n, unsigned long long row_bytes, unsigned type,
    nya_gather_tokens tokens, unsigned count, float scale)
{
    unsigned long long i = (unsigned long long)blockIdx.x*256+threadIdx.x;
    if (i < n*count) {
        unsigned token = tokens.values[i/n];
        out[i] = nya_weight(w+(unsigned long long)token*row_bytes,i%n,type)*scale;
    }
}

extern "C" __global__ void nya_norm(const float *x, float *out, const unsigned char *w,
    unsigned long long n, unsigned type, float epsilon)
{
    unsigned long long base = (unsigned long long)blockIdx.x * n;
    float sum = 0;
    for (unsigned long long i = threadIdx.x; i < n; i += 256) sum += x[base+i] * x[base+i];
    sum = nya_sum(sum);
    __shared__ float scale;
    if (!threadIdx.x) scale = rsqrtf(sum / (float)n + epsilon);
    __syncthreads();
    for (unsigned long long i = threadIdx.x; i < n; i += 256)
        out[base+i] = x[base+i] * scale * (w ? nya_weight(w, i, type) : 1.0f);
}

/* Residual addition immediately precedes FFN normalization in each supported
   dense layer. Preserve the rounded F32 sum in x, then use the same RMSNorm
   reduction as the separate kernels. This saves a launch and an extra read of
   the residual stream; no half conversion or changed precision is involved. */
extern "C" __global__ void nya_add_norm(float *x, float *out, const float *residual,
    const unsigned char *w, unsigned long long n, unsigned type, float epsilon)
{
    unsigned long long base = (unsigned long long)blockIdx.x*n;
    float sum = 0;
    for (unsigned long long i = threadIdx.x; i < n; i += 256) {
        float value = x[base+i]+residual[base+i];
        x[base+i] = value;
        sum += value*value;
    }
    sum = nya_sum(sum);
    __shared__ float scale;
    if (!threadIdx.x) scale = rsqrtf(sum/(float)n+epsilon);
    __syncthreads();
    for (unsigned long long i = threadIdx.x; i < n; i += 256)
        out[base+i] = x[base+i]*scale*(w ? nya_weight(w,i,type) : 1.0f);
}

extern "C" __global__ void nya_rope(float *x, const float *frequencies,
    unsigned width, unsigned heads, unsigned position, unsigned gemma, unsigned batch,
    const unsigned *request, float *cache_keys, float *cache_values, const float *value)
{
    if (request) position = request[1];
    unsigned i = blockIdx.x * 256 + threadIdx.x, half = width / 2;
    if (i >= heads * half * batch) return;
    unsigned token = i / (heads * half);
    position += token;
    unsigned head = i / half, pair = i % half;
    unsigned a = head * width + (gemma ? pair : pair * 2), b = a + (gemma ? half : 1);
    float co = cosf((float)position * frequencies[pair]);
    float si = sinf((float)position * frequencies[pair]);
    float first = x[a], second = x[b];
    x[a] = first * co - second * si;
    x[b] = second * co + first * si;
    /* Decode projections use fixed temporary addresses. Each RoPE pair also
       commits its rotated key and normalized value to the indexed cache. This
       fuses the store without another launch or changing reduction arithmetic.
       Query RoPE and ordinary prefill pass null cache pointers. Shared-KV
       consumers omit key RoPE and use their producer's existing cache. */
    if (cache_keys) {
        unsigned long long offset = (unsigned long long)position * heads * width;
        cache_keys[offset+a] = x[a]; cache_keys[offset+b] = x[b];
        cache_values[offset+a] = value[a]; cache_values[offset+b] = value[b];
    }
}

/* kind: copy/scale, add, SiLU gate, GELU gate, multiply, logit softcap. */
extern "C" __global__ void nya_vector(float *out, const float *x, const float *y,
    unsigned long long n, unsigned kind, float scale)
{
    unsigned long long i = (unsigned long long)blockIdx.x * 256 + threadIdx.x;
    if (i >= n) return;
    float a = x[i];
    if (kind == 0) out[i] = a * scale;
    else if (kind == 1) out[i] = a + y[i];
    else if (kind == 2) out[i] = a / (1 + expf(-a)) * y[i];
    else if (kind == 3) out[i] = 0.5f * a * (1 + tanhf(0.7978845608028654f * (a + 0.044715f * a*a*a))) * y[i];
    else if (kind == 4) out[i] = a * y[i];
    else if (kind == 5) out[i] = scale * tanhf(a / scale);
}

/* One block owns one head and its score row. This bounded scratch design avoids
   materializing a full sequence-squared attention matrix during decode. */
extern "C" __global__ void nya_attention_reference(const float *q, const float *keys,
    const float *values, float *out, float *scores, unsigned width, unsigned heads,
    unsigned kvheads, unsigned window, unsigned end, unsigned capacity, float scale,
    const unsigned *request)
{
    if (request) end = request[1];
    end += blockIdx.y;
    unsigned first = window && end >= window ? end - window + 1 : 0;
    q += (unsigned long long)blockIdx.y * heads * width;
    out += (unsigned long long)blockIdx.y * heads * width;
    scores += (unsigned long long)blockIdx.y * heads * capacity;
    unsigned head = blockIdx.x, kh = head / (heads / kvheads), stride = kvheads * width;
    float *row = scores + (unsigned long long)head * capacity;
    __shared__ float reduction[256];
    float maximum = -__int_as_float(0x7f800000);
    for (unsigned t = first + threadIdx.x; t <= end; t += 256) {
        float value = 0;
        for (unsigned j = 0; j < width; ++j) value += q[head*width+j] * keys[(unsigned long long)t*stride+kh*width+j];
        row[t] = value * scale;
        maximum = fmaxf(maximum, row[t]);
    }
    reduction[threadIdx.x] = maximum;
    __syncthreads();
    for (unsigned step = 128; step; step >>= 1) {
        if (threadIdx.x < step) reduction[threadIdx.x] = fmaxf(reduction[threadIdx.x], reduction[threadIdx.x+step]);
        __syncthreads();
    }
    maximum = reduction[0];
    float sum = 0;
    for (unsigned t = first + threadIdx.x; t <= end; t += 256) { row[t] = expf(row[t] - maximum); sum += row[t]; }
    sum = nya_sum(sum);
    __shared__ float denominator;
    if (!threadIdx.x) denominator = sum;
    __syncthreads();
    for (unsigned j = threadIdx.x; j < width; j += 256) {
        float value = 0;
        for (unsigned t = first; t <= end; ++t) value += (row[t] / denominator) * values[(unsigned long long)t*stride+kh*width+j];
        out[head*width+j] = value;
    }
}

/* Eight warps cooperate on one head. A warp consumes contiguous head elements
   of one key, replacing the reference kernel's strided per-thread key reads.
   Value accumulation partitions the context across warps and reduces their
   partial vectors in shared memory. Scratch remains O(batch*heads*capacity),
   and causal/shared-KV indexing is identical to the reference kernel above. */
__device__ __forceinline__ void nya_attention_partition(const float *q, const float *keys,
    const float *values, float *out, float *scores, unsigned width, unsigned heads,
    unsigned kvheads, unsigned window, unsigned end, unsigned capacity, float scale,
    const unsigned *request, unsigned partitions)
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
    __shared__ float partial[8][32];
    __shared__ float maxima[8], maximum, denominator;
    float local_max = -__int_as_float(0x7f800000);
    for (unsigned t = first+warp; t <= end; t += 8) {
        float value = 0;
        for (unsigned j = lane; j < width; j += 32)
            value += q[j]*keys[(unsigned long long)t*stride+kh*width+j];
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
    /* All lanes enter every tile, including partial head widths, so barriers
       cannot diverge. Only the final guarded store touches partial outputs. */
    for (unsigned base = 0; base < width; base += 32) {
        unsigned j = base+lane;
        float value = 0;
        if (j < width) for (unsigned t = first+warp; t <= end; t += 8)
            value += row[t]*values[(unsigned long long)t*stride+kh*width+j];
        partial[warp][lane] = value;
        __syncthreads();
        if (!warp && j < width) {
            for (unsigned i = 1; i < 8; ++i) value += partial[i][lane];
            out[j] = partitions == 1 ? value/denominator : value;
        }
        __syncthreads();
    }
    if (partitions > 1 && !threadIdx.x) { out[width] = maximum; out[width+1] = denominator; }
}

#define NYA_ATTENTION_ENTRY(NAME, PARTITIONS) \
extern "C" __global__ void NAME(const float *q, const float *keys, const float *values, \
    float *out, float *scores, unsigned width, unsigned heads, unsigned kvheads, \
    unsigned window, unsigned end, unsigned capacity, float scale, const unsigned *request) \
{ nya_attention_partition(q,keys,values,out,scores,width,heads,kvheads,window,end,capacity,scale,request,PARTITIONS); }
NYA_ATTENTION_ENTRY(nya_attention, 1)
NYA_ATTENTION_ENTRY(nya_attention_split, 4)
#undef NYA_ATTENTION_ENTRY

/* Four independent context ranges expose more blocks for long decode. Combine
   unnormalized value sums using a common softmax maximum. This is an original
   F32 implementation of the partition-and-rescale technique also studied in
   ggml's MIT-licensed fattn-common.cuh (b10809); no upstream source is copied.
   The host admits only ranges of at least 257 keys, so every partition is live. */
extern "C" __global__ void nya_attention_combine(const float *parts, float *out, unsigned width)
{
    unsigned j = threadIdx.x, head = blockIdx.x;
    parts += (unsigned long long)head*4*(width+2);
    if (j >= width) return;
    float maximum = parts[width];
    #pragma unroll
    for (unsigned i = 1; i < 4; ++i) maximum = fmaxf(maximum,parts[i*(width+2)+width]);
    float numerator = 0, denominator = 0;
    #pragma unroll
    for (unsigned i = 0; i < 4; ++i) {
        const float *p = parts+i*(width+2);
        float correction = expf(p[width]-maximum);
        numerator += correction*p[j];
        denominator += correction*p[width+1];
    }
    out[(unsigned long long)head*width+j] = numerator/denominator;
}

/* Prefill owns eight queries per block. Load each 32-key K/V tile once into
   shared memory, then reuse it across all eight query warps. Online softmax
   carries a running maximum, denominator and unnormalized value numerator;
   rescaling both accumulators when the maximum grows avoids overflow without
   writing an intermediate score matrix to global memory. This is F32 SIMT
   arithmetic throughout, including the persistent KV cache.

   The host admits 64/128-wide heads and batches of at least eight. Other
   shapes and decode retain the general kernel. Masking is per query, so tails,
   existing prefixes, GQA and sliding windows do not read future cache entries. */
extern "C" __global__ void nya_attention_prefill(const float *q, const float *keys,
    const float *values, float *out, unsigned width, unsigned heads,
    unsigned kvheads, unsigned window, unsigned position, unsigned count, float scale)
{
    const unsigned lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const unsigned query_index = blockIdx.y*8+warp, head = blockIdx.x;
    const unsigned kh = head/(heads/kvheads), stride = kvheads*width;
    const bool valid = query_index < count;
    const unsigned end = valid ? position+query_index : 0;
    const unsigned first = window && end >= window ? end-window+1 : 0;
    const unsigned group_start = blockIdx.y*8;
    const unsigned group_end = position+(group_start+8 < count ? group_start+8 : count)-1;
    const unsigned group_first = window && position+group_start >= window ? position+group_start-window+1 : 0;
    /* Exact-width allocation avoids paying the 128-wide shared-memory cost
       for 64-wide heads, which would unnecessarily limit block occupancy. */
    extern __shared__ float tile_data[];
    /* Transpose K with a 33-float pitch: global loads stay contiguous and
       the transpose stores avoid 32-way shared-memory bank conflicts. Each
       query warp then evaluates 32 independent scores (one per lane), rather
       than performing a warp reduction for every key. Q is warp-broadcast. */
    float *tile_k = tile_data, *tile_v = tile_k+33*width;
    float *tile_q = tile_v+32*width;
    float numerator[4] = {0,0,0,0};
    #pragma unroll
    for (unsigned i = 0; i < 4; ++i)
        if (lane+i*32 < width)
            tile_q[warp*width+lane+i*32] = valid ? q[((unsigned long long)query_index*heads+head)*width+lane+i*32] : 0;
    __syncwarp();
    const float negative_infinity = -__int_as_float(0x7f800000);
    float maximum = negative_infinity, denominator = 0;
    for (unsigned long long tile = group_first; tile <= group_end; tile += 32) {
        /* Every block thread participates, including inactive final queries.
           Guard the final tile against the actual populated cache prefix. */
        for (unsigned i = threadIdx.x; i < 32*width; i += 256) {
            unsigned t = i/width, j = i%width;
            unsigned long long offset = (tile+t)*stride+kh*width+j;
            tile_k[j*33+t] = tile+t <= group_end ? keys[offset] : 0;
            tile_v[t*width+j] = tile+t <= group_end ? values[offset] : 0;
        }
        __syncthreads();
        float dot[4] = {0,0,0,0};
        for (unsigned j = 0; j < width; j += 4) {
            #pragma unroll
            for (unsigned i = 0; i < 4; ++i)
                dot[i] += tile_q[warp*width+j+i]*tile_k[(j+i)*33+lane];
        }
        float score = valid && tile+lane >= first && tile+lane <= end ? ((dot[0]+dot[1])+(dot[2]+dot[3]))*scale : negative_infinity;
        float tile_max = score;
        for (int step = 16; step; step >>= 1) tile_max = fmaxf(tile_max,__shfl_xor_sync(0xffffffffU,tile_max,step));
        /* A sliding-window query can mask a whole tile. Skipping that warp's
           update also avoids evaluating -infinity minus -infinity. */
        if (tile_max != negative_infinity) {
            float next_max = fmaxf(maximum,tile_max);
            float correction = expf(maximum-next_max);
            float probability = expf(score-next_max), mass = probability;
            for (int step = 16; step; step >>= 1) mass += __shfl_xor_sync(0xffffffffU,mass,step);
            denominator = denominator*correction+mass;
            #pragma unroll
            for (unsigned i = 0; i < 4; ++i) numerator[i] *= correction;
            for (unsigned t = 0; t < 32; ++t) {
                float p = __shfl_sync(0xffffffffU,probability,(int)t);
                #pragma unroll
                for (unsigned i = 0; i < 4; ++i)
                    if (lane+i*32 < width) numerator[i] += p*tile_v[t*width+lane+i*32];
            }
            maximum = next_max;
        }
        __syncthreads();
    }
    #pragma unroll
    for (unsigned i = 0; i < 4; ++i)
        if (valid && lane+i*32 < width)
            out[((unsigned long long)query_index*heads+head)*width+lane+i*32] = numerator[i]/denominator;
}
