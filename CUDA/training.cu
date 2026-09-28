/* Native training derivatives. Each output has one writer, so accumulation is
   stream ordered without floating-point atomics. Packed frozen weights stay
   compressed for dX. These are original Fyodor kernels. */
extern "C" __global__ void nya_train_zero(unsigned char *data, unsigned long long bytes)
{
    unsigned long long i = (unsigned long long)blockIdx.x*256+threadIdx.x;
    if (i < bytes) data[i] = 0;
}
/* Classify F32 storage bits so subnormals and signed zero remain finite, while
   both signs and every payload of infinity/NaN are rejected. All threads join
   the vote, including the tail. Only one thread per failing block publishes. */
extern "C" __global__ void nya_train_check_finite(const float *data, unsigned *status,
    unsigned long long count, unsigned tag)
{
    unsigned long long i = (unsigned long long)blockIdx.x*256+threadIdx.x;
    int bad = i < count && (__float_as_uint(data[i]) & 0x7f800000u) == 0x7f800000u;
    if (__syncthreads_or(bad) && threadIdx.x == 0) atomicCAS(status,0u,tag);
}
__device__ __forceinline__ void nya_train_dx(const unsigned char *w, const float *dy, float *dx,
    unsigned long long inputs, unsigned long long row_bytes, unsigned outputs, unsigned tokens, unsigned TYPE)
{
    __shared__ float weights[32][33], gradients[32][33];
    unsigned linear = threadIdx.y*32+threadIdx.x;
    unsigned long long column = (unsigned long long)blockIdx.x*32+threadIdx.x;
    unsigned token = blockIdx.y*32+threadIdx.y;
    float sum[4][2] = {};
    for (unsigned long long base = 0; base < outputs; base += 32) {
        #pragma unroll
        for (unsigned slot = 0; slot < 4; ++slot) {
            unsigned row = linear%32, k = linear/32+slot*8;
            unsigned long long source_column = (unsigned long long)blockIdx.x*32+row;
            /* Lanes read adjacent original weight columns, then transpose into
               padded shared memory for the W^T multiply. */
            weights[row][k] = source_column < inputs && base+k < outputs ?
                nya_weight(w+(base+k)*row_bytes,source_column,TYPE) : 0;
            unsigned source_token = blockIdx.y*32+k;
            gradients[k][row] = source_token < tokens && base+row < outputs ?
                dy[(unsigned long long)source_token*outputs+base+row] : 0;
        }
        __syncthreads();
        #pragma unroll
        for (unsigned k = 0; k < 32; ++k) {
            float weight = weights[threadIdx.x][k];
            #pragma unroll
            for (unsigned slot = 0; slot < 4; ++slot)
                sum[slot][k&1] += weight*gradients[threadIdx.y+slot*8][k];
        }
        __syncthreads();
    }
    if (column < inputs) {
        #pragma unroll
        for (unsigned slot = 0; slot < 4; ++slot)
            if (token+slot*8 < tokens) dx[(unsigned long long)(token+slot*8)*inputs+column] += sum[slot][0]+sum[slot][1];
    }
}
#define NYA_TRAIN_DX(TYPE) extern "C" __global__ void nya_train_dx_##TYPE(const unsigned char *w, const float *dy, float *dx, unsigned long long i, unsigned long long s, unsigned o, unsigned n) { nya_train_dx(w,dy,dx,i,s,o,n,TYPE); }
NYA_TRAIN_DX(0) NYA_TRAIN_DX(1) NYA_TRAIN_DX(2) NYA_TRAIN_DX(8) NYA_TRAIN_DX(12) NYA_TRAIN_DX(14) NYA_TRAIN_DX(30)
#undef NYA_TRAIN_DX

extern "C" __global__ void nya_train_dw(const float *x, const float *dy, float *dw,
    unsigned inputs, unsigned outputs, unsigned tokens)
{
    __shared__ float gradients[32][33], inputs_tile[32][33];
    unsigned linear = threadIdx.y*32+threadIdx.x;
    unsigned column = blockIdx.x*32+threadIdx.x, row = blockIdx.y*32+threadIdx.y;
    float sum[4][2] = {};
    for (unsigned long long base = 0; base < tokens; base += 32) {
        #pragma unroll
        for (unsigned slot = 0; slot < 4; ++slot) {
            unsigned lane = linear%32, k = linear/32+slot*8;
            unsigned source_row = blockIdx.y*32+lane, source_column = blockIdx.x*32+lane;
            gradients[lane][k] = source_row < outputs && base+k < tokens ? dy[(base+k)*outputs+source_row] : 0;
            inputs_tile[k][lane] = source_column < inputs && base+k < tokens ? x[(base+k)*inputs+source_column] : 0;
        }
        __syncthreads();
        #pragma unroll
        for (unsigned k = 0; k < 32; ++k) {
            float input = inputs_tile[k][threadIdx.x];
            #pragma unroll
            for (unsigned slot = 0; slot < 4; ++slot)
                sum[slot][k&1] += gradients[threadIdx.y+slot*8][k]*input;
        }
        __syncthreads();
    }
    if (column < inputs) {
        #pragma unroll
        for (unsigned slot = 0; slot < 4; ++slot)
            if (row+slot*8 < outputs) dw[(unsigned long long)(row+slot*8)*inputs+column] += sum[slot][0]+sum[slot][1];
    }
}
