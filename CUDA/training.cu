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
/* Match the CPU graph's double intermediates and stable negative SiLU branch.
   Storage and accumulated gradients stay F32. */
extern "C" __global__ void nya_train_unary(const float *x, const float *dy, float *out,
    unsigned long long count, unsigned operation, double scalar, int backward)
{
    unsigned long long i = (unsigned long long)blockIdx.x*256+threadIdx.x;
    if (i >= count) return;
    double v = x[i], result;
    if (!backward) {
        if (operation == 0) result = v*scalar;
        else if (operation == 1) result = v*(v >= 0 ? 1.0/(1.0+exp(-v)) : exp(v)/(1.0+exp(v)));
        else if (operation == 2) result = 0.5*v*(1.0+tanh(0.7978845608028654*(v+0.044715*v*v*v)));
        else result = scalar*tanh(v/scalar);
        out[i] = (float)result;
    } else {
        if (operation == 0) result = scalar;
        else if (operation == 1) {
            double s = v >= 0 ? 1.0/(1.0+exp(-v)) : exp(v)/(1.0+exp(v));
            result = s*(1.0+v*(1.0-s));
        } else if (operation == 2) {
            double z = tanh(0.7978845608028654*(v+0.044715*v*v*v));
            result = 0.5*(1.0+z)+0.5*v*(1.0-z*z)*0.7978845608028654*(1.0+0.134145*v*v);
        } else {
            double z = tanh(v/scalar); result = 1.0-z*z;
        }
        out[i] += (float)((double)dy[i]*result);
    }
}
__device__ __forceinline__ unsigned long long nya_train_broadcast_index(unsigned long long row,
    unsigned long long col, unsigned long long rows, unsigned long long cols)
{
    return (rows == 1 ? 0 : row)*cols+(cols == 1 ? 0 : col);
}
/* A gradient element has exactly one writer. Broadcast reductions visit the
   same row-major contributions as the CPU graph, without atomic float sums.
   Shared gradient destinations interleave a/b contributions at each position. */
__device__ __forceinline__ float nya_train_binary_gradient(const float *a, const float *b,
    const float *dy, float initial, unsigned long long i, unsigned long long ar, unsigned long long ac,
    unsigned long long br, unsigned long long bc, unsigned long long rows, unsigned long long cols,
    unsigned operation, unsigned side, int shared)
{
    unsigned long long own_rows = side ? br : ar, own_cols = side ? bc : ac;
    unsigned long long first_row = own_rows == 1 ? 0 : i/own_cols;
    unsigned long long last_row = own_rows == 1 ? rows : first_row+1;
    unsigned long long first_col = own_cols == 1 ? 0 : i%own_cols;
    unsigned long long last_col = own_cols == 1 ? cols : first_col+1;
    float sum = initial;
    for (unsigned long long row = first_row; row < last_row; ++row)
        for (unsigned long long col = first_col; col < last_col; ++col) {
            float g = dy[row*cols+col];
            unsigned long long ia = nya_train_broadcast_index(row,col,ar,ac);
            unsigned long long ib = nya_train_broadcast_index(row,col,br,bc);
            sum += operation == 0 ? g : g*(side ? a[ia] : b[ib]);
            if (shared) sum += operation == 0 ? g : g*a[ia];
        }
    return sum;
}
extern "C" __global__ void nya_train_binary(const float *a, const float *b, const float *dy,
    float *out_a, float *out_b, unsigned long long ar, unsigned long long ac,
    unsigned long long br, unsigned long long bc, unsigned long long rows, unsigned long long cols,
    unsigned operation, int backward)
{
    unsigned long long i = (unsigned long long)blockIdx.x*256+threadIdx.x;
    if (!backward) {
        if (i >= rows*cols) return;
        float x = a[nya_train_broadcast_index(i/cols,i%cols,ar,ac)];
        float y = b[nya_train_broadcast_index(i/cols,i%cols,br,bc)];
        out_a[i] = operation == 0 ? x+y : x*y;
    } else {
        int shared = out_a && out_a == out_b;
        if (out_a && i < ar*ac)
            out_a[i] = nya_train_binary_gradient(a,b,dy,out_a[i],i,ar,ac,br,bc,rows,cols,operation,0,shared);
        if (out_b && !shared && i < br*bc)
            out_b[i] = nya_train_binary_gradient(a,b,dy,out_b[i],i,ar,ac,br,bc,rows,cols,operation,1,0);
    }
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
