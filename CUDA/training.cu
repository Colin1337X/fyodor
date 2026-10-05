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
/* Token maps are validated, sorted and sealed at the explicit upload boundary.
   The backward grid visits only unique referenced rows. Each element has one
   writer and adds occurrences in token order, preserving CPU F32 rounding. */
extern "C" __global__ void nya_train_embedding(const unsigned char *table, const unsigned *ids,
    float *out, unsigned long long columns, unsigned long long row_bytes, unsigned type)
{
    unsigned long long token=blockIdx.x;
    const unsigned char *row=table+(unsigned long long)ids[token]*row_bytes;
    for (unsigned long long j=threadIdx.x;j<columns;j+=256)
        out[token*columns+j]=nya_weight(row,j,type);
}
extern "C" __global__ void nya_train_embedding_back(const float *dy, const unsigned *map, float *dt,
    unsigned long long columns, unsigned long long count, unsigned long long groups)
{
    unsigned long long group=blockIdx.x;
    const unsigned *keys=map+count, *offsets=keys+groups, *positions=offsets+groups+1;
    unsigned long long base=(unsigned long long)keys[group]*columns;
    for (unsigned long long j=threadIdx.x;j<columns;j+=256) {
        float sum=dt[base+j];
        for (unsigned long long k=offsets[group];k<offsets[group+1];++k)
            sum+=dy[(unsigned long long)positions[k]*columns+j];
        dt[base+j]=sum;
    }
}
/* One row per block, fixed-order double reduction, with saved inverse RMS.
   This keeps extreme finite F32 squares representable and avoids a separate
   norm recomputation in the per-column weight-gradient kernel. */
extern "C" __global__ void nya_train_rms(const float *x, const float *weight, float *y,
    double *inverse, unsigned long long columns, double epsilon)
{
    __shared__ double sums[256];
    unsigned lane = threadIdx.x;
    unsigned long long base = (unsigned long long)blockIdx.x*columns;
    double sum = 0;
    for (unsigned long long j = lane; j < columns; j += 256) { double v = x[base+j]; sum += v*v; }
    sums[lane] = sum;
    __syncthreads();
    for (unsigned stride = 128; stride; stride >>= 1) {
        if (lane < stride) sums[lane] += sums[lane+stride];
        __syncthreads();
    }
    double scale = 1.0/sqrt(sums[0]/(double)columns+epsilon);
    if (!lane) inverse[blockIdx.x] = scale;
    for (unsigned long long j = lane; j < columns; j += 256)
        y[base+j] = (float)((double)x[base+j]*scale*(weight ? weight[j] : 1.0));
}
extern "C" __global__ void nya_train_rms_dx(const float *x, const float *weight, const float *dy,
    const double *inverse, float *dx, unsigned long long columns)
{
    __shared__ double sums[256];
    unsigned lane = threadIdx.x;
    unsigned long long base = (unsigned long long)blockIdx.x*columns;
    double dot = 0;
    for (unsigned long long j = lane; j < columns; j += 256)
        dot += (double)dy[base+j]*(weight ? weight[j] : 1.0)*x[base+j];
    sums[lane] = dot;
    __syncthreads();
    for (unsigned stride = 128; stride; stride >>= 1) {
        if (lane < stride) sums[lane] += sums[lane+stride];
        __syncthreads();
    }
    double scale = inverse[blockIdx.x];
    for (unsigned long long j = lane; j < columns; j += 256)
        dx[base+j] += (float)(scale*((double)dy[base+j]*(weight ? weight[j] : 1.0) -
            (double)x[base+j]*scale*scale*sums[0]/(double)columns));
}
/* One writer per weight, ordered F32 accumulation across rows as on CPU.
   A shared dx/dweight is safe for one row: stream order adds dx before dw. */
extern "C" __global__ void nya_train_rms_dw(const float *x, const float *dy, const double *inverse,
    float *dw, unsigned long long rows, unsigned long long columns)
{
    unsigned long long j = (unsigned long long)blockIdx.x*256+threadIdx.x;
    if (j >= columns) return;
    float sum = dw[j];
    for (unsigned long long row = 0; row < rows; ++row)
        sum += (float)((double)dy[row*columns+j]*x[row*columns+j]*inverse[row]);
    dw[j] = sum;
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
    /* Preserve CPU backward's F32 product/addition order, including the
       gradient already present from an earlier branch or microbatch. */
    float sum[4];
    #pragma unroll
    for (unsigned slot = 0; slot < 4; ++slot)
        sum[slot] = column < inputs && token+slot*8 < tokens ?
            dx[(unsigned long long)(token+slot*8)*inputs+column] : 0;
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
                if (base+k < outputs)
                    sum[slot] = __fadd_rn(sum[slot],__fmul_rn(weight,gradients[threadIdx.y+slot*8][k]));
        }
        __syncthreads();
    }
    if (column < inputs) {
        #pragma unroll
        for (unsigned slot = 0; slot < 4; ++slot)
            if (token+slot*8 < tokens) dx[(unsigned long long)(token+slot*8)*inputs+column] = sum[slot];
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
    float sum[4];
    #pragma unroll
    for (unsigned slot = 0; slot < 4; ++slot)
        sum[slot] = column < inputs && row+slot*8 < outputs ?
            dw[(unsigned long long)(row+slot*8)*inputs+column] : 0;
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
                if (base+k < tokens)
                    sum[slot] = __fadd_rn(sum[slot],__fmul_rn(gradients[threadIdx.y+slot*8][k],input));
        }
        __syncthreads();
    }
    if (column < inputs) {
        #pragma unroll
        for (unsigned slot = 0; slot < 4; ++slot)
            if (row+slot*8 < outputs) dw[(unsigned long long)(row+slot*8)*inputs+column] = sum[slot];
    }
}

/* One thread owns each pair. Double trigonometry and intermediate products
   preserve the CPU graph contract; only storage and gradient addition are F32.
   No atomics, saved activation, or synchronization is needed for the adjoint. */
extern "C" __global__ void nya_train_rope(const float *source, const float *frequencies, float *out,
    unsigned long long pairs, unsigned long long columns, unsigned long long dimension,
    int split_half, int backward)
{
    unsigned long long pair=(unsigned long long)blockIdx.x*256+threadIdx.x;
    if (pair>=pairs) return;
    unsigned long long half=dimension/2, j=pair%half;
    unsigned long long base=(pair/half)*dimension;
    unsigned long long i=base+(split_half ? j : 2*j), k=i+(split_half ? half : 1);
    double angle=(double)(base/columns)*(double)frequencies[j];
    double c=cos(angle), s=sin(angle), a=source[i], b=source[k];
    float first=(float)(backward ? a*c+b*s : a*c-b*s);
    float second=(float)(backward ? b*c-a*s : b*c+a*s);
    if (backward) { out[i]=__fadd_rn(out[i],first); out[k]=__fadd_rn(out[k],second); }
    else { out[i]=first; out[k]=second; }
}
