/* Native training derivatives. Each output has one writer, so accumulation is
   stream ordered without floating-point atomics. Packed frozen weights stay
   compressed for dX. These are original Fyodor kernels. */
/* Stable hard-label log softmax: double reductions, O(rows) saved state.
   Inspired by the row/block reduction organization studied in PyTorch's
   SoftMax.cu; original implementation, preserving Fyodor masking/precision. */
extern "C" __global__ void nya_train_loss_rows(const float *x, const unsigned *labels,
    const unsigned char *mask, double *state, unsigned long long columns)
{
    unsigned lane=threadIdx.x;
    unsigned long long row=blockIdx.x, base=row*columns;
    double *saved=state+row*4;
    if (mask && !mask[row]) {
        if (!lane) { saved[0]=0; saved[1]=1; saved[2]=0; saved[3]=0; }
        return;
    }
    if ((unsigned long long)labels[row]>=columns) {
        if (!lane) { saved[0]=__int_as_float(0x7fc00000); saved[1]=__int_as_float(0x7fc00000); saved[2]=__int_as_float(0x7fc00000); saved[3]=1; }
        return;
    }
    __shared__ double values[256];
    double maximum=-1.7976931348623157e308;
    int bad=0;
    for (unsigned long long j=lane;j<columns;j+=256) {
        double v=x[base+j]; bad|=!isfinite(v); maximum=fmax(maximum,v);
    }
    if (__syncthreads_or(bad)) {
        if (!lane) { saved[0]=__int_as_float(0x7fc00000); saved[1]=__int_as_float(0x7fc00000); saved[2]=__int_as_float(0x7fc00000); saved[3]=1; }
        return;
    }
    values[lane]=maximum; __syncthreads();
    for (unsigned stride=128;stride;stride>>=1) {
        if (lane<stride) values[lane]=fmax(values[lane],values[lane+stride]);
        __syncthreads();
    }
    maximum=values[0]; __syncthreads();
    double mass=0;
    for (unsigned long long j=lane;j<columns;j+=256) mass+=exp((double)x[base+j]-maximum);
    values[lane]=mass; __syncthreads();
    for (unsigned stride=128;stride;stride>>=1) {
        if (lane<stride) values[lane]+=values[lane+stride];
        __syncthreads();
    }
    if (!lane) {
        saved[0]=maximum; saved[1]=values[0];
        saved[2]=((double)x[base+labels[row]]-maximum)-log(values[0]); saved[3]=1;
    }
}
extern "C" __global__ void nya_train_loss_reduce(double *state, float *y,
    unsigned long long rows, unsigned operation)
{
    double total=0; unsigned long long active=0;
    for (unsigned long long row=0;row<rows;++row) {
        if (state[row*4+3]!=0) { ++active; total+=state[row*4+2]; }
    }
    double coefficient=active ? (operation ? 1.0 : -1.0/(double)active) : __int_as_float(0x7fc00000);
    state[rows*4]=coefficient; *y=(float)(total*coefficient);
}
extern "C" __global__ void nya_train_loss_back(const float *x, const unsigned *labels,
    const double *state, const float *dy, float *dx, unsigned long long rows, unsigned long long columns)
{
    unsigned long long row=blockIdx.x,base=row*columns;
    if (state[row*4+3]==0) return;
    double maximum=state[row*4],mass=state[row*4+1],scale=(double)*dy*state[rows*4];
    for (unsigned long long j=threadIdx.x;j<columns;j+=256) {
        double p=exp((double)x[base+j]-maximum)/mass;
        float contribution=(float)(scale*((j==labels[row]?1.0:0.0)-p));
        dx[base+j]=__fadd_rn(dx[base+j],contribution);
    }
}
extern "C" __global__ void nya_train_dpo(const float *chosen, const float *rejected,
    float *y, double *state, double reference_chosen, double reference_rejected, double beta)
{
    double margin=beta*(((double)*chosen-*rejected)-(reference_chosen-reference_rejected));
    if (!isfinite(margin)) { *y=__int_as_float(0x7fc00000); *state=__int_as_float(0x7fc00000); return; }
    *y=(float)(fmax(-margin,0.0)+log1p(exp(-fabs(margin))));
    double e=exp(-fabs(margin));
    *state=-beta*(margin>=0 ? e/(1.0+e) : 1.0/(1.0+e));
}
extern "C" __global__ void nya_train_dpo_back(const double *state, const float *dy,
    float *dchosen, float *drejected)
{
    float contribution=(float)((double)(*dy)*(*state));
    if (dchosen) *dchosen=__fadd_rn(*dchosen,contribution);
    if (drejected) *drejected=__fsub_rn(*drejected,contribution);
}
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

/* Attention saves O(tokens*heads) normalization state. Backward recomputes
   probabilities into a bounded 16-query workspace, then gives every gradient
   element one writer. Tile order preserves row/head accumulation across tiles. */
__device__ __forceinline__ bool nya_train_attn_visible(unsigned long long row, unsigned long long col,
    unsigned long long window, const unsigned *groups)
{
    if (window && row>=window && col<row-window+1) return false;
    return col<=row || (window && groups && groups[row] && groups[row]==groups[col]);
}
__device__ __forceinline__ float nya_train_attn_score(const float *q,const float *k,
    unsigned long long row,unsigned long long head,unsigned long long col,
    unsigned long long heads,unsigned long long kv_heads,unsigned long long dim,double scale)
{
    unsigned long long qi=(row*heads+head)*dim,ki=(col*kv_heads+head/(heads/kv_heads))*dim;
    double dot=0;
    for (unsigned long long j=0;j<dim;++j) dot+=(double)q[qi+j]*(double)k[ki+j];
    double score=dot*scale;
    /* Check before the F32 cast: a slight overflow can round back to FLT_MAX. */
    if (!isfinite(score) || fabs(score)>(double)__int_as_float(0x7f7fffff)) return __int_as_float(0x7fc00000);
    return (float)score;
}
extern "C" __global__ void nya_train_attention(const float *q,const float *k,const float *v,
    const unsigned *groups,float *out,double *state,unsigned long long n,unsigned long long heads,
    unsigned long long kv_heads,unsigned long long dim,unsigned long long window,double scale)
{
    __shared__ double reduction[256];
    __shared__ float probability[256];
    unsigned lane=threadIdx.x;
    unsigned long long row=blockIdx.x/heads,head=blockIdx.x%heads,kh=head/(heads/kv_heads);
    unsigned long long first=window && row>=window?row-window+1:0;
    unsigned long long end=window && groups?n:row+1;
    double maximum=-(double)__int_as_float(0x7f800000);
    int bad=0;
    for (unsigned long long col=first+lane;col<end;col+=256) if (nya_train_attn_visible(row,col,window,groups)) {
        float score=nya_train_attn_score(q,k,row,head,col,heads,kv_heads,dim,scale);
        if (!isfinite(score)) bad=1;
        else if (score>maximum) maximum=score;
    }
    if (__syncthreads_or(bad)) {
        if (!lane) { state[2*blockIdx.x]=__int_as_float(0x7fc00000); state[2*blockIdx.x+1]=__int_as_float(0x7fc00000); }
        for (unsigned long long j=lane;j<dim;j+=256) out[((unsigned long long)blockIdx.x)*dim+j]=__int_as_float(0x7fc00000);
        return;
    }
    reduction[lane]=maximum; __syncthreads();
    for (unsigned stride=128;stride;stride>>=1) {
        if (lane<stride && reduction[lane+stride]>reduction[lane]) reduction[lane]=reduction[lane+stride];
        __syncthreads();
    }
    maximum=reduction[0];
    double mass=0;
    for (unsigned long long col=first+lane;col<end;col+=256) if (nya_train_attn_visible(row,col,window,groups)) {
        float score=nya_train_attn_score(q,k,row,head,col,heads,kv_heads,dim,scale);
        mass+=(float)exp((double)score-maximum);
    }
    /* Every lane has read the shared maximum before overwriting reduction. */
    __syncthreads(); reduction[lane]=mass; __syncthreads();
    for (unsigned stride=128;stride;stride>>=1) {
        if (lane<stride) reduction[lane]+=reduction[lane+stride];
        __syncthreads();
    }
    mass=reduction[0];
    if (!lane) { state[2*blockIdx.x]=maximum; state[2*blockIdx.x+1]=mass; }
    for (unsigned long long base_dim=0;base_dim<dim;base_dim+=256) {
        unsigned long long j=base_dim+lane;
        double sum=0;
        for (unsigned long long base=first;base<end;base+=256) {
            unsigned long long col=base+lane;
            float p=0;
            if (col<end && nya_train_attn_visible(row,col,window,groups)) {
                float score=nya_train_attn_score(q,k,row,head,col,heads,kv_heads,dim,scale);
                p=(float)((float)exp((double)score-maximum)/mass);
            }
            probability[lane]=p; __syncthreads();
            if (j<dim) for (unsigned t=0;t<256 && base+t<end;++t)
                sum+=(double)probability[t]*v[((base+t)*kv_heads+kh)*dim+j];
            __syncthreads();
        }
        if (j<dim) out[(row*heads+head)*dim+j]=(float)sum;
    }
}
extern "C" __global__ void nya_train_attention_prepare(const float *q,const float *k,const float *v,
    const unsigned *groups,const float *dy,const double *state,double *workspace,
    unsigned long long n,unsigned long long heads,unsigned long long kv_heads,unsigned long long dim,
    unsigned long long window,double scale,unsigned long long start,int scores)
{
    __shared__ double reduction[256];
    unsigned lane=threadIdx.x;
    unsigned long long row=start+blockIdx.x/heads,head=blockIdx.x%heads,kh=head/(heads/kv_heads);
    unsigned long long qi=(row*heads+head)*dim,base=(unsigned long long)blockIdx.x*n;
    double maximum=state[2*(row*heads+head)],mass=state[2*(row*heads+head)+1],average=0;
    for (unsigned long long col=lane;col<n;col+=256) {
        float p=0; double dp=0;
        if (nya_train_attn_visible(row,col,window,groups)) {
            float score=nya_train_attn_score(q,k,row,head,col,heads,kv_heads,dim,scale);
            p=(float)((float)exp((double)score-maximum)/mass);
            if (scores && p!=0) {
                unsigned long long ki=(col*kv_heads+kh)*dim;
                for (unsigned long long j=0;j<dim;++j) dp+=(double)dy[qi+j]*(double)v[ki+j];
                average+=(double)p*dp;
            }
        }
        workspace[2*(base+col)]=p; workspace[2*(base+col)+1]=dp;
    }
    reduction[lane]=average; __syncthreads();
    for (unsigned stride=128;stride;stride>>=1) {
        if (lane<stride) reduction[lane]+=reduction[lane+stride];
        __syncthreads();
    }
    average=reduction[0];
    for (unsigned long long col=lane;col<n;col+=256) {
        double p=workspace[2*(base+col)],dp=workspace[2*(base+col)+1];
        workspace[2*(base+col)+1]=p!=0?p*(dp-average)*scale:0;
    }
}
extern "C" __global__ void nya_train_attention_dq(const float *k,const double *workspace,float *dq,
    unsigned long long n,unsigned long long heads,unsigned long long kv_heads,unsigned long long dim,unsigned long long start)
{
    unsigned long long row=start+blockIdx.x/heads,head=blockIdx.x%heads,kh=head/(heads/kv_heads);
    unsigned long long base=(unsigned long long)blockIdx.x*n;
    for (unsigned long long j=threadIdx.x;j<dim;j+=256) {
        unsigned long long qi=(row*heads+head)*dim+j;
        float sum=dq[qi];
        for (unsigned long long col=0;col<n;++col) if (workspace[2*(base+col)]!=0) {
            double ds=workspace[2*(base+col)+1];
            sum=__fadd_rn(sum,(float)(ds*(double)k[(col*kv_heads+kh)*dim+j]));
        }
        dq[qi]=sum;
    }
}
extern "C" __global__ void nya_train_attention_dkv(const float *q,const float *dy,const double *workspace,
    float *dk,float *dv,unsigned long long n,unsigned long long heads,unsigned long long kv_heads,
    unsigned long long dim,unsigned long long start,unsigned long long count)
{
    unsigned long long col=blockIdx.x/kv_heads,kh=blockIdx.x%kv_heads,group=heads/kv_heads;
    for (unsigned long long j=threadIdx.x;j<dim;j+=256) {
        unsigned long long ki=(col*kv_heads+kh)*dim+j;
        float sum_k=dk?dk[ki]:0,sum_v=dv?dv[ki]:0;
        for (unsigned long long r=0;r<count;++r) for (unsigned long long h=kh*group;h<(kh+1)*group;++h) {
            unsigned long long index=(r*heads+h)*n+col,qi=((start+r)*heads+h)*dim+j;
            float p=(float)workspace[2*index];
            if (p==0) continue;
            if (dk) sum_k=__fadd_rn(sum_k,(float)(workspace[2*index+1]*(double)q[qi]));
            if (dv) sum_v=__fadd_rn(sum_v,__fmul_rn(p,dy[qi]));
        }
        if (dk) dk[ki]=sum_k;
        if (dv) dv[ki]=sum_v;
    }
}
