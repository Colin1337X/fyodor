/* 32x32 output tiles reuse each dequantized weight across thirty-two prompt
   positions. Accumulation stays F32; compressed weights never expand to a full
   device-side floating-point model. This custom GEMM requires no cuBLAS SDK. */
__device__ __forceinline__ void nya_tile(const unsigned char *w, const float *x,
    float *y, unsigned long long columns, unsigned long long row_bytes,
    unsigned rows, unsigned batch, unsigned TYPE)
{
    __shared__ float weights[32][33], inputs[32][33];
    unsigned row = blockIdx.x*32 + threadIdx.x;
    unsigned token = blockIdx.y*32 + threadIdx.y;
    unsigned linear = threadIdx.y*32 + threadIdx.x;
    float sums[4] = {0,0,0,0};
    /* F32 exports have long dot products without quantization noise. Two
       independent chains reduce their rounding depth and expose instruction
       parallelism; TYPE is constant, so other formats keep their old kernel. */
    float precise[4][2] = {};
    for (unsigned long long k = 0; k < columns; k += 32) {
        #pragma unroll
        for (unsigned slot = 0; slot < 4; ++slot) {
            unsigned tile_row = linear/32 + slot*8, tile_col = linear%32;
            unsigned source_row = blockIdx.x*32 + tile_row;
            unsigned source_token = blockIdx.y*32 + tile_row;
            weights[tile_row][tile_col] = source_row < rows && k+tile_col < columns ?
                nya_weight(w + (unsigned long long)source_row*row_bytes,k+tile_col,TYPE) : 0;
            inputs[tile_row][tile_col] = source_token < batch && k+tile_col < columns ?
                x[(unsigned long long)source_token*columns+k+tile_col] : 0;
        }
        __syncthreads();
        #pragma unroll
        for (unsigned j = 0; j < 32; ++j) {
            float weight = weights[threadIdx.x][j];
            #pragma unroll
            for (unsigned slot = 0; slot < 4; ++slot) {
                if (TYPE == 0) precise[slot][j&1] += weight * inputs[threadIdx.y+slot*8][j];
                else sums[slot] += weight * inputs[threadIdx.y+slot*8][j];
            }
        }
        __syncthreads();
    }
    if (row < rows) {
        #pragma unroll
        for (unsigned slot = 0; slot < 4; ++slot)
            if (token+slot*8 < batch) y[(unsigned long long)(token+slot*8)*rows+row] = TYPE == 0 ?
                precise[slot][0]+precise[slot][1] : sums[slot];
    }
}
#define NYA_GEMM(TYPE) extern "C" __global__ void nya_gemm_##TYPE(const unsigned char *w, const float *x, float *y, unsigned long long c, unsigned long long s, unsigned r, unsigned b) { nya_tile(w,x,y,c,s,r,b,TYPE); }
NYA_GEMM(0) NYA_GEMM(1) NYA_GEMM(2) NYA_GEMM(8) NYA_GEMM(12) NYA_GEMM(14) NYA_GEMM(30)
#undef NYA_GEMM

/* A 64x64 output tile gives each thread a 4x4 register tile. Each shared
   operand now feeds four products before another load, doubling reuse over
   the 32x32 kernel. Quantized formats keep their ascending F32 reduction;
   F32 weights use the two shorter chains described above. Padding
   avoids shared-memory bank conflicts when lanes access distinct rows. */
__device__ __forceinline__ void nya_tile64(const unsigned char *w, const float *x,
    float *y, unsigned long long columns, unsigned long long row_bytes, unsigned rows, unsigned batch, unsigned TYPE)
{
    __shared__ float weights[64][33], inputs[64][33];
    unsigned linear = threadIdx.y*16+threadIdx.x;
    float sums[4][4] = {};
    float precise[4][4][2] = {};
    for (unsigned long long base = 0; base < columns; base += 32) {
        #pragma unroll
        for (unsigned slot = 0; slot < 8; ++slot) {
            unsigned r = linear/32+slot*8, k = linear%32;
            unsigned row = blockIdx.x*64+r, token = blockIdx.y*64+r;
            weights[r][k] = row < rows && base+k < columns ? nya_weight(w+(unsigned long long)row*row_bytes,base+k,TYPE) : 0;
            inputs[r][k] = token < batch && base+k < columns ? x[(unsigned long long)token*columns+base+k] : 0;
        }
        __syncthreads();
        #pragma unroll
        for (unsigned k = 0; k < 32; ++k) {
            float a[4], b[4];
            #pragma unroll
            for (unsigned i = 0; i < 4; ++i) { a[i] = weights[threadIdx.x+i*16][k]; b[i] = inputs[threadIdx.y+i*16][k]; }
            #pragma unroll
            for (unsigned i = 0; i < 4; ++i) {
                #pragma unroll
                for (unsigned j = 0; j < 4; ++j) {
                    if (TYPE == 0) precise[i][j][k&1] += a[i]*b[j];
                    else sums[i][j] += a[i]*b[j];
                }
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for (unsigned i = 0; i < 4; ++i) {
        unsigned row = blockIdx.x*64+threadIdx.x+i*16;
        #pragma unroll
        for (unsigned j = 0; j < 4; ++j) {
            unsigned token = blockIdx.y*64+threadIdx.y+j*16;
            if (row < rows && token < batch) y[(unsigned long long)token*rows+row] = TYPE == 0 ?
                precise[i][j][0]+precise[i][j][1] : sums[i][j];
        }
    }
}
#define NYA_GEMM64(TYPE) extern "C" __global__ void nya_gemm64_##TYPE(const unsigned char *w, const float *x, float *y, unsigned long long c, unsigned long long s, unsigned r, unsigned b) { nya_tile64(w,x,y,c,s,r,b,TYPE); }
NYA_GEMM64(0) NYA_GEMM64(1) NYA_GEMM64(2) NYA_GEMM64(8) NYA_GEMM64(12) NYA_GEMM64(14) NYA_GEMM64(30)
#undef NYA_GEMM64

/* Optional BLAS consumes one F32 matrix chunk. Adjacent threads decode adjacent
   columns, preserving the same storage conversion as the native quant kernels.
   Constant TYPE arguments are folded by NVRTC without C++ templates. */
__device__ __forceinline__ void nya_expand(const unsigned char *w, float *out,
    unsigned long long columns, unsigned long long stride, unsigned long long count, unsigned type)
{
    unsigned long long i = (unsigned long long)blockIdx.x*256 + threadIdx.x;
    if (i < count) out[i] = nya_weight(w + (i/columns)*stride, i%columns, type);
}
#define NYA_EXPAND(TYPE) extern "C" __global__ void nya_expand_##TYPE(const unsigned char *w, float *out, unsigned long long k, unsigned long long s, unsigned long long n) { nya_expand(w,out,k,s,n,TYPE); }
NYA_EXPAND(0) NYA_EXPAND(1) NYA_EXPAND(2) NYA_EXPAND(8) NYA_EXPAND(12) NYA_EXPAND(14) NYA_EXPAND(30)
#undef NYA_EXPAND

/* Decode and transpose one bounded matrix chunk in the same pass. The padded
   shared tile turns coalesced reads along quantized rows into coalesced F32
   writes along the output's rows. BLAS can then consume column-major weights
   without its transposed-A path. Both edge dimensions are independently masked;
   every value read after the barrier was written by a corresponding load. */
__device__ __forceinline__ void nya_expand_transposed(const unsigned char *w, float *out,
    unsigned long long columns, unsigned long long stride, unsigned long long count, unsigned type)
{
    __shared__ float tile[32][33];
    unsigned long long rows = count / columns, tiles = (columns+31)/32;
    unsigned long long row_base = (blockIdx.x/tiles)*32, col_base = (blockIdx.x%tiles)*32;
    unsigned lane = threadIdx.x%32, group = threadIdx.x/32;
    #pragma unroll
    for (unsigned i=0; i<4; ++i) {
        unsigned r = group+i*8;
        tile[r][lane] = row_base+r < rows && col_base+lane < columns ?
            nya_weight(w+(row_base+r)*stride,col_base+lane,type) : 0;
    }
    __syncthreads();
    #pragma unroll
    for (unsigned i=0; i<4; ++i) {
        unsigned col = group+i*8;
        if (row_base+lane < rows && col_base+col < columns)
            out[(col_base+col)*rows+row_base+lane] = tile[lane][col];
    }
}
#define NYA_EXPAND_T(TYPE) extern "C" __global__ void nya_expand_t_##TYPE(const unsigned char *w, float *out, unsigned long long k, unsigned long long s, unsigned long long n) { nya_expand_transposed(w,out,k,s,n,TYPE); }
NYA_EXPAND_T(0) NYA_EXPAND_T(1) NYA_EXPAND_T(2) NYA_EXPAND_T(8) NYA_EXPAND_T(12) NYA_EXPAND_T(14) NYA_EXPAND_T(30)
#undef NYA_EXPAND_T
