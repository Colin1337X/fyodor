/* Original Fyodor small-batch scheduling. Reuse the native row reduction
   exactly; grid Y selects independent input vectors. No padded 32-token tile,
   shared barriers, or full-weight expansion. The native GEMM remains fallback. */
#define NYA_SMALL_BATCH(TYPE) \
extern "C" __global__ void nya_small_batch_##TYPE(const unsigned char *w, \
 const float *x,float *y,unsigned long long columns,unsigned long long stride, \
 unsigned rows,unsigned batch) { \
 if(blockIdx.y<batch) nya_row_dot_at(w,x+(unsigned long long)blockIdx.y*columns, \
 y+(unsigned long long)blockIdx.y*rows,columns,stride,rows,TYPE, \
 (unsigned long long)blockIdx.x*8+(threadIdx.x>>5)); }
NYA_SMALL_BATCH(0) NYA_SMALL_BATCH(1) NYA_SMALL_BATCH(2)
NYA_SMALL_BATCH(8) NYA_SMALL_BATCH(12) NYA_SMALL_BATCH(14) NYA_SMALL_BATCH(30)
#undef NYA_SMALL_BATCH
