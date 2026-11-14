/* Original Fyodor experiment. Preserve RMSNorm arithmetic and row-major
   output while also producing the narrow-batch matrix input layout. */
extern "C" __global__ void nya_norm_interleaved(const float *x, float *out,
    float *interleaved, const unsigned char *w, unsigned long long n,
    unsigned type, float epsilon, unsigned batch)
{
    if ((batch != 2 && batch != 4) || blockIdx.x >= batch ||
        blockDim.x != 256 || blockDim.y != 1 || blockDim.z != 1 || !n) return;
    unsigned long long base = (unsigned long long)blockIdx.x * n;
    float sum = 0;
    for (unsigned long long i = threadIdx.x; i < n; i += 256)
        sum += x[base+i] * x[base+i];
    sum = nya_sum(sum);
    __shared__ float scale;
    if (!threadIdx.x) scale = rsqrtf(sum / (float)n + epsilon);
    __syncthreads();
    for (unsigned long long i = threadIdx.x; i < n; i += 256) {
        float value = x[base+i] * scale * (w ? nya_weight(w, i, type) : 1.0f);
        out[base+i] = value;
        interleaved[i*batch+blockIdx.x] = value;
    }
}
