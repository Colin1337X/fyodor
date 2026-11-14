/* Original Fyodor batched graph-input prototype; no production dispatch.
   Five device words preserve the resident request ABI: token zero, position,
   then tokens one through three. Update on the SAME stream, outside capture,
   before every replay. Allocate five words (the current plan allocates two).
   Host admission must validate tokens, sizes, pointers and batch in {2,4}.
   Changing batch requires a distinct graph. No host pointer is retained. */
struct nya_graph_tokens { unsigned values[4]; };

extern "C" __global__ void nya_decode_batch_request(unsigned *request,
    nya_graph_tokens tokens, unsigned position)
{
    if (blockIdx.x || threadIdx.x) return;
    request[0] = tokens.values[0];
    request[1] = position;
    request[2] = tokens.values[1];
    request[3] = tokens.values[2];
    request[4] = tokens.values[3];
}

/* Uses the existing weight decoder and exactly the ordinary gather expression.
   Kernel parameters contain only fixed pointers/shape/scale during capture;
   token IDs are fetched from the stream-ordered request on every replay. */
extern "C" __global__ void nya_gather_graph_batch(const unsigned char *weights,
    float *out, unsigned long long width, unsigned long long row_bytes,
    unsigned type, unsigned count, float scale, const unsigned *request)
{
    if (!width || (count != 2 && count != 4)) return;
    unsigned long long i = (unsigned long long)blockIdx.x * 256 + threadIdx.x;
    if (i < width * count) {
        unsigned query = (unsigned)(i / width);
        unsigned token = request[query ? query + 1 : 0];
        out[i] = nya_weight(weights + (unsigned long long)token * row_bytes,
            i % width, type) * scale;
    }
}
