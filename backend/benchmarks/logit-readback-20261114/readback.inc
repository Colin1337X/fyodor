/* Experimental contiguous verification-logit readback; original Fyodor C. */
static int plan_read_logits_bank(nya_cuda_plan *p, float *output, size_t rows)
{
    const nya_llm_context *m = p->model;
    nya_cuda_context *c = p->cuda;
    size_t vocabulary = m->tokenizer.vocabulary_size;
    if (!output || !rows || rows > p->logits_batch || !vocabulary ||
        vocabulary > SIZE_MAX / sizeof(float) ||
        rows > SIZE_MAX / (vocabulary * sizeof(float))) return -1;
    size_t elements = rows * vocabulary;
    ++p->stats.synchronizations;
    if (c->stream_sync(c->stream) != CUDA_SUCCESS) return -1;
    ++p->stats.downloads;
    if (c->download(output, p->logits, elements * sizeof(float)) != CUDA_SUCCESS) return -1;
    for (size_t i = 0; i < elements; ++i) if (!isfinite(output[i])) return -1;
    for (size_t row = 0; row < rows; ++row)
        for (size_t i = 0; i < m->tokenizer.suppressed_count; ++i)
            output[row * vocabulary + m->tokenizer.suppressed_tokens[i]] = -INFINITY;
    return 0;
}
