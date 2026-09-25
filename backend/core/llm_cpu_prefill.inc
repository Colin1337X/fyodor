/* Optimized dense host-graph prefill. The transformer equations remain the same, but
   execution visits layers before tokens so each projection can reuse weight
   rows across a prompt tile. This code never enters the autograd graph. */
static void nya_llm_batch_projection(const nya_llm_context *c, float *out,
    const nya_llm_tensor *w, const float *in, size_t columns, size_t rows, size_t batch)
{
    if (nya_compute_matmul_typed(c->compute, w->data, rows, columns, w->type, in, out, batch) == 0) return;
    for (size_t i = 0; i < batch; ++i) nya_llm_matvec(NULL, out+i*rows, w, in+i*columns, columns, rows);
}

static int nya_llm_cpu_prefill(const nya_llm_context *c, nya_llm_run_state *s,
    const uint32_t *tokens, size_t count, size_t position)
{
    if (!c->compute || s->reference_only || s->device_plan ||
        c->is_assistant || c->expert_count || c->per_layer_embedding_length || count < 2 || !s->prefill_batch) return 1;
    const char *profile_setting = getenv("NYA_CPU_PROFILE");
    int profile = profile_setting && !strcmp(profile_setting,"1");
    uint64_t started = profile ? nya_runtime_monotonic_milliseconds() : 0, matrix_ms = 0, attention_ms = 0;
#define PROJECTION(...) do { uint64_t before_ = profile ? nya_runtime_monotonic_milliseconds() : 0; nya_llm_batch_projection(__VA_ARGS__); if (profile) matrix_ms += nya_runtime_monotonic_milliseconds()-before_; } while (0)
    size_t batch = s->prefill_batch;
    if (batch > count) batch = count;
    size_t d = c->embedding_length;
    float **buffers = s->prefill_buffers;
    float *x = buffers[0], *norm = buffers[1], *residual = buffers[2];
    float *query = buffers[3], *gate = buffers[4], *up = buffers[5];
    for (size_t base = 0; base < count; ) {
        size_t n = count - base < batch ? count - base : batch;
        size_t start = position + base;
        for (size_t t = 0; t < n; ++t) {
            nya_llm_copy_row(x+t*d, c->token_embedding, tokens[base+t], d);
            if (c->is_gemma) for (size_t j = 0; j < d; ++j) x[t*d+j] *= sqrtf((float)d);
        }
        for (size_t layer = 0; layer < c->block_count; ++layer) {
            const nya_llm_layer *w = &c->layers[layer];
            size_t hd = w->head_dimension, heads = w->head_count, kvheads = w->kv_head_count;
            size_t qdim = heads*hd, kvdim = kvheads*hd, h = w->hidden_length;
            size_t offset = w->cache_offset * s->sequence_capacity;
            float *keys = s->key_cache + offset, *values = s->value_cache + offset;
            for (size_t t = 0; t < n; ++t) nya_llm_rms_norm(norm+t*d, x+t*d, w->attention_norm, d, c->norm_epsilon);
            PROJECTION(c, query, w->query, norm, d, qdim, n);
            if (w->kv_source == layer) {
                PROJECTION(c, keys+start*kvdim, w->key, norm, d, kvdim, n);
                if (w->value) PROJECTION(c, values+start*kvdim, w->value, norm, d, kvdim, n);
                else memcpy(values+start*kvdim, keys+start*kvdim, n*kvdim*sizeof(float));
            }
            uint64_t attention_started = profile ? nya_runtime_monotonic_milliseconds() : 0;
            for (size_t t = 0; t < n; ++t) {
                size_t pos = start+t;
                size_t rope_offset = w->sliding_window ? s->rope_stride : 0;
                for (size_t j = 0; j < hd/2; ++j) {
                    float angle = (float)pos*s->rope_frequencies[rope_offset+j];
                    s->rope_cosines[j] = cosf(angle); s->rope_sines[j] = sinf(angle);
                }
                float *q = query+t*qdim, *k = keys+pos*kvdim, *v = values+pos*kvdim;
                if (c->is_gemma) nya_llm_head_norm(q, heads, hd, w->query_norm, c->norm_epsilon);
                if (w->kv_source == layer) {
                    if (c->is_gemma) {
                        nya_llm_head_norm(k, kvheads, hd, w->key_norm, c->norm_epsilon);
                        nya_llm_head_norm(v, kvheads, hd, NULL, c->norm_epsilon);
                        nya_llm_gemma_rope(k, kvheads, hd, s->rope_cosines, s->rope_sines);
                    } else nya_llm_rope(k, kvheads, hd, s->rope_cosines, s->rope_sines);
                }
                if (c->is_gemma) nya_llm_gemma_rope(q, heads, hd, s->rope_cosines, s->rope_sines);
                else nya_llm_rope(q, heads, hd, s->rope_cosines, s->rope_sines);
            }
            nya_compute_attention attention = {query,keys,values,norm,n,heads,kvheads,hd,s->sequence_capacity,start,w->sliding_window,
                c->is_gemma ? 1 : 1/sqrtf((float)hd)};
            if (nya_compute_attention_f32(c->compute,&attention)) for (size_t t = 0; t < n; ++t) {
                size_t pos = start+t;
                float *q = query+t*qdim;
                size_t first = w->sliding_window && pos >= w->sliding_window ? pos-w->sliding_window+1 : 0;
                for (size_t head = 0; head < heads; ++head) {
                    size_t kh = head/(heads/kvheads);
                    float scale = c->is_gemma ? 1 : 1/sqrtf((float)hd);
                    for (size_t at = first; at <= pos; ++at) {
                        float dot = 0;
                        for (size_t j = 0; j < hd; ++j) dot += q[head*hd+j]*keys[at*kvdim+kh*hd+j];
                        s->attention[at-first] = dot*scale;
                    }
                    if (nya_llm_softmax(s->attention, pos-first+1)) return -1;
                    float *out = norm+t*qdim+head*hd;
                    memset(out, 0, hd*sizeof(float));
                    for (size_t at = first; at <= pos; ++at)
                        for (size_t j = 0; j < hd; ++j) out[j] += s->attention[at-first]*values[at*kvdim+kh*hd+j];
                }
            }
            if (profile) attention_ms += nya_runtime_monotonic_milliseconds()-attention_started;
            PROJECTION(c, residual, w->attention_output, norm, qdim, d, n);
            for (size_t t = 0; t < n; ++t) {
                if (c->is_gemma) nya_llm_rms_norm(residual+t*d, residual+t*d, w->attention_post_norm, d, c->norm_epsilon);
                for (size_t j = 0; j < d; ++j) x[t*d+j] += residual[t*d+j];
                nya_llm_rms_norm(norm+t*d, x+t*d, w->feed_forward_norm, d, c->norm_epsilon);
            }
            PROJECTION(c, gate, w->feed_forward_gate, norm, d, h, n);
            PROJECTION(c, up, w->feed_forward_up, norm, d, h, n);
            nya_llm_gated_activation(gate, up, n*h, c->is_gemma);
            PROJECTION(c, residual, w->feed_forward_down, gate, h, d, n);
            for (size_t t = 0; t < n; ++t) {
                if (c->is_gemma) nya_llm_rms_norm(residual+t*d, residual+t*d, w->ffn_post_norm, d, c->norm_epsilon);
                float scale = w->output_scale ? nya_llm_tensor_value(w->output_scale, 0) : 1;
                for (size_t j = 0; j < d; ++j) x[t*d+j] = (x[t*d+j]+residual[t*d+j])*scale;
            }
        }
        memcpy(s->activation, x+(n-1)*d, d*sizeof(float));
        base += n;
    }
    nya_llm_rms_norm(s->normalized, s->activation, c->output_norm, d, c->norm_epsilon);
    nya_llm_matvec(c->compute, s->logits, c->output, s->normalized, d, c->tokenizer.vocabulary_size);
    for (size_t i = 0; i < c->tokenizer.vocabulary_size; ++i) {
        if (c->final_logit_softcap > 0) s->logits[i] = c->final_logit_softcap*tanhf(s->logits[i]/c->final_logit_softcap);
        if (!isfinite(s->logits[i])) return -1;
    }
    for (size_t i = 0; i < c->tokenizer.suppressed_count; ++i) s->logits[c->tokenizer.suppressed_tokens[i]] = -INFINITY;
    if (profile) fprintf(stderr,"CPU prefill profile: tokens=%zu total_ms=%llu projections_ms=%llu attention_rope_ms=%llu\n",count,
        (unsigned long long)(nya_runtime_monotonic_milliseconds()-started),(unsigned long long)matrix_ms,(unsigned long long)attention_ms);
#undef PROJECTION
    return 0;
}
