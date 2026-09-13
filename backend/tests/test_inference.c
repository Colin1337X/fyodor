#include "llm_internal.h"
#include "file.h"
#include "pretraining.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int check_cache_snapshot(nya_llm_session *candidate, nya_llm_session *reference,
    const nya_llm_context *model, size_t position, size_t count)
{
    for (size_t layer = 0; layer < model->block_count; ++layer) {
        size_t n = count*model->layers[layer].head_dimension*model->layers[layer].kv_head_count;
        if (n > (SIZE_MAX/sizeof(float)-2)/4) return -1;
        float *buffer = malloc((4*n+2)*sizeof(float));
        if (!buffer) return -1;
        buffer[0] = buffer[4*n+1] = 12345;
        int result = nya_llm_session_read_kv(reference,layer,position,count,buffer+1,buffer+1+n,n) ||
            nya_llm_session_read_kv(candidate,layer,position,count,buffer+1+2*n,buffer+1+3*n,n);
        for (size_t i = 0; !result && i < 2*n; ++i) {
            float a = buffer[1+i], b = buffer[1+2*n+i];
            if (!isfinite(a) || !isfinite(b) || fabs((double)a-b) > 0.00015*(1+fabs(a))) result = 1;
        }
        if (buffer[0] != 12345 || buffer[4*n+1] != 12345) result = 1;
        /* Wrong shape, layer and valid-prefix bounds reject before any store. */
        if (nya_llm_session_read_kv(candidate,layer,position,count,buffer,buffer+n,n+1) == 0 ||
            nya_llm_session_read_kv(candidate,SIZE_MAX,position,count,buffer,buffer+n,n) == 0 ||
            nya_llm_session_read_kv(candidate,layer,SIZE_MAX,1,buffer,buffer+n,n) == 0 || buffer[0] != 12345) result = 1;
        free(buffer);
        if (result) { fprintf(stderr,"KV snapshot differs at layer %zu position %zu\n",layer,position); return -1; }
    }
    return 0;
}

/* A reused graph must observe new token IDs and positions after reset, after
   batched prefill, and after another prefill interleaved with decode. Compare
   every resulting logit, not merely a plausible generated token sequence. */
static int check_graph_reuse(nya_llm_session *candidate, nya_llm_session *reference,
    const nya_llm_context *model, uint32_t *tokens, size_t capacity)
{
    unsigned long long expected_replays = 0;
    nya_compute_stats start;
    nya_llm_session_stats(candidate, &start);
    for (size_t pass = 0; pass < 3; ++pass) {
        nya_llm_session_reset(candidate); nya_llm_session_reset(reference);
        float untouched = 12345;
        if (nya_llm_session_read_kv(candidate,0,0,1,&untouched,&untouched,1) == 0 || untouched != 12345) return -1;
        for (size_t i = 0; i < capacity; ++i)
            tokens[i] = (uint32_t)((i*7+pass*3+1) % model->tokenizer.vocabulary_size);
        for (size_t position = 0; position < capacity;) {
            size_t prefill = pass == 2 ? 9U : 3U;
            size_t chunk = pass && (position == 0 || position == prefill+2) && capacity-position >= prefill ? prefill : 1;
            if (nya_llm_session_run(candidate, tokens+position, chunk, 0) ||
                nya_llm_session_run(reference, tokens+position, chunk, 1)) return -1;
            if (chunk == 1) ++expected_replays;
            if (check_cache_snapshot(candidate,reference,model,position,chunk)) return -1;
            const float *a = nya_llm_session_logits(reference), *b = nya_llm_session_logits(candidate);
            for (size_t i = 0; i < model->tokenizer.vocabulary_size; ++i) {
                if (a[i] == b[i]) continue;
                if (!isfinite(a[i]) || !isfinite(b[i]) || fabs((double)a[i]-b[i]) > 0.0005*(1+fabs(a[i]))) {
                    fprintf(stderr, "graph reuse pass %zu position %zu logit %zu: %.9g vs %.9g\n", pass,position,i,(double)a[i],(double)b[i]);
                    return -1;
                }
            }
            position += chunk;
        }
    }
    nya_compute_stats end;
    nya_llm_session_stats(candidate, &end);
    /* A CPU fallback or recapture on every token must not pass as reuse. */
    if (strcmp(nya_llm_session_execution(candidate), "resident") || end.graph_captures != 1 ||
        end.graph_replays-start.graph_replays != expected_replays) return -1;
    fprintf(stderr, "one graph capture; %llu checked replays across reset and prefill\n", expected_replays);
    return 0;
}

/* Compare complete logits across chunking, prefix reuse and reset. A shallow
   model copy selects reference CPU execution without duplicating mapped weights. */
static int validate_model(int argc, char **argv, unsigned window, int shared_kv)
{
    if (argc < 2) return 2;
    FILE *f = nya_file_open_read(argv[1]);
    uint64_t bytes;
    if (!f) return 1;
    int result = nya_file_regular_size(f, &bytes);
    fclose(f);
    if (result) return 1;
    nya_llm_context *m = NULL;
    nya_llm_session *reference = NULL, *candidate = NULL;
    uint32_t *tokens = NULL;
    char error[512] = {0};
    result = 1;
    if (nya_llm_load(argv[1], bytes, &m, error, sizeof(error))) goto done;
    /* Synthetic-only graph variations exercise window masking and cache alias
       handling independently of a particular tokenizer or architecture file. */
    if (window) {
        m->sliding_rope_base = m->rope_frequency_base;
        for (size_t i = 0; i < m->block_count; ++i) m->layers[i].sliding_window = window;
    }
    if (shared_kv && m->block_count == 2) {
        m->layers[1].kv_source = 0;
        m->layers[1].cache_offset = m->layers[0].cache_offset;
    }
    /* A configured accelerator without a usable device is a skipped hardware
       test. Never let transparent CPU fallback masquerade as GPU coverage. */
    const char *requested = getenv("NYA_COMPUTE");
    if (requested && (!strcmp(requested, "cuda") || !strcmp(requested, "vulkan")) &&
        strcmp(requested, nya_compute_name(m->compute))) { result = 77; goto done; }
    nya_llm_context view = *m;
    /* Diagnostic prefix-of-layers run localizes numerical drift without
       mutating the loaded model or changing production architecture binding. */
    const char *layer_limit = getenv("NYA_TEST_LAYERS");
    if (layer_limit) {
        char *end; unsigned long n = strtoul(layer_limit,&end,10);
        if (*end || !n || n > view.block_count) goto done;
        view.block_count = (uint32_t)n;
    }
    nya_llm_context cpu = view; cpu.compute = NULL;
    size_t capacity = m->context_length < 128 || getenv("NYA_TEST_ATTENTION_TILES") ? m->context_length : 128;
    if (argc > 3) { char *end; unsigned long n = strtoul(argv[3], &end, 10);
        if (*end || n < 3 || n > m->context_length || n > 4096) goto done;
        capacity = (size_t)n;
    }
    reference = nya_llm_session_create(&cpu, capacity, error, sizeof(error));
    candidate = nya_llm_session_create(&view, capacity, error, sizeof(error));
    if (!reference || !candidate) goto done;
    nya_compute_stats resources;
    nya_llm_session_stats(candidate, &resources);
    if (getenv("NYA_TEST_BLAS") && !resources.external_matmul_bytes) { result = 77; goto done; }
    if (getenv("NYA_TEST_CUTLASS") && resources.external_matmul_kind != 2) { result = 77; goto done; }
    if (getenv("NYA_TEST_NO_BLAS") && resources.external_matmul_bytes) goto done;
    if (argc > 2 && strcmp(argv[2], nya_llm_session_execution(candidate))) {
        fprintf(stderr, "Expected execution %s; got %s\n", argv[2], nya_llm_session_execution(candidate)); goto done;
    }
    tokens = malloc(capacity * sizeof(*tokens));
    if (!tokens) goto done;
    for (size_t i = 0; i < capacity; ++i) tokens[i] = (uint32_t)((17+i*13) % m->tokenizer.vocabulary_size);
    for (size_t pass = 0; pass < 3; ++pass) {
        nya_llm_session_reset(reference); nya_llm_session_reset(candidate);
        if (nya_llm_session_logits(candidate) != NULL) goto done;
        size_t count = capacity - pass;
        if (nya_llm_session_run(candidate, tokens, count, getenv("NYA_TEST_DECODE") != NULL)) goto done;
        nya_llm_session_stats(candidate, &resources);
        if (getenv("NYA_TEST_ATTENTION_TILES")) {
            unsigned width = m->layers[0].head_dimension;
            int tiled = (width == 64 || width == 128) && count >= 8;
            if (tiled != (resources.tiled_attention_calls != 0)) {
                fprintf(stderr,"Unexpected tiled attention dispatch count: %llu\n",resources.tiled_attention_calls);
                goto done;
            }
        }
        if (getenv("NYA_TEST_BLAS") && !resources.external_matmul_calls) goto done;
        if (getenv("NYA_TEST_CUTLASS") && !resources.cutlass_matmul_calls) goto done;
        for (size_t i = 0; i < count; ++i) if (nya_llm_session_run(reference, tokens+i, 1, 1)) goto done;
        if (getenv("NYA_TEST_CACHE_TRACE")) for (size_t layer = 0; layer < view.block_count; ++layer) {
            size_t width = view.layers[layer].head_dimension*view.layers[layer].kv_head_count;
            size_t n = count*width;
            if (n > SIZE_MAX/(4*sizeof(float))) goto done;
            float *buffer = malloc(n*4*sizeof(float));
            if (!buffer) goto done;
            if (nya_llm_session_read_kv(reference,layer,0,count,buffer,buffer+n,n) ||
                nya_llm_session_read_kv(candidate,layer,0,count,buffer+2*n,buffer+3*n,n)) { free(buffer); goto done; }
            for (size_t kind = 0; kind < 2; ++kind) {
                size_t differing = 0, worst = 0; double maximum = 0;
                for (size_t i = 0; i < n; ++i) {
                    double delta = fabs((double)buffer[kind*n+i]-buffer[(kind+2)*n+i]);
                    if (delta > 0) ++differing;
                    if (delta > maximum) { maximum = delta; worst = i; }
                    if (!layer &&
                        nya_llm_round_f16(buffer[kind*n+i]) != nya_llm_round_f16(buffer[(kind+2)*n+i]))
                        fprintf(stderr,"half boundary %s position=%zu lane=%zu CPU=%.12g GPU=%.12g rounded_CPU=%.9g rounded_GPU=%.9g\n",
                            kind ? "V" : "K",i/width,i%width,(double)buffer[kind*n+i],(double)buffer[(kind+2)*n+i],
                            (double)nya_llm_round_f16(buffer[kind*n+i]),(double)nya_llm_round_f16(buffer[(kind+2)*n+i]));
                }
                fprintf(stderr,"cache layer %zu %s: differing=%zu/%zu max=%.9g position=%zu lane=%zu CPU=%.9g GPU=%.9g\n",
                    layer,kind ? "V" : "K",differing,n,maximum,worst/width,worst%width,
                    (double)buffer[kind*n+worst],(double)buffer[(kind+2)*n+worst]);
            }
            free(buffer);
        }
        const float *a = nya_llm_session_logits(reference), *b = nya_llm_session_logits(candidate);
        double maximum = 0, relative = 0, squared = 0;
        size_t worst = 0;
        for (size_t i = 0; i < m->tokenizer.vocabulary_size; ++i) {
            if (a[i] == b[i]) continue; /* includes intentional -infinity masks */
            if (!isfinite(a[i]) || !isfinite(b[i])) goto done;
            double delta = fabs((double)a[i]-b[i]);
            squared += delta*delta;
            if (delta > maximum) { maximum = delta; worst = i; }
            if (delta/(1+fabs((double)a[i])) > relative) relative = delta/(1+fabs((double)a[i]));
        }
        fprintf(stderr, "prefix %zu: max_abs=%.9g rms=%.9g max_scaled=%.9g worst_logit=%zu\n",
            count, maximum, sqrt(squared/m->tokenizer.vocabulary_size), relative, worst);
        /* The optional real-model run traverses many more layers than fixtures.
           Keep its bound explicit and print both absolute and RMS error. The
           independent Gemma fixtures still enforce their existing 3e-5 bound. */
        if (relative > (argc > 3 ? 0.001 : 0.0005)) goto done;
        uint32_t invalid = m->tokenizer.vocabulary_size;
        if (nya_llm_session_run(candidate, &invalid, 1, 1) == 0 ||
            nya_llm_session_run(candidate, tokens, capacity+1, 0) == 0) goto done;
    }
    if (getenv("NYA_TEST_GRAPH") && check_graph_reuse(candidate, reference, m, tokens, capacity)) goto done;
    result = 0;
done:
    if (result == 77) fprintf(stderr, "Requested accelerator is unavailable\n");
    else if (result) fprintf(stderr, "inference session validation failed: %s\n", error);
    nya_llm_session_free(reference); nya_llm_session_free(candidate); nya_llm_free(m); free(tokens);
    return result;
}

int main(int argc, char **argv)
{
    if (argc < 2 || strcmp(argv[1], "--attention-shapes")) return validate_model(argc,argv,0,0);
    if (argc != 3 || strlen(argv[2]) > 1000) return 2;
    const unsigned widths[] = {32,64,64,128,128,130,256};
    const unsigned windows[] = {0,0,1,0,17,17,0};
    for (size_t i = 0; i < sizeof(widths)/sizeof(widths[0]); ++i) {
        nya_train_decoder_config config; nya_train_decoder_defaults(&config);
        config.embedding_length = widths[i]*2; config.feed_forward_length = widths[i]*4;
        config.head_count = 2; config.kv_head_count = 1; config.block_count = 2;
        /* The first tiled shape crosses two full 512-token prefill chunks,
           then checks decode/reset reuse with more than 1024 cached tokens. */
        config.context_length = i == 1 ? 1025 : 65; config.seed = 123+i;
        char error[512] = {0}, path[1100];
        snprintf(path,sizeof(path),"%s-%zu.gguf",argv[2],i);
        nya_train_decoder *decoder = nya_train_decoder_create(&config,error,sizeof(error));
        if (!decoder) { fprintf(stderr,"shape initialization: %s\n",error); return 1; }
        FILE *file = fopen(path,"wb");
        int result = file ? nya_train_decoder_export(decoder,file,error,sizeof(error)) : -1;
        if (file && fclose(file)) result = -1;
        nya_train_decoder_free(decoder);
        if (result) { fprintf(stderr,"shape export: %s\n",error); return 1; }
        char *args[] = {argv[0],path,"resident"};
        fprintf(stderr,"attention width=%u window=%u shared=%u\n",widths[i],windows[i],(unsigned)(i&1));
        result = validate_model(3,args,windows[i],(int)(i&1));
        if (result) return result;
    }
    return 0;
}
