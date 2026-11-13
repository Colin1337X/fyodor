#include "llm_internal.h"
#include "file.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Explicitly prove dispatch/fallback in addition to comparing every logit. */
int main(int argc, char **argv)
{
    if (argc != 2) return 2;
    char *end;
    unsigned long parsed = strtoul(argv[1], &end, 10);
    if (*end || parsed < 3 || parsed > 8194) return 2;
    size_t count = (size_t)parsed;
    const char *path = "E:/models/Gemma4-12B-QAT-Uncensored-HauhauCS-Balanced-Q4_K_M.gguf";
    FILE *file = nya_file_open_read(path);
    uint64_t bytes = 0;
    if (!file) return 2;
    int failed = nya_file_regular_size(file, &bytes);
    fclose(file);
    if (failed) return 2;
    char error[512] = {0};
    nya_llm_context *model = NULL;
    if (nya_llm_load(path, bytes, &model, error, sizeof(error))) { puts(error); return 2; }
    size_t vocabulary = model->tokenizer.vocabulary_size;
    float *reference = malloc(vocabulary * sizeof(*reference));
    uint32_t *tokens = malloc(count * sizeof(*tokens));
    if (!reference || !tokens) { free(reference); free(tokens); nya_llm_free(model); return 2; }
    for (size_t i = 0; i < count; ++i) tokens[i] = 20 + (uint32_t)(i % 128);
    size_t baseline_kv=0;
    float *snapshots[2]={NULL,NULL};size_t snapshot_elements=0;
    float *reset_reference=malloc(vocabulary*sizeof(float));
    if(!reset_reference) return 2;
    for (unsigned mode = 0; mode < 2 && !failed; ++mode) {
        if (_putenv_s("NYA_CUDA_RING_KV", mode ? "1" : "0")) { failed = 1; break; }
        nya_llm_session *session = nya_llm_session_create(model, count, error, sizeof(error));
        if (!session) { puts(error); failed = 1; break; }
        if (strcmp(nya_llm_session_execution(session), "resident") ||
            nya_llm_session_run(session, tokens, count - 2, 0) ||
            nya_llm_session_run(session, tokens + count - 2, 1, 1) ||
            nya_llm_session_run(session, tokens + count - 1, 1, 1)) {
            puts("Resident inference failed"); failed = 1;
        } else {
            nya_compute_stats stats;
            nya_llm_session_stats(session, &stats);
            unsigned long long launches = (unsigned long long)stats.kernel_launches;
            printf("mode=%u tokens=%zu kernels=%llu weights=%zu kv=%zu scratch=%zu\n",
                mode, count, launches, stats.weights_bytes, stats.kv_bytes, stats.scratch_bytes);
            if(!mode) baseline_kv=stats.kv_bytes;
            else if(count>1535 && stats.kv_bytes>=baseline_kv) failed=1;
            const float *actual = nya_llm_session_logits(session);
            double maximum = 0, scaled_maximum = 0, squared = 0;
            for (size_t j = 0; j < vocabulary; ++j) {
                int suppressed = 0;
                for (size_t k = 0; k < model->tokenizer.suppressed_count; ++k)
                    if (model->tokenizer.suppressed_tokens[k] == j) { suppressed = 1; break; }
                if (suppressed) {
                    if (actual[j] != -INFINITY || (mode && reference[j] != -INFINITY)) failed = 1;
                    if (!mode) reference[j] = actual[j];
                    continue;
                }
                if (!isfinite(actual[j]) || (mode && !isfinite(reference[j]))) { failed = 1; continue; }
                if (!mode) { reference[j] = actual[j]; continue; }
                double difference = fabs((double)actual[j] - reference[j]);
                double scaled = difference / (1 + fabs(reference[j]));
                if (difference > maximum) maximum = difference;
                if (scaled > scaled_maximum) scaled_maximum = scaled;
                squared += difference * difference;
            }
            if (mode) {
                printf("max_abs=%.9g max_scaled=%.9g rms=%.9g\n", maximum, scaled_maximum, sqrt(squared / vocabulary));
                if (maximum != 0) failed = 1;
            }
            fflush(stdout);
            /* Snapshot crosses the ring wrap, retaining absolute sequence order. */
            size_t width=model->layers[0].head_dimension*model->layers[0].kv_head_count;
            size_t rows=count<1024?count:1024,start=count-rows;
            snapshot_elements=rows*width;
            if(!mode) for(int i=0;i<2;++i) snapshots[i]=malloc(snapshot_elements*sizeof(float));
            float *k=malloc(snapshot_elements*sizeof(float)),*v=malloc(snapshot_elements*sizeof(float));
            if(!k || !v || !snapshots[0] || !snapshots[1] ||
                nya_llm_session_read_kv(session,0,start,rows,k,v,snapshot_elements)) failed=1;
            else if(!mode) {memcpy(snapshots[0],k,snapshot_elements*sizeof(float));memcpy(snapshots[1],v,snapshot_elements*sizeof(float));}
            else if(memcmp(snapshots[0],k,snapshot_elements*sizeof(float)) || memcmp(snapshots[1],v,snapshot_elements*sizeof(float))) failed=1;
            if(mode && count>1535 && nya_llm_session_read_kv(session,0,0,1,k,v,width)==0) failed=1;
            free(k);free(v);
            nya_llm_session_reset(session);
            if(nya_llm_session_run(session,tokens,3,1)) failed=1;
            else {
                const float *reset=nya_llm_session_logits(session);
                if(!mode) memcpy(reset_reference,reset,vocabulary*sizeof(float));
                else if(memcmp(reset_reference,reset,vocabulary*sizeof(float))) failed=1;
            }
            printf("snapshot_and_graph_reset mode=%u passed=%d\n",mode,!failed);fflush(stdout);
        }
        nya_llm_session_free(session);
    }
    free(snapshots[0]);free(snapshots[1]);free(reset_reference);
    free(tokens); free(reference); nya_llm_free(model);
    return failed;
}
