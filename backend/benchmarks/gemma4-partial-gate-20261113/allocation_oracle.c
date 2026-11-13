#include "llm_internal.h"
#include "file.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Explicitly prove dispatch/fallback in addition to comparing every logit. */
int main(int argc, char **argv)
{
    if (argc != 3) return 2;
    char *end;
    unsigned long parsed = strtoul(argv[1], &end, 10);
    if (*end || parsed < 3 || parsed > 8194) return 2;
    size_t count = (size_t)parsed;
    int fused = !strcmp(argv[2], "fused");
    int partial = !strcmp(argv[2], "partial");
    if (!fused && !partial && strcmp(argv[2], "fallback")) return 2;
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
    unsigned long long baseline = 0;
    size_t baseline_weights = 0;
    for (unsigned mode = 0; mode < 2 && !failed; ++mode) {
        if (_putenv_s("NYA_CUDA_GEMMA_GATE", mode ? "1" : "0")) { failed = 1; break; }
        if (mode && !strcmp(argv[2], "fallback")) {
            if (_putenv_s("NYA_CUDA_FAIL_ALLOCATION_AFTER", "1")) { failed = 1; break; }
        }
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
            if (!mode) { baseline = launches; baseline_weights = stats.weights_bytes; }
            else {
                unsigned long long expected = fused ? (count == 3 ? 288 : 192) : 0;
                if (partial) {
                    size_t pair_bytes = 2 * 15360 * 2400;
                    size_t extra = stats.weights_bytes > baseline_weights ? stats.weights_bytes - baseline_weights : 0;
                    size_t layers = extra / pair_bytes;
                    if (!layers || layers >= 48 || extra % pair_bytes) failed = 1;
                    expected = (unsigned long long)layers * 2 * (count == 3 ? 3 : 2);
                    printf("partial_layers=%zu extra_bytes=%zu expected_saved=%llu\n", layers, extra, expected);
                }
                if (baseline < launches || baseline - launches != expected) {
                    printf("Unexpected launch reduction; expected %llu\n", expected); failed = 1;
                }
            }
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
                if (scaled_maximum > 0.001) failed = 1;
            }
            fflush(stdout);
        }
        nya_llm_session_free(session);
    }
    free(tokens); free(reference); nya_llm_free(model);
    return failed;
}
