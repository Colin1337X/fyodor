/* Token-ID microbenchmark: no HTTP, tokenizer, sampling or text rendering is
   inside the timed interval. Decode feeds deterministic IDs, ignoring EOS, so
   every repetition performs exactly the requested amount of transformer work. */
#include "llm_internal.h"
#include "file.h"
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifdef _WIN32
#include <windows.h>
#else
#include <sys/utsname.h>
#endif

static double clock_seconds(void)
{
#ifdef _WIN32
    LARGE_INTEGER now, frequency;
    if (!QueryPerformanceFrequency(&frequency) || !QueryPerformanceCounter(&now)) return -1;
    return (double)now.QuadPart / (double)frequency.QuadPart;
#else
    struct timespec now;
    if (clock_gettime(CLOCK_MONOTONIC, &now) != 0) return -1;
    return (double)now.tv_sec + (double)now.tv_nsec * 1e-9;
#endif
}

/* Escape path and host strings, including control bytes, for script-safe JSON. */
static void json_string(const char *s)
{
    putchar('"');
    for (const unsigned char *p = (const unsigned char *)s; *p; ++p) {
        if (*p == '"' || *p == '\\') printf("\\%c", *p);
        else if (*p < 32) printf("\\u%04x", (unsigned)*p);
        else putchar(*p);
    }
    putchar('"');
}

static int number(const char *s, size_t *out)
{
    if (*s < '0' || *s > '9') return -1;
    char *end;
    errno = 0;
    unsigned long long n = strtoull(s, &end, 10);
    if (errno || *end || n > 1048576) return -1;
    *out = (size_t)n;
    return 0;
}

static void usage(void)
{
    puts("fyodor-bench -m MODEL [-b cpu|cuda|vulkan|rocm|mlx] [-p 512] [-n 128]\n"
         "             [-r 5] [--warmup 1] [--context 0] [--json]\n"
         "-p: prefill tokens; -n: decode steps; 0 disables either measurement.\n"
         "--context: untimed prefix for decode (0 starts with an empty KV cache).\n"
         "Warmup is a number of complete workloads, not a number of tokens.");
}

int main(int argc, char **argv)
{
    const char *path = NULL, *backend = "cpu";
    size_t pp = 512, tg = 128, reps = 5, warmup = 1, prefix = 0;
    int json = 0, status = 1;
    for (int i = 1; i < argc; ++i) {
        const char *a = argv[i];
        if (!strcmp(a, "--help")) { usage(); return 0; }
        if (!strcmp(a, "--json")) { json = 1; continue; }
        if (++i == argc) { usage(); return 2; }
        if (!strcmp(a, "-m") || !strcmp(a, "--model")) path = argv[i];
        else if (!strcmp(a, "-b") || !strcmp(a, "--backend")) backend = argv[i];
        else {
            size_t *value = !strcmp(a, "-p") ? &pp : !strcmp(a, "-n") ? &tg :
                !strcmp(a, "-r") ? &reps : !strcmp(a, "--warmup") ? &warmup :
                !strcmp(a, "--context") ? &prefix : NULL;
            if (value == NULL || number(argv[i], value) != 0) { usage(); return 2; }
        }
    }
    if (path == NULL || (!pp && !tg) || reps == 0 || reps > 1000 || warmup > 1000 ||
        !nya_compute_backend_known(backend)) { usage(); return 2; }
    uint64_t file_bytes;
    FILE *f = nya_file_open_read(path);
    if (f == NULL) { fprintf(stderr, "Cannot open model\n"); return 1; }
    int sized = nya_file_regular_size(f, &file_bytes);
    fclose(f);
    if (sized != 0) return 1;
    char error[512] = {0};
    nya_llm_context *c = NULL;
    nya_llm_session *s = NULL;
    uint32_t *tokens = NULL;
    double *samples = NULL;
    if (nya_llm_load(path, file_bytes, &c, error, sizeof(error)) != 0) goto cleanup;
    nya_compute_free(c->compute);
    c->compute = nya_compute_create_for(backend);
    if (strcmp(backend, nya_compute_name(c->compute))) {
        snprintf(error, sizeof(error), "requested backend %s is unavailable", backend); goto cleanup;
    }
    size_t capacity = pp > prefix + tg ? pp : prefix + tg;
    if (capacity > c->context_length || c->is_assistant) {
        snprintf(error, sizeof(error), "workload exceeds model context or requires an MTP target"); goto cleanup;
    }
    s = nya_llm_session_create(c, capacity, error, sizeof(error));
    tokens = (uint32_t *)malloc(capacity * sizeof(*tokens));
    samples = (double *)malloc(reps * sizeof(*samples));
    if (s == NULL || tokens == NULL || samples == NULL) goto cleanup;
    for (size_t i = 0; i < capacity; ++i) tokens[i] = (uint32_t)((i * 15485863 + 17) % c->tokenizer.vocabulary_size);
    char machine[256] = {0};
#ifdef _WIN32
    SYSTEM_INFO info;
    GetSystemInfo(&info);
    snprintf(machine, sizeof(machine), "Windows; %lu logical processors; %zu-bit", (unsigned long)info.dwNumberOfProcessors, sizeof(void *) * 8);
#else
    struct utsname info;
    if (uname(&info) == 0) snprintf(machine, sizeof(machine), "%.64s %.64s %.64s", info.sysname, info.release, info.machine);
#endif
    if (json) {
        nya_compute_stats memory; nya_llm_session_stats(s, &memory);
        printf("{\"schema_version\":1,\"model\":"); json_string(path);
        printf(",\"backend\":"); json_string(backend);
        printf(",\"machine\":"); json_string(machine);
        printf(",\"execution\":"); json_string(nya_llm_session_execution(s));
        printf(",\"device_weights_bytes\":%zu,\"device_kv_bytes\":%zu,\"device_scratch_bytes\":%zu,\"prefill_batch\":%zu",
            memory.weights_bytes, memory.kv_bytes, memory.scratch_bytes, memory.prefill_batch);
        const char *implementations[]={"native","cublas-f32","cutlass-3xtf32","rocblas-f32","mlx-f32"};
        const char *implementation=memory.external_matmul_kind<sizeof(implementations)/sizeof(implementations[0]) ?
            implementations[memory.external_matmul_kind] : "unknown";
        printf(",\"kv_type\":\"f32\",\"prefill_implementation\":\"%s\",\"external_matmul_bytes\":%zu",
            implementation, memory.external_matmul_bytes);
        printf(",\"model_bytes\":%llu,\"architecture\":\"%s\",\"layers\":%u,\"embedding\":%u,\"quant_tensor_counts\":{",
            (unsigned long long)file_bytes, c->is_gemma ? "gemma4" : "llama", c->block_count, c->embedding_length);
        int comma = 0;
        for (unsigned type = 0; type <= 30; ++type) {
            size_t count = 0;
            for (size_t i = 0; i < c->tensor_count; ++i) count += c->tensors[i].type == type;
            if (count) { printf("%s\"%u\":%zu", comma ? "," : "", type, count); comma = 1; }
        }
        printf("},\"warmup\":%zu,\"repetitions\":%zu,\"results\":[", warmup, reps);
    } else printf("%s\n%s | %s | %s | %llu bytes\nworkload    tokens/s       stddev      mean ms     ms/token\n", path, machine, backend, c->is_gemma ? "gemma4" : "llama", (unsigned long long)file_bytes);
    for (int mode = 0; mode < 2; ++mode) {
        size_t count = mode ? tg : pp;
        if (count == 0) continue;
        double sum = 0, rates = 0, variance = 0;
        nya_compute_stats measured = {0};
        const char *execution = nya_llm_session_execution(s);
        for (size_t r = 0; r < warmup + reps; ++r) {
            nya_llm_session_reset(s);
            if (mode && prefix && nya_llm_session_run(s, tokens, prefix, 0) != 0) goto run_failure;
            nya_compute_stats before, after;
            nya_llm_session_stats(s, &before);
            double start = clock_seconds();
            if (nya_llm_session_run(s, tokens + (mode ? prefix : 0), count, mode) != 0) goto run_failure;
            double elapsed = clock_seconds() - start;
            nya_llm_session_stats(s, &after);
            if (start < 0 || elapsed <= 0 || strcmp(backend, nya_compute_name(c->compute)) || strcmp(execution, nya_llm_session_execution(s))) goto run_failure;
            if (r >= warmup) {
                samples[r - warmup] = (double)count / elapsed; sum += elapsed; rates += (double)count / elapsed;
                measured.kernel_launches += after.kernel_launches - before.kernel_launches;
                measured.external_matmul_calls += after.external_matmul_calls - before.external_matmul_calls;
                measured.cutlass_matmul_calls += after.cutlass_matmul_calls - before.cutlass_matmul_calls;
                measured.tiled_attention_calls += after.tiled_attention_calls - before.tiled_attention_calls;
                measured.graph_captures += after.graph_captures - before.graph_captures;
                measured.graph_replays += after.graph_replays - before.graph_replays;
                measured.uploads += after.uploads - before.uploads;
                measured.downloads += after.downloads - before.downloads;
                measured.synchronizations += after.synchronizations - before.synchronizations;
            }
        }
        rates /= (double)reps;
        for (size_t r = 0; r < reps; ++r) variance += (samples[r] - rates) * (samples[r] - rates);
        double sd = reps > 1 ? sqrt(variance / (double)(reps - 1)) : 0;
        if (json) printf("%s{\"test\":\"%s%zu\",\"tokens\":%zu,\"prefix_tokens\":%zu,\"tokens_per_second\":%.9g,\"stddev\":%.9g,\"elapsed_seconds\":%.9g,\"mean_latency_ms\":%.9g,\"ms_per_token\":%.9g,\"kernel_launches\":%llu,\"uploads\":%llu,\"downloads\":%llu,\"synchronizations\":%llu,\"external_matmul_calls\":%llu,\"graph_captures\":%llu,\"graph_replays\":%llu,\"cutlass_matmul_calls\":%llu,\"tiled_attention_calls\":%llu}",
            mode && pp ? "," : "", mode ? "tg" : "pp", count, count, mode ? prefix : 0, rates, sd, sum, sum * 1000 / (double)reps, sum * 1000 / (double)(reps * count),
            measured.kernel_launches, measured.uploads, measured.downloads, measured.synchronizations, measured.external_matmul_calls,
            measured.graph_captures, measured.graph_replays, measured.cutlass_matmul_calls, measured.tiled_attention_calls);
        else printf("%s%-7zu %12.3f %12.3f %12.3f %12.3f\n", mode ? "tg" : "pp", count, rates, sd, sum * 1000 / (double)reps, sum * 1000 / (double)(reps * count));
        fflush(stdout);
    }
    if (json) {
        nya_compute_stats final_memory; nya_llm_session_stats(s,&final_memory);
        /* Assisted providers can populate their weight caches during warmup.
           Preserve admission-time fields and add an explicit final snapshot. */
        printf("],\"device_memory_after\":{\"weights_bytes\":%zu,\"kv_bytes\":%zu,\"scratch_bytes\":%zu}}\n",
            final_memory.weights_bytes,final_memory.kv_bytes,final_memory.scratch_bytes);
    }
    status = 0;
    goto cleanup;
run_failure:
    snprintf(error, sizeof(error), "inference failed, clock failed, or backend changed during measurement; discard partial output");
cleanup:
    if (status) fprintf(stderr, "fyodor-bench: %s\n", error[0] ? error : "allocation failed");
    nya_llm_session_free(s); nya_llm_free(c); free(tokens); free(samples);
    return status;
}
