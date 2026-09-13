/* Independent upstream comparison executable, NOT part of Fyodor's backend.
 * Build against the exact llama.cpp headers matching the external shared library.
 * Kept outside CMake so a normal Fyodor build never needs upstream code/libraries.
 * F32 KV is accepted by b10809's C API but excluded by its llama-bench CLI.
 */
#define _POSIX_C_SOURCE 200809L
#include "llama.h"
#include "ggml-backend.h"
#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifdef _WIN32
#include <windows.h>
#endif

static double seconds(void) {
#ifdef _WIN32
    LARGE_INTEGER now, frequency;
    if (!QueryPerformanceCounter(&now) || !QueryPerformanceFrequency(&frequency)) return -1;
    return (double)now.QuadPart / (double)frequency.QuadPart;
#else
    struct timespec now;
    if (clock_gettime(CLOCK_MONOTONIC, &now)) return -1;
    return (double)now.tv_sec + (double)now.tv_nsec * 1e-9;
#endif
}

static int number(const char *s, int *out) {
    char *end;
    if (*s < '0' || *s > '9') return -1;
    errno = 0;
    long n = strtol(s, &end, 10);
    if (errno || *end || n > 1048576) return -1;
    *out = (int)n;
    return 0;
}

static void json_string(const char *s) {
    putchar('"');
    for (const unsigned char *p = (const unsigned char *)s; *p; ++p) {
        if (*p == '"' || *p == '\\') printf("\\%c", *p);
        else if (*p < 32) printf("\\u%04x", (unsigned)*p);
        else putchar(*p);
    }
    putchar('"');
}

/* The output head runs only for the final prompt position, or for every decode
 * token. Synchronization includes the CPU-visible logits transfer. Prefix work,
 * cache reset, allocation and post-run finite checks are outside measurement. */
static int run(struct llama_context *ctx, llama_token *tokens, int count,
               int each, int batch_size, int8_t *outputs) {
    for (int offset = 0; offset < count;) {
        int n = each ? 1 : count-offset;
        if (n > batch_size) n = batch_size;
        memset(outputs, 0, (size_t)n);
        outputs[n-1] = (int8_t)(each || offset+n == count);
        struct llama_batch batch = {0};
        batch.n_tokens = n; batch.token = tokens+offset; batch.logits = outputs;
        if (llama_decode(ctx, batch)) return -1;
        offset += n;
        if (each || offset == count) {
            llama_synchronize(ctx);
            if (!llama_get_logits_ith(ctx, -1)) return -1;
        }
    }
    return 0;
}

int main(int argc, char **argv) {
    const char *path = NULL, *backend = "cuda";
    int pp=512, tg=128, prefix=0, reps=5, warmup=1, batch_size=512, threads=6;
    for (int i=1; i<argc; ++i) {
        const char *arg=argv[i];
        if (!strcmp(arg,"--json")) continue;
        if (++i == argc) return 2;
        if (!strcmp(arg,"-m")) path=argv[i];
        else if (!strcmp(arg,"-b")) backend=argv[i];
        else {
            int *value = !strcmp(arg,"-p") ? &pp : !strcmp(arg,"-n") ? &tg :
                !strcmp(arg,"--context") ? &prefix : !strcmp(arg,"-r") ? &reps :
                !strcmp(arg,"--warmup") ? &warmup : !strcmp(arg,"--batch") ? &batch_size :
                !strcmp(arg,"--threads") ? &threads : NULL;
            if (!value || number(argv[i],value)) return 2;
        }
    }
    if (!path || (!pp && !tg) || !reps || reps>1000 || warmup>1000 ||
        !batch_size || !threads || (strcmp(backend,"cuda") && strcmp(backend,"cpu"))) return 2;
    int capacity = pp > prefix+tg ? pp : prefix+tg;
    int status=1, vocabulary=0;
    struct llama_model *model=NULL;
    struct llama_context *ctx=NULL;
    llama_token *tokens=NULL;
    int8_t *outputs=NULL;
    double *samples=NULL;
    ggml_backend_load_all();
    llama_backend_init();
    ggml_backend_dev_t devices[2]={NULL,NULL};
    if (!strcmp(backend,"cuda")) {
        devices[0]=ggml_backend_dev_by_name("CUDA0");
        if (!devices[0]) { fprintf(stderr,"CUDA0 unavailable; refusing CPU fallback\n"); goto done; }
    }
    struct llama_model_params mp=llama_model_default_params();
    mp.devices=devices; mp.n_gpu_layers=devices[0] ? 99 : 0;
    model=llama_model_load_from_file(path,mp);
    if (!model || capacity>llama_model_n_ctx_train(model)) goto done;
    vocabulary=llama_vocab_n_tokens(llama_model_get_vocab(model));
    if (vocabulary<=0) goto done;
    struct llama_context_params cp=llama_context_default_params();
    cp.n_ctx=(uint32_t)capacity; cp.n_batch=(uint32_t)batch_size; cp.n_ubatch=(uint32_t)batch_size;
    cp.n_threads=threads; cp.n_threads_batch=threads;
    cp.type_k=GGML_TYPE_F32; cp.type_v=GGML_TYPE_F32;
    cp.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cp.no_perf=true;
    ctx=llama_init_from_model(model,cp);
    if (!ctx || !llama_get_memory(ctx)) goto done;
    tokens=malloc((size_t)capacity*sizeof(*tokens));
    outputs=malloc((size_t)batch_size);
    samples=malloc((size_t)reps*sizeof(*samples));
    if (!tokens || !outputs || !samples) goto done;
    /* Explicit modulo exactly matches fyodor-bench; no BOS insertion/tokenizer. */
    for (int i=0; i<capacity; ++i)
        tokens[i]=(llama_token)(((uint64_t)i*UINT64_C(15485863)+17)%(uint64_t)vocabulary);
    printf("{\"schema_version\":1,\"engine\":\"llama.cpp C API\",\"model\":"); json_string(path);
    printf(",\"backend\":"); json_string(backend);
    printf(",\"kv_type\":\"f32\",\"flash_attention\":false,\"requested_context\":%d,\"allocated_context\":%u,"
           "\"batch\":%d,\"threads\":%d,\"warmup\":%d,\"repetitions\":%d,\"results\":[",
           capacity,llama_n_ctx(ctx),batch_size,threads,warmup,reps);
    for (int mode=0; mode<2; ++mode) {
        int count=mode ? tg : pp;
        if (!count) continue;
        double elapsed_sum=0, mean=0, variance=0;
        for (int r=0; r<warmup+reps; ++r) {
            /* Removing cache entries is untimed, as in Fyodor session_reset.
             * Do not clear device storage: valid-position metadata bounds reads. */
            llama_memory_clear(llama_get_memory(ctx),false);
            if (mode && prefix && run(ctx,tokens,prefix,0,batch_size,outputs)) goto done;
            double start=seconds();
            if (run(ctx,tokens+(mode ? prefix : 0),count,mode,batch_size,outputs)) goto done;
            double elapsed=seconds()-start;
            if (start<0 || elapsed<=0) goto done;
            const float *logits=llama_get_logits_ith(ctx,-1);
            for (int j=0; j<vocabulary; ++j) if (!isfinite(logits[j])) goto done;
            if (r>=warmup) {
                samples[r-warmup]=(double)count/elapsed;
                mean+=samples[r-warmup]; elapsed_sum+=elapsed;
            }
        }
        mean/=(double)reps;
        for (int r=0; r<reps; ++r) variance+=(samples[r]-mean)*(samples[r]-mean);
        double sd=reps>1 ? sqrt(variance/(double)(reps-1)) : 0;
        printf("%s{\"test\":\"%s%d\",\"tokens\":%d,\"prefix_tokens\":%d,\"tokens_per_second\":%.9g,"
               "\"stddev\":%.9g,\"elapsed_seconds\":%.9g,\"mean_latency_ms\":%.9g,\"ms_per_token\":%.9g}",
               mode && pp ? "," : "",mode ? "tg" : "pp",count,count,mode ? prefix : 0,mean,sd,
               elapsed_sum,elapsed_sum*1000/(double)reps,elapsed_sum*1000/((double)reps*count));
        fflush(stdout);
    }
    puts("]}"); status=0;
done:
    if (status) fprintf(stderr,"Comparison failed; discard partial JSON\n");
    free(tokens); free(outputs); free(samples);
    if (ctx) llama_free(ctx);
    if (model) llama_model_free(model);
    llama_backend_free();
    return status;
}
