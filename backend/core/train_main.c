/* Small native training runner built on the same public eager C API as an
   embedding application. Keep dataset policy here, outside the model kernels.
   A corpus uses next-token windows; SFT/DPO use one tab-separated record per
   line. Files are bounded and validated before any optimizer update. */
#include "pretraining.h"
#include "model.h"
#include "file.h"
#include "train_clock.h"
#include "train_control.h"

#include <errno.h>
#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <sys/stat.h>
#ifdef _WIN32
#include <io.h>
#else
#include <unistd.h>
#endif

typedef struct train_sequence {
    uint32_t *tokens;
    unsigned char *mask;
    size_t count, supervised;
} train_sequence;

typedef struct train_record {
    train_sequence chosen, rejected;
    double reference_chosen, reference_rejected;
} train_record;

typedef struct train_options {
    const char *mode, *data, *output, *base, *checkpoint, *resume, *metrics;
    size_t steps, rank, memory, accumulate, control_stdin, threads;
    nya_train_executor *executor;
    float rate, beta;
    nya_train_decoder_config config;
} train_options;

static void usage(void)
{
    puts("fyodor-train --mode pretrain|cpt|sft|dpo --data FILE --output MODEL.gguf\n"
         "  --base MODEL.gguf       Required for cpt/sft/dpo; dense LLaMA or Gemma 4\n"
         "  --rank N                LoRA rank (default 8); 0 trains all weights\n"
         "  --steps N --lr X        Updates and learning rate (default 100, 0.001)\n"
         "  --accumulate N          Sequences/pairs per update, 1..1024 (default 1)\n"
         "  --threads N             CPU threads, 1..64; 0 selects host cores (default)\n"
         "  --context N             Corpus window; random model context (default 256)\n"
         "  --dimension N --ff N --layers N --heads N --kv-heads N --seed N\n"
         "                          Random-model dimensions and initialization seed\n"
         "  --memory-mib N          Budget for each of parameters and graph (default 256)\n"
         "  --checkpoint FILE --resume FILE   Save/load weights and AdamW state\n"
         "  --metrics FILE          Write per-update timing/loss CSV to a new path\n"
         "  --control-stdin 1       Pipe control: S or EOF saves and stops between updates\n"
         "  --beta X                DPO beta (default 0.1)\n"
         "Corpora are UTF-8 text. SFT lines: prompt<TAB>completion.\n"
         "DPO lines: prompt<TAB>chosen<TAB>rejected. Tabs/newlines delimit records.\n"
         "Outputs must be new paths. Resume with the SAME model/config/data/objective.\n"
         "CPU training; accelerator backward, MoE/MTP/encoder training and distributed runs are pending.");
}

static int integer(const char *text, size_t *value)
{
    if (*text == '\0') return -1;
    size_t v = 0;
    for (const unsigned char *p = (const unsigned char *)text; *p; ++p) {
        if (*p < '0' || *p > '9' || v > (SIZE_MAX-(size_t)(*p-'0'))/10) return -1;
        v = v*10+(size_t)(*p-'0');
    }
    *value = v; return 0;
}

static int real(const char *text, float *value)
{
    char *end; errno = 0; float v = strtof(text,&end);
    if (errno != 0 || end == text || *end != '\0' || !isfinite(v) || v <= 0) return -1;
    *value = v; return 0;
}

static int options(int argc, char **argv, train_options *o)
{
    memset(o,0,sizeof(*o)); nya_train_decoder_defaults(&o->config);
    o->mode = "pretrain"; o->steps = 100; o->rank = 8; o->rate = 0.001f;
    o->beta = 0.1f; o->memory = 256U*1024U*1024U; o->accumulate = 1;
    for (int i = 1; i < argc; i += 2) {
        const char *key = argv[i];
        if (i+1 >= argc) return -1;
        const char *value = argv[i+1]; size_t n;
        if (strcmp(key,"--mode") == 0) o->mode = value;
        else if (strcmp(key,"--data") == 0) o->data = value;
        else if (strcmp(key,"--output") == 0) o->output = value;
        else if (strcmp(key,"--base") == 0) o->base = value;
        else if (strcmp(key,"--checkpoint") == 0) o->checkpoint = value;
        else if (strcmp(key,"--resume") == 0) o->resume = value;
        else if (strcmp(key,"--metrics") == 0) o->metrics = value;
        else if (strcmp(key,"--lr") == 0) { if (real(value,&o->rate) != 0) return -1; }
        else if (strcmp(key,"--beta") == 0) { if (real(value,&o->beta) != 0) return -1; }
        else {
            if (integer(value,&n) != 0) return -1;
            if (strcmp(key,"--steps") == 0) o->steps = n;
            else if (strcmp(key,"--accumulate") == 0) o->accumulate = n;
            else if (strcmp(key,"--control-stdin") == 0) o->control_stdin = n;
            else if (strcmp(key,"--threads") == 0) o->threads = n;
            else if (strcmp(key,"--rank") == 0) o->rank = n;
            else if (strcmp(key,"--seed") == 0) o->config.seed = n;
            else if (strcmp(key,"--memory-mib") == 0) {
                if (n == 0 || n > SIZE_MAX/(1024U*1024U)) return -1;
                o->memory = n*1024U*1024U;
            } else {
                if (n == 0 || n > UINT32_MAX) return -1;
                if (strcmp(key,"--context") == 0) o->config.context_length = (uint32_t)n;
                else if (strcmp(key,"--dimension") == 0) o->config.embedding_length = (uint32_t)n;
                else if (strcmp(key,"--ff") == 0) o->config.feed_forward_length = (uint32_t)n;
                else if (strcmp(key,"--layers") == 0) o->config.block_count = (uint32_t)n;
                else if (strcmp(key,"--heads") == 0) o->config.head_count = (uint32_t)n;
                else if (strcmp(key,"--kv-heads") == 0) o->config.kv_head_count = (uint32_t)n;
                else return -1;
            }
        }
    }
    int pretrain = strcmp(o->mode,"pretrain") == 0;
    if (!pretrain && strcmp(o->mode,"cpt") != 0 && strcmp(o->mode,"sft") != 0 && strcmp(o->mode,"dpo") != 0) return -1;
    if (o->data == NULL || o->output == NULL || o->steps == 0 || o->rank > 256 ||
        o->accumulate == 0 || o->accumulate > 1024 || o->control_stdin > 1 || o->threads > 64 ||
        (pretrain ? o->base != NULL : o->base == NULL)) return -1;
    o->config.parameter_memory_limit = o->memory; return 0;
}

static char *read_data(const char *path, size_t *size)
{
    /* The demo runner deliberately caps datasets; applications can stream their
       own batches through pretraining.h without this whole-file allocation. */
    FILE *f = nya_file_open_read(path); if (f == NULL) return NULL;
    if (fseek(f,0,SEEK_END) != 0) { fclose(f); return NULL; }
    long length = ftell(f);
    if (length <= 0 || length > 64L*1024L*1024L || fseek(f,0,SEEK_SET) != 0) { fclose(f); return NULL; }
    char *data = (char *)malloc((size_t)length+1);
    if (data == NULL) { fclose(f); return NULL; }
    int valid = fread(data,1,(size_t)length,f) == (size_t)length && !ferror(f);
    fclose(f);
    if (!valid || memchr(data,0,(size_t)length) != NULL) { free(data); return NULL; }
    data[length] = '\0'; *size = (size_t)length; return data;
}

static void sequence_free(train_sequence *s) { free(s->tokens); free(s->mask); }

static int encode_record(nya_train_decoder *model, const char *prompt, const char *completion,
    train_sequence *s, size_t context, char *error, size_t capacity)
{
    size_t a = strlen(prompt), b = strlen(completion), prefix_count = 0;
    if (b == 0 || a > SIZE_MAX-b-1) return -1;
    char *joined = (char *)malloc(a+b+1); uint32_t *prefix = NULL;
    if (joined == NULL) return -1;
    memcpy(joined,prompt,a); memcpy(joined+a,completion,b+1);
    int result = nya_train_decoder_tokenize(model,prompt,&prefix,&prefix_count,error,capacity);
    if (result == 0) result = nya_train_decoder_tokenize(model,joined,&s->tokens,&s->count,error,capacity);
    free(joined);
    if (result != 0 || s->count < 2 || s->count-1 > context) { free(prefix); return -1; }
    /* Tokenize the concatenation once. A subword crossing the text boundary is
       supervised; preceding prompt-only tokens are masked. This avoids adding
       a second BOS or changing whitespace by tokenizing completions separately. */
    size_t common = 0;
    while (common < prefix_count && common < s->count && prefix[common] == s->tokens[common]) ++common;
    free(prefix);
    if (common == s->count) return -1;
    s->mask = (unsigned char *)calloc(s->count-1,1); if (s->mask == NULL) return -1;
    for (size_t i = 0; i < s->count-1; ++i) {
        s->mask[i] = (unsigned char)(i+1 >= common);
        s->supervised += s->mask[i] != 0;
    }
    return 0;
}

static nya_train_tensor *sequence_loss(nya_train_decoder *model, nya_train_graph *graph,
    const train_sequence *s, int logprob)
{
    nya_train_tensor *logits = nya_train_decoder_forward(model,graph,s->tokens,s->count-1);
    return logprob ? nya_train_logprob(logits,s->tokens+1,s->mask,s->count-1) :
        nya_train_cross_entropy(logits,s->tokens+1,s->mask,s->count-1);
}

typedef struct train_dataset {
    train_record *records;
    train_sequence corpus;
    size_t context, count;
    int dpo;
} train_dataset;

typedef struct train_metrics {
    double loss, forward, backward, optimizer, elapsed;
    size_t tokens, units, graph_bytes;
} train_metrics;

/* (step * accumulation) mod count, without overflowing even at UINT64_MAX.
   Both addends stay below count. The runner caps accumulation at 1024. */
static size_t batch_start(uint64_t step, size_t accumulation, size_t count)
{
    size_t index = 0, add = (size_t)(step % count);
    for (size_t i = 0; i < accumulation; ++i)
        index = add >= count-index ? add-(count-index) : index+add;
    return index;
}

static train_sequence batch_sequence(const train_dataset *data, size_t index)
{
    if (data->records != NULL) return data->records[index].chosen;
    size_t begin = index*data->context, count = data->corpus.count-begin;
    if (count-1 > data->context) count = data->context+1;
    train_sequence sequence = {data->corpus.tokens+begin,NULL,count,count-1};
    return sequence;
}

/* One optimizer update retains only one sequence graph (one pair for DPO).
   CE is a mean over all supervised tokens, not a mean of sequence means;
   DPO is a mean over pairs. Clip/decay/moments/step run once after all backward
   passes. A failed microbatch never triggers a partial optimizer update. */
static int train_update(nya_train_decoder *model, const train_dataset *data,
    const train_options *o, nya_train_adamw *optimizer, train_metrics *m,
    char *error, size_t capacity)
{
    memset(m,0,sizeof(*m));
    double start = tr_seconds();
    size_t first = batch_start(optimizer->step,o->accumulate,data->count), index = first;
    for (size_t micro = 0; micro < o->accumulate; ++micro) {
        train_sequence sequence = batch_sequence(data,index);
        size_t units = data->dpo ? 1 : sequence.supervised, tokens = sequence.count-1;
        if (data->dpo) {
            size_t rejected = data->records[index].rejected.count-1;
            if (rejected > SIZE_MAX-tokens) goto overflow;
            tokens += rejected;
        }
        if (!units || units > SIZE_MAX-m->units || tokens > SIZE_MAX-m->tokens) goto overflow;
        m->units += units; m->tokens += tokens;
        index = index+1 == data->count ? 0 : index+1;
    }
    size_t parameter_count;
    nya_train_parameter *const *parameters = nya_train_decoder_parameters(model,&parameter_count);
    for (size_t i = 0; i < parameter_count; ++i) nya_train_zero_grad(parameters[i]);
    index = first;
    for (size_t micro = 0; micro < o->accumulate; ++micro) {
        train_sequence sequence = batch_sequence(data,index);
        double weight = (double)(data->dpo ? 1 : sequence.supervised)/(double)m->units;
        nya_train_graph *g = nya_train_graph_create_with_executor(o->memory,o->executor);
        double forward_start = tr_seconds();
        nya_train_tensor *loss = sequence_loss(model,g,&sequence,data->dpo);
        if (data->dpo) {
            train_record *record = &data->records[index];
            loss = nya_train_dpo(loss,sequence_loss(model,g,&record->rejected,1),
                record->reference_chosen,record->reference_rejected,o->beta);
        }
        if (loss != NULL) m->loss += (double)nya_train_data(loss)[0]*weight;
        /* Avoid an extra operation/rounding in the existing default path. */
        if (o->accumulate != 1) loss = nya_train_scale(loss,(float)weight);
        double backward_start = tr_seconds();
        if (loss == NULL || nya_train_backward(loss) != 0) {
            snprintf(error,capacity,"microbatch %zu: %s",micro+1,nya_train_error(g));
            nya_train_graph_free(g);
            for (size_t i = 0; i < parameter_count; ++i) nya_train_zero_grad(parameters[i]);
            return -1;
        }
        m->forward += backward_start-forward_start;
        m->backward += tr_seconds()-backward_start;
        size_t bytes = nya_train_memory_used(g);
        if (bytes > m->graph_bytes) m->graph_bytes = bytes;
        nya_train_graph_free(g);
        index = index+1 == data->count ? 0 : index+1;
    }
    double optimizer_start = tr_seconds();
    if (nya_train_adamw_step(optimizer,parameters,parameter_count,error,capacity) != 0) return -1;
    double end = tr_seconds(); m->optimizer = end-optimizer_start; m->elapsed = end-start;
    return 0;
overflow:
    snprintf(error,capacity,"accumulation group has no supervised targets or its token count overflows");
    return -1;
}

/* CLI checkpoints bind the low-level optimizer snapshot to this runner's data
   order and objective. Canonical bytes avoid struct padding/host endianness.
   FNV-1a detects accidental mismatches; it is not an authentication mechanism. */
static uint64_t run_hash_bytes(uint64_t hash, const void *data, size_t count)
{
    const unsigned char *bytes = (const unsigned char *)data;
    for (size_t i = 0; i < count; ++i) { hash ^= bytes[i]; hash *= UINT64_C(1099511628211); }
    return hash;
}

static uint64_t run_hash_u64(uint64_t hash, uint64_t value)
{
    unsigned char bytes[8];
    for (size_t i = 0; i < 8; ++i) bytes[i] = (unsigned char)(value >> (8*i));
    return run_hash_bytes(hash,bytes,sizeof(bytes));
}

static int run_identity(const train_options *o, nya_train_decoder *model,
    size_t context, const char *data, size_t bytes, uint64_t *identity)
{
    const nya_train_decoder_config *c = nya_train_decoder_configuration(model);
    uint64_t hash = run_hash_u64(UINT64_C(14695981039346656037),bytes);
    hash = run_hash_bytes(hash,data,bytes);
    hash = run_hash_bytes(hash,o->mode,strlen(o->mode)+1);
    uint64_t fields[] = {context,o->base == NULL ? 0 : o->rank,c->vocabulary_size,
        c->embedding_length,c->feed_forward_length,c->block_count,c->head_count,
        c->kv_head_count,c->context_length,o->base == NULL ? c->seed : 0};
    for (size_t i = 0; i < sizeof(fields)/sizeof(fields[0]); ++i) hash = run_hash_u64(hash,fields[i]);
    float settings[] = {c->norm_epsilon,c->rope_base,c->initializer_std,
        strcmp(o->mode,"dpo") == 0 ? o->beta : 0};
    for (size_t i = 0; i < sizeof(settings)/sizeof(settings[0]); ++i) {
        uint32_t bits; memcpy(&bits,&settings[i],sizeof(bits)); hash = run_hash_u64(hash,bits);
    }
    /* Frozen LoRA weights, tokenizer and architecture metadata are not in the
       optimizer payload. Hash the entire base container, including those parts. */
    if (o->base != NULL) {
        FILE *file = nya_file_open_read(o->base);
        if (file == NULL) return -1;
        unsigned char buffer[65536]; size_t count;
        while ((count = fread(buffer,1,sizeof(buffer),file)) != 0) hash = run_hash_bytes(hash,buffer,count);
        int failed = ferror(file); if (fclose(file) != 0) failed = 1;
        if (failed) return -1;
    }
    /* Keep existing single-sequence NYARUN v1 checkpoints compatible. Groups
       greater than one extend the identity with their objective/data order. */
    if (o->accumulate != 1) {
        hash = run_hash_bytes(hash,"accumulate",sizeof("accumulate"));
        hash = run_hash_u64(hash,o->accumulate);
    }
    *identity = hash; return 0;
}

static int run_checkpoint_header(FILE *file, int writing, uint64_t identity)
{
    unsigned char expected[16] = {'N','Y','A','R','U','N',1,0}, actual[16];
    for (size_t i = 0; i < 8; ++i) expected[8+i] = (unsigned char)(identity >> (8*i));
    if (writing) return fwrite(expected,1,sizeof(expected),file) == sizeof(expected) ? 0 : -1;
    return fread(actual,1,sizeof(actual),file) == sizeof(actual) && memcmp(actual,expected,sizeof(actual)) == 0 ? 0 : -1;
}

static int train_main(int argc, char **argv)
{
    train_options o;
    if (argc == 2 && strcmp(argv[1],"--help") == 0) { usage(); return 0; }
    if (options(argc,argv,&o) != 0) { usage(); return 2; }
    char error[256] = {0}; int result = 1;
    if (o.control_stdin && !train_control_init()) {
        fputs("training failed: --control-stdin requires a pipe on stdin\n",stderr);
        return 2;
    }
    FILE *metrics = NULL;
    uint64_t identity = 0;
    nya_model_registry registry; nya_model_registry_init(&registry); const nya_model *base = NULL;
    nya_train_decoder *model = NULL; char *data = NULL; size_t bytes = 0, record_count = 0, initialized = 0;
    train_record *records = NULL; train_sequence corpus = {0};
    o.executor = nya_train_executor_create(o.threads);
    if (o.executor == NULL) { snprintf(error,sizeof(error),"cannot create requested CPU training workers"); goto cleanup; }
    if (o.base != NULL && nya_model_load(&registry,o.base,&base) != NYA_MODEL_OK) {
        snprintf(error,sizeof(error),"cannot load base model"); goto cleanup;
    }
    model = base == NULL ? nya_train_decoder_create(&o.config,error,sizeof(error)) :
        nya_train_decoder_from_model((nya_model *)base,o.rank,(float)(o.rank == 0 ? 1 : o.rank*2),o.memory,error,sizeof(error));
    if (model == NULL) goto cleanup;
    size_t context = o.config.context_length;
    if (context > nya_train_decoder_configuration(model)->context_length) context = nya_train_decoder_configuration(model)->context_length;
    data = read_data(o.data,&bytes);
    if (data == NULL) { snprintf(error,sizeof(error),"dataset must be a readable, nonempty text file without NUL bytes, at most 64 MiB"); goto cleanup; }
    if ((o.resume != NULL || o.checkpoint != NULL) && run_identity(&o,model,context,data,bytes,&identity) != 0) {
        snprintf(error,sizeof(error),"cannot fingerprint training inputs"); goto cleanup;
    }
    int dpo = strcmp(o.mode,"dpo") == 0, structured = dpo || strcmp(o.mode,"sft") == 0;
    if (!structured) {
        if (nya_train_decoder_tokenize(model,data,&corpus.tokens,&corpus.count,error,sizeof(error)) != 0 || corpus.count < 2) goto cleanup;
    } else {
        size_t lines = 1;
        for (size_t i = 0; i < bytes; ++i) if (data[i] == '\n') ++lines;
        if (lines > 100000) { snprintf(error,sizeof(error),"at most 100000 structured records are supported"); goto cleanup; }
        records = (train_record *)calloc(lines,sizeof(*records)); if (records == NULL) goto cleanup;
        char *line = data;
        while (*line != '\0') {
            char *next = strchr(line,'\n'); if (next != NULL) *next++ = '\0';
            size_t length = strlen(line); if (length && line[length-1] == '\r') line[--length] = '\0';
            if (length != 0) {
                char *chosen = strchr(line,'\t'), *rejected = NULL;
                if (chosen == NULL) goto bad_record;
                *chosen++ = '\0';
                if (dpo) { rejected = strchr(chosen,'\t'); if (rejected == NULL) goto bad_record; *rejected++ = '\0'; }
                if (strchr(rejected == NULL ? chosen : rejected,'\t') != NULL) goto bad_record;
                train_record *record = &records[record_count]; initialized = record_count+1;
                if (encode_record(model,line,chosen,&record->chosen,context,error,sizeof(error)) != 0 ||
                    (dpo && encode_record(model,line,rejected,&record->rejected,context,error,sizeof(error)) != 0)) goto bad_record;
                /* The initial policy is the fixed DPO reference. Cache sums
                   before loading a resumed policy or performing any updates. */
                if (dpo) {
                    nya_train_graph *g = nya_train_graph_create_with_executor(o.memory,o.executor);
                    nya_train_tensor *c = sequence_loss(model,g,&record->chosen,1);
                    nya_train_tensor *r = sequence_loss(model,g,&record->rejected,1);
                    if (c == NULL || r == NULL) { snprintf(error,sizeof(error),"%s",nya_train_error(g)); nya_train_graph_free(g); goto cleanup; }
                    record->reference_chosen = nya_train_data(c)[0]; record->reference_rejected = nya_train_data(r)[0];
                    nya_train_graph_free(g);
                }
                ++record_count;
            }
            if (next == NULL) break;
            line = next;
        }
        if (record_count == 0) goto bad_record;
    }
    size_t parameter_count;
    nya_train_parameter *const *parameters = nya_train_decoder_parameters(model,&parameter_count);
    nya_train_adamw optimizer; nya_train_adamw_defaults(&optimizer); optimizer.learning_rate = o.rate;
    if (o.resume != NULL) {
        FILE *f = nya_file_open_read(o.resume);
        int loaded = f != NULL && run_checkpoint_header(f,0,identity) == 0 &&
            nya_train_checkpoint_read(f,&optimizer,parameters,parameter_count) == 0;
        if (f != NULL) fclose(f);
        if (!loaded) { snprintf(error,sizeof(error),"checkpoint is unreadable, corrupt, legacy, or mismatches model/data/objective/context; CLI resume requires a matching NYARUN v1 checkpoint"); goto cleanup; }
    }
    if (o.metrics != NULL) {
        metrics = nya_file_create_exclusive(o.metrics);
        if (metrics == NULL) { snprintf(error,sizeof(error),"metrics output must be a new writable path"); goto cleanup; }
        fputs("step,loss,tokens,forward_ms,backward_ms,optimizer_ms,step_ms,tokens_per_second,graph_bytes,microbatches,loss_units,cpu_threads\n",metrics);
    }
    train_dataset dataset = {records,corpus,context,structured ? record_count : (corpus.count-2)/context+1,dpo};
    for (size_t step = 0; step < o.steps; ++step) {
        if (o.control_stdin && train_control_stop()) {
            printf("stopping step=%" PRIu64 " saving_outputs=1\n",optimizer.step);
            fflush(stdout);
            break;
        }
        train_metrics m;
        if (train_update(model,&dataset,&o,&optimizer,&m,error,sizeof(error)) != 0) goto cleanup;
        double throughput = m.elapsed > 0 ? (double)m.tokens/m.elapsed : 0;
        if (metrics != NULL && (fprintf(metrics,"%" PRIu64 ",%.9g,%zu,%.6f,%.6f,%.6f,%.6f,%.6f,%zu,%zu,%zu,%zu\n",
            optimizer.step,m.loss,m.tokens,m.forward*1000,m.backward*1000,m.optimizer*1000,m.elapsed*1000,
            throughput,m.graph_bytes,o.accumulate,m.units,nya_train_executor_threads(o.executor)) < 0 || fflush(metrics) != 0)) {
            snprintf(error,sizeof(error),"metrics write failed"); goto cleanup;
        }
        if (step == 0 || (step+1)%10 == 0 || step+1 == o.steps) {
            if (printf("step=%" PRIu64 " loss=%.7f graph_bytes=%zu tokens_per_second=%.3f microbatches=%zu cpu_threads=%zu\n",
                optimizer.step,m.loss,m.graph_bytes,throughput,o.accumulate,nya_train_executor_threads(o.executor)) < 0 || fflush(stdout) != 0) {
                /* A disconnected pipe controller may also close the log
                   reader. Keep the last state recoverable through its EOF. */
                if (!o.control_stdin) { snprintf(error,sizeof(error),"progress output failed"); goto cleanup; }
            }
        }
    }
    /* Exclusive creation prevents an accidental overwrite of the user's base
       model or checkpoint. Failed writes remove only the file created here.
       A process killed during I/O may leave a partial file; do not reuse it. */
    if (o.checkpoint != NULL) {
        FILE *f = nya_file_create_exclusive(o.checkpoint);
        if (f == NULL) { snprintf(error,sizeof(error),"checkpoint output must be a new writable path"); goto cleanup; }
        int written = run_checkpoint_header(f,1,identity);
        if (written == 0) written = nya_train_checkpoint_write(f,&optimizer,parameters,parameter_count);
        if (fclose(f) != 0) written = -1;
        if (written != 0) { nya_file_remove(o.checkpoint); snprintf(error,sizeof(error),"checkpoint write failed"); goto cleanup; }
    }
    FILE *output = nya_file_create_exclusive(o.output);
    if (output == NULL) { snprintf(error,sizeof(error),"GGUF output must be a new writable path"); goto cleanup; }
    int written = nya_train_decoder_export(model,output,error,sizeof(error));
    if (fclose(output) != 0) { written = -1; snprintf(error,sizeof(error),"GGUF close failed"); }
    if (written != 0) { nya_file_remove(o.output); goto cleanup; }
    printf("exported=%s\n",o.output); result = 0; goto cleanup;
bad_record:
    snprintf(error,sizeof(error),"invalid SFT/DPO record %zu: check tabs, nonempty completions, and context length",record_count+1);
cleanup:
    if (metrics != NULL && fclose(metrics) != 0) {
        snprintf(error,sizeof(error),"metrics close failed"); result = 1;
    }
    if (result != 0) fprintf(stderr,"training failed: %s\n",error[0] == '\0' ? "allocation or dataset error" : error);
    for (size_t i = 0; i < initialized; ++i) { sequence_free(&records[i].chosen); sequence_free(&records[i].rejected); }
    free(records); sequence_free(&corpus); free(data); nya_train_decoder_free(model); nya_model_registry_shutdown(&registry);
    nya_train_executor_free(o.executor);
    return result;
}

#ifdef _WIN32
/* Preserve the OS command line before any active-code-page conversion. The
   runner and model/file APIs use UTF-8 internally; no global locale change. */
int wmain(int argc, wchar_t **wide_argv)
{
    if (argc < 0 || (size_t)argc > SIZE_MAX/sizeof(char *)-1) return 2;
    char **argv = calloc((size_t)argc+1,sizeof(*argv));
    if (argv == NULL) return 2;
    int result = 2;
    for (int i = 0; i < argc; ++i) {
        int bytes = WideCharToMultiByte(CP_UTF8,WC_ERR_INVALID_CHARS,wide_argv[i],-1,NULL,0,NULL,NULL);
        if (bytes <= 0) goto cleanup_arguments;
        argv[i] = malloc((size_t)bytes);
        if (argv[i] == NULL || WideCharToMultiByte(CP_UTF8,WC_ERR_INVALID_CHARS,
            wide_argv[i],-1,argv[i],bytes,NULL,NULL) != bytes) goto cleanup_arguments;
    }
    result = train_main(argc,argv);
cleanup_arguments:
    for (int i = 0; i < argc; ++i) free(argv[i]);
    free(argv);
    return result;
}
#else
int main(int argc, char **argv)
{
    return train_main(argc,argv);
}
#endif
