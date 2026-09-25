/* Exercise the CLI's private update boundary against a single combined graph.
   The reference constructs its objectives directly; it does not use the
   runner's selection, token counting, or loss-normalization helpers. */
#define main nya_train_cli_main
#define wmain nya_train_cli_wmain
#include "../core/train_main.c"
#undef wmain
#undef main

#define REQUIRE(x) do { if (!(x)) { fprintf(stderr,"runner line %d: %s\n",__LINE__,#x); return 1; } } while (0)

static nya_train_tensor *reference_loss(nya_train_decoder *model, nya_train_graph *g,
    const train_record *record, int dpo)
{
    const train_sequence *s = &record->chosen, *r = &record->rejected;
    nya_train_tensor *chosen = nya_train_decoder_forward(model,g,s->tokens,s->count-1);
    if (!dpo) return nya_train_cross_entropy(chosen,s->tokens+1,s->mask,s->count-1);
    nya_train_tensor *rejected = nya_train_decoder_forward(model,g,r->tokens,r->count-1);
    return nya_train_dpo(nya_train_logprob(chosen,s->tokens+1,s->mask,s->count-1),
        nya_train_logprob(rejected,r->tokens+1,r->mask,r->count-1),
        record->reference_chosen,record->reference_rejected,0.1f);
}

static int compare(nya_train_decoder *actual, nya_train_decoder *expected, int cleared)
{
    size_t na, nb;
    nya_train_parameter *const *a = nya_train_decoder_parameters(actual,&na);
    nya_train_parameter *const *b = nya_train_decoder_parameters(expected,&nb);
    nya_train_graph *g = nya_train_graph_create(1024*1024);
    REQUIRE(na == nb && g != NULL);
    double max_data = 0, max_grad = 0;
    for (size_t i = 0; i < na; ++i) {
        nya_train_tensor *leaf = nya_train_leaf(g,a[i]); REQUIRE(leaf != NULL);
        size_t n = nya_train_rows(leaf)*nya_train_columns(leaf);
        for (size_t j = 0; j < n; ++j) {
            double delta = fabs((double)nya_train_parameter_data(a[i])[j]-nya_train_parameter_data(b[i])[j]);
            if (delta > max_data) max_data = delta;
            REQUIRE(delta < 0.000002);
            if (cleared) REQUIRE(delta == 0);
            double gradient = nya_train_parameter_gradient(a[i])[j];
            if (cleared) REQUIRE(gradient == 0);
            else {
                delta = fabs(gradient-nya_train_parameter_gradient(b[i])[j]);
                if (delta > max_grad) max_grad = delta;
                REQUIRE(delta < 0.000002);
            }
        }
    }
    printf("combined graph: parameter max_abs=%.9g gradient max_abs=%.9g\n",max_data,max_grad);
    nya_train_graph_free(g); return 0;
}

static int objective(int dpo)
{
    char error[256] = {0};
    nya_train_decoder_config config; nya_train_decoder_defaults(&config);
    config.embedding_length = 8; config.feed_forward_length = 16;
    config.block_count = 1; config.head_count = 2; config.kv_head_count = 1; config.context_length = 16;
    nya_train_decoder *actual = nya_train_decoder_create(&config,error,sizeof(error));
    nya_train_decoder *expected = nya_train_decoder_create(&config,error,sizeof(error));
    REQUIRE(actual != NULL && expected != NULL);
    uint32_t a[] = {100,101,102,100}, b[] = {102,101,100,102,100,101,102,100};
    uint32_t c[] = {100,102,101,101,102,100,101}, d[] = {102,100,101};
    unsigned char am[] = {0,1,1}, bm[] = {0,0,1,1,1,0,1};
    unsigned char cm[] = {0,1,1,1,1,1}, dm[] = {1,1};
    train_record records[] = {
        {{a,am,4,2},{c,cm,7,5},-12,-13},
        {{b,bm,8,4},{d,dm,3,2},-9,-14}
    };
    train_dataset data = {records,{0},16,2,dpo};
    train_options o = {0}; o.accumulate = 2; o.beta = 0.1f;
    /* Establish the exact largest single graph budget, including loss scaling.
       A combined graph would exceed it, so retaining all microbatches fails. */
    size_t first_budget = 0;
    for (size_t i = 0; i < 2; ++i) {
        nya_train_graph *g = nya_train_graph_create(1024*1024);
        nya_train_tensor *loss = nya_train_scale(reference_loss(actual,g,&records[i],dpo),0.5f);
        REQUIRE(loss != NULL && nya_train_backward(loss) == 0);
        size_t bytes = nya_train_memory_used(g);
        if (!i) first_budget = bytes;
        if (bytes > o.memory) o.memory = bytes;
        nya_train_graph_free(g);
    }
    nya_train_adamw actual_opt, expected_opt;
    nya_train_adamw_defaults(&actual_opt); actual_opt.max_grad_norm = 0.03f;
    expected_opt = actual_opt;
    size_t count;
    nya_train_parameter *const *parameters = nya_train_decoder_parameters(expected,&count);
    for (size_t step = 0; step < 3; ++step) {
        for (size_t i = 0; i < count; ++i) nya_train_zero_grad(parameters[i]);
        nya_train_graph *g = nya_train_graph_create(1024*1024);
        /* Unequal masks: two and four supervised labels. DPO gives each pair
           equal weight regardless of either sequence's token count. */
        nya_train_tensor *left = nya_train_scale(reference_loss(expected,g,&records[0],dpo),dpo ? 0.5f : 2.0f/6.0f);
        nya_train_tensor *right = nya_train_scale(reference_loss(expected,g,&records[1],dpo),dpo ? 0.5f : 4.0f/6.0f);
        nya_train_tensor *loss = nya_train_add(left,right);
        REQUIRE(loss != NULL && nya_train_backward(loss) == 0);
        double reference_value = nya_train_data(loss)[0];
        size_t combined_bytes = nya_train_memory_used(g);
        REQUIRE(nya_train_adamw_step(&expected_opt,parameters,count,error,sizeof(error)) == 0);
        nya_train_graph_free(g);
        train_metrics metrics;
        REQUIRE(train_update(actual,&data,&o,&actual_opt,&metrics,error,sizeof(error)) == 0);
        REQUIRE(actual_opt.step == step+1 && fabs(metrics.loss-reference_value) < 0.000002);
        REQUIRE(metrics.tokens == (dpo ? 18U : 10U) && metrics.units == (dpo ? 2U : 6U));
        REQUIRE(metrics.graph_bytes == o.memory && metrics.graph_bytes < combined_bytes);
        REQUIRE(compare(actual,expected,0) == 0);
        size_t actual_count;
        nya_train_parameter *const *actual_parameters = nya_train_decoder_parameters(actual,&actual_count);
        FILE *before = tmpfile(), *after = tmpfile(); REQUIRE(before && after);
        REQUIRE(nya_train_checkpoint_write(before,&actual_opt,actual_parameters,actual_count) == 0);
        train_evaluation evaluated;
        REQUIRE(evaluate(actual,&data,&o,&evaluated,error,sizeof(error)) == 0);
        REQUIRE(evaluated.records == 2 && evaluated.units == (dpo ? 2U : 6U));
        REQUIRE(evaluated.tokens == (dpo ? 18U : 10U) && evaluated.graph_bytes < o.memory);
        g = nya_train_graph_create(1024*1024);
        left = reference_loss(actual,g,&records[0],dpo);
        right = reference_loss(actual,g,&records[1],dpo);
        REQUIRE(left && right);
        double independent = dpo ? ((double)nya_train_data(left)[0]+nya_train_data(right)[0])/2 :
            ((double)nya_train_data(left)[0]*2+(double)nya_train_data(right)[0]*4)/6;
        REQUIRE(evaluated.loss == independent);
        o.eval_records = 1;
        REQUIRE(evaluate(actual,&data,&o,&evaluated,error,sizeof(error)) == 0);
        REQUIRE(evaluated.records == 1 && evaluated.loss == nya_train_data(left)[0]);
        o.eval_records = 0;
        size_t saved_budget = o.memory; o.memory = 1;
        REQUIRE(evaluate(actual,&data,&o,&evaluated,error,sizeof(error)) == -1);
        o.memory = saved_budget; error[0] = '\0';
        REQUIRE(nya_train_checkpoint_write(after,&actual_opt,actual_parameters,actual_count) == 0);
        rewind(before); rewind(after);
        int before_byte, after_byte;
        do { before_byte = fgetc(before); after_byte = fgetc(after); REQUIRE(before_byte == after_byte); } while (before_byte != EOF);
        REQUIRE(!ferror(before) && !ferror(after)); fclose(before); fclose(after);
        nya_train_graph_free(g);
    }
    if (!dpo) {
        /* Microbatch one fits; two fails. No partial Adam update, no stale
           accumulated gradients, and the previous parameter state survives. */
        size_t actual_count;
        nya_train_parameter *const *actual_parameters = nya_train_decoder_parameters(actual,&actual_count);
        REQUIRE(actual_count == count);
        nya_train_graph *shapes = nya_train_graph_create(1024*1024);
        REQUIRE(shapes != NULL);
        /* Freeze the exact pre-failure weights, rather than accepting the
           combined graph's already-measured last-bit difference as rollback. */
        for (size_t i = 0; i < count; ++i) {
            nya_train_tensor *leaf = nya_train_leaf(shapes,actual_parameters[i]); REQUIRE(leaf != NULL);
            size_t bytes = nya_train_rows(leaf)*nya_train_columns(leaf)*sizeof(float);
            memcpy(nya_train_parameter_data(parameters[i]),nya_train_parameter_data(actual_parameters[i]),bytes);
        }
        nya_train_graph_free(shapes);
        REQUIRE(first_budget < o.memory); o.memory = first_budget;
        train_metrics metrics;
        REQUIRE(train_update(actual,&data,&o,&actual_opt,&metrics,error,sizeof(error)) != 0);
        REQUIRE(strstr(error,"microbatch 2") != NULL && actual_opt.step == 3);
        REQUIRE(compare(actual,expected,1) == 0);
    }
    nya_train_decoder_free(actual); nya_train_decoder_free(expected); return 0;
}

static int control_pipe(void)
{
    int ends[2];
#ifdef _WIN32
    REQUIRE(_pipe(ends,256,_O_BINARY|_O_NOINHERIT) == 0);
    int saved = _dup(0);
    REQUIRE(_dup2(ends[0],0) == 0);
    _close(ends[0]);
#else
    REQUIRE(pipe(ends) == 0);
    int saved = dup(0);
    REQUIRE(dup2(ends[0],0) == 0);
    close(ends[0]);
#endif
    REQUIRE(train_control_valid());
    REQUIRE(!train_control_stop()); /* An empty, live pipe must not block. */
#ifdef _WIN32
    REQUIRE(_write(ends[1],"?\n",2) == 2);
#else
    REQUIRE(write(ends[1],"?\n",2) == 2);
#endif
    REQUIRE(!train_control_stop());
#ifdef _WIN32
    REQUIRE(_write(ends[1],"S",1) == 1);
#else
    REQUIRE(write(ends[1],"S",1) == 1);
#endif
    REQUIRE(train_control_stop());
    REQUIRE(!train_control_stop()); /* The command was consumed. */
#ifdef _WIN32
    _close(ends[1]);
#else
    close(ends[1]);
#endif
    REQUIRE(train_control_stop()); /* EOF requests a safe stop too. */
    FILE *regular = tmpfile(); REQUIRE(regular != NULL);
#ifdef _WIN32
    REQUIRE(_dup2(_fileno(regular),0) == 0);
#else
    REQUIRE(dup2(fileno(regular),0) == 0);
#endif
    REQUIRE(!train_control_valid());
    fclose(regular);
#ifdef _WIN32
    if (saved >= 0) { REQUIRE(_dup2(saved,0) == 0); _close(saved); }
    else _close(0);
#else
    if (saved >= 0) { REQUIRE(dup2(saved,0) == 0); close(saved); }
    else close(0);
#endif
    return 0;
}

static int unicode_files(void)
{
    /* UTF-8 Hangul + non-BMP emoji, independent of the source code page.
       Exclusive creation makes a stale/colliding file a test failure, never
       permission to remove or overwrite a pre-existing file. */
    const char *path = "nya-train-path-\xed\x95\x9c\xea\xb8\x80-\xf0\x9f\x90\xbe.bin";
    const unsigned char bytes[] = {0,10,13,255,42};
    FILE *file = nya_file_create_exclusive(path); REQUIRE(file != NULL);
    REQUIRE(fwrite(bytes,1,sizeof(bytes),file) == sizeof(bytes));
    REQUIRE(fclose(file) == 0);
    REQUIRE(nya_file_create_exclusive(path) == NULL && errno == EEXIST);
    file = nya_file_open_read(path); REQUIRE(file != NULL);
    uint64_t size = 0; REQUIRE(nya_file_regular_size(file,&size) == 0 && size == sizeof(bytes));
    unsigned char readback[sizeof(bytes)];
    REQUIRE(fread(readback,1,sizeof(readback),file) == sizeof(readback));
    REQUIRE(memcmp(bytes,readback,sizeof(bytes)) == 0 && fclose(file) == 0);
    REQUIRE(nya_file_remove(path) == 0);
    REQUIRE(nya_file_open_read(path) == NULL && errno == ENOENT);
    REQUIRE(nya_file_create_exclusive(NULL) == NULL && errno == EINVAL);
    REQUIRE(nya_file_remove(NULL) != 0 && errno == EINVAL);
#ifdef _WIN32
    REQUIRE(nya_file_create_exclusive("nya-invalid-\xff") == NULL && errno == EINVAL);
    REQUIRE(nya_file_open_read("nya-invalid-\xff") == NULL && errno == EINVAL);
    REQUIRE(nya_file_remove("nya-invalid-\xff") != 0 && errno == EINVAL);
#endif
    return 0;
}

int main(void)
{
    REQUIRE(batch_start(UINT64_MAX,1024,9) == 6);
    REQUIRE(batch_start(SIZE_MAX-1,1024,SIZE_MAX) == SIZE_MAX-1024);
    REQUIRE(batch_start(0,1024,1) == 0);
    return unicode_files() || control_pipe() || objective(0) || objective(1);
}
