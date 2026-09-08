/* Compare sequence-at-once autograd with the inference engine's cached token
   steps. The two execution paths must agree after both full and LoRA export. */
#include "../core/llm_cpu.c"
#include "pretraining.h"
#include "model.h"

#define REQUIRE(x) do { if (!(x)) { fprintf(stderr,"pretraining:%d: %s (%s)\n",__LINE__,#x,error); return 1; } } while (0)

static const uint32_t tokens[] = {1,100,101,102,100,101,102,100,101,102,100,101};
static const uint32_t labels[] = {100,101,102,100,101,102,100,101,102,100,101,102};

static int export_load(nya_train_decoder *decoder, const char *path, nya_model_registry *registry,
    const nya_model **model, char *error, size_t capacity)
{
    FILE *file = fopen(path,"wb");
    if (file == NULL) return -1;
    int result = nya_train_decoder_export(decoder,file,error,capacity);
    if (fclose(file) != 0) result = -1;
    if (result != 0 || nya_model_load(registry,path,model) != NYA_MODEL_OK) return -1;
    if (!(*model)->generation_supported) {
        snprintf(error,capacity,"%s",(*model)->generation_error); return -1;
    }
    return 0;
}

static int compare_logits(nya_train_decoder *decoder, const nya_model *model)
{
    char error[256] = {0};
    nya_llm_context *c = (nya_llm_context *)model->generation_context;
    nya_llm_run_state state;
    nya_train_graph *graph = nya_train_graph_create(16*1024*1024);
    nya_train_tensor *logits = nya_train_decoder_forward(decoder,graph,tokens,12);
    REQUIRE(logits != NULL && nya_llm_state_create(c,16,&state,error,sizeof(error)) == 0);
    for (size_t i = 0; i < 12; ++i) {
        REQUIRE(nya_llm_forward(c,&state,tokens[i],i,1) == 0);
        for (size_t j = 0; j < 259; ++j) {
            double delta = fabs((double)state.logits[j]-nya_train_data(logits)[i*259+j]);
            if (delta > 3e-5) { fprintf(stderr,"logit[%zu,%zu] delta=%g\n",i,j,delta); return 1; }
        }
    }
    nya_llm_state_free(&state); nya_train_graph_free(graph);
    return 0;
}

static int fit(nya_train_decoder *decoder, const uint32_t *target, size_t steps,
    const unsigned char *mask, float *first, float *last)
{
    char error[256] = {0}; size_t count;
    nya_train_parameter *const *parameters = nya_train_decoder_parameters(decoder,&count);
    nya_train_adamw optimizer; nya_train_adamw_defaults(&optimizer);
    optimizer.learning_rate = 0.015f; optimizer.weight_decay = 0;
    for (size_t step = 0; step <= steps; ++step) {
        for (size_t i = 0; i < count; ++i) nya_train_zero_grad(parameters[i]);
        nya_train_graph *graph = nya_train_graph_create(16*1024*1024);
        nya_train_tensor *loss = nya_train_cross_entropy(nya_train_decoder_forward(decoder,graph,tokens,12),target,mask,12);
        if (loss == NULL) snprintf(error,sizeof(error),"%s",nya_train_error(graph));
        REQUIRE(loss != NULL);
        *last = nya_train_data(loss)[0]; if (step == 0) *first = *last;
        if (step < steps) {
            REQUIRE(nya_train_backward(loss) == 0);
            nya_train_graph_free(graph);
            REQUIRE(nya_train_adamw_step(&optimizer,parameters,count,error,sizeof(error)) == 0);
        } else nya_train_graph_free(graph);
    }
    return 0;
}

int main(int argc, char **argv)
{
    char error[256] = {0}, path[1200]; float first, last;
    REQUIRE(argc == 2 && strlen(argv[1]) < 1100);
    nya_train_decoder_config config; nya_train_decoder_defaults(&config);
    config.embedding_length = 16; config.feed_forward_length = 32;
    config.block_count = 1; config.head_count = 2; config.kv_head_count = 1; config.context_length = 32;
    nya_train_decoder *decoder = nya_train_decoder_create(&config,error,sizeof(error));
    nya_model_registry registry; nya_model_registry_init(&registry); const nya_model *base, *merged;
    REQUIRE(decoder != NULL);
    snprintf(path,sizeof(path),"%s.random.gguf",argv[1]);
    REQUIRE(export_load(decoder,path,&registry,&base,error,sizeof(error)) == 0);
    REQUIRE(compare_logits(decoder,base) == 0);
    REQUIRE(nya_model_unload(&registry,base->id) == NYA_MODEL_OK);
    REQUIRE(fit(decoder,labels,100,NULL,&first,&last) == 0);
    printf("random-weight pretraining loss: %.6f -> %.6f\n",(double)first,(double)last);
    REQUIRE(first > 5 && last < 0.02f);
    snprintf(path,sizeof(path),"%s.trained.gguf",argv[1]);
    REQUIRE(export_load(decoder,path,&registry,&base,error,sizeof(error)) == 0);
    REQUIRE(compare_logits(decoder,base) == 0);
    nya_generation_request request = {0}; nya_generation_response response;
    request.prompt = "a"; request.max_tokens = 8; request.max_output_bytes = 64; request.top_p = 1;
    REQUIRE(nya_generation_run((nya_model *)base,&request,&response,error,sizeof(error)) == 0);
    REQUIRE(strcmp(response.text,"bcabcabc") == 0);
    nya_generation_response_free(&response);
    /* Full-weight import is the CPT/SFT entry point and must initially recover
       the exact base policy. Adapter import starts equal too because B is zero. */
    nya_train_decoder *full = nya_train_decoder_from_model((nya_model *)base,0,1,16*1024*1024,error,sizeof(error));
    nya_train_decoder *lora = nya_train_decoder_from_model((nya_model *)base,4,8,16*1024*1024,error,sizeof(error));
    REQUIRE(full != NULL && lora != NULL && compare_logits(full,base) == 0 && compare_logits(lora,base) == 0);
    const uint32_t changed[] = {102,102,100,101,102,100,101,102,100,101,102,100};
    const unsigned char mask[] = {0,0,0,1,1,1,1,1,1,1,1,1};
    REQUIRE(fit(lora,changed,300,mask,&first,&last) == 0);
    printf("LoRA SFT loss: %.6f -> %.6f\n",(double)first,(double)last);
    REQUIRE(last < 0.03f && last < first/50);
    snprintf(path,sizeof(path),"%s.merged.gguf",argv[1]);
    REQUIRE(export_load(lora,path,&registry,&merged,error,sizeof(error)) == 0);
    REQUIRE(compare_logits(lora,merged) == 0 && compare_logits(decoder,base) == 0);
    REQUIRE(fit(full,changed,40,NULL,&first,&last) == 0 && last < first/20);
    nya_train_decoder_free(lora); nya_train_decoder_free(full); nya_train_decoder_free(decoder);
    nya_model_registry_shutdown(&registry);
    return 0;
}
