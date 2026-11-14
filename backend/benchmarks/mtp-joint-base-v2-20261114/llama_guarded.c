/* Independent reference diagnostic, never linked into Fyodor.
 * Uses the public C API from llama.cpp b10809 (MIT); no upstream implementation
 * is copied. Compile against that revision's headers and matching shared DLLs.
 * Generation contract: 32 outputs including the sample from final prompt logits.
 * This is not the fixed-forward llama-bench contract or an accepted score.
 */
#include "llama.h"
#include "ggml-backend.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <windows.h>

static double seconds(void)
{
    LARGE_INTEGER now, frequency;
    if (!QueryPerformanceCounter(&now) || !QueryPerformanceFrequency(&frequency)) return -1;
    return (double)now.QuadPart / (double)frequency.QuadPart;
}

int main(int argc, char **argv)
{
    if (argc != 3 || (strcmp(argv[2], "short") && strcmp(argv[2], "long"))) return 2;
    char prompt[40000] = "<|turn>user\n";
    if (!strcmp(argv[2], "long")) for (int i=0;i<4090;++i) strcat(prompt,"Hello. ");
    strcat(prompt,"Write a numbered list of twenty concise facts about the Moon.<turn|>\n<|turn>model\n<|channel>thought\n<channel|>");
    int status=1;
    struct llama_model *model=NULL;
    struct llama_context *ctx=NULL;
    llama_token *tokens=NULL;
    ggml_backend_load_all();
    llama_backend_init();
    ggml_backend_dev_t devices[2]={ggml_backend_dev_by_name("CUDA0"),NULL};
    if (!devices[0]) { fprintf(stderr,"CUDA0 unavailable; refusing fallback\n"); goto done; }
    struct llama_model_params mp=llama_model_default_params();
    mp.devices=devices; mp.n_gpu_layers=99;
    model=llama_model_load_from_file(argv[1],mp);
    if (!model) goto done;
    const struct llama_vocab *vocab=llama_model_get_vocab(model);
    int32_t count=llama_tokenize(vocab,prompt,(int32_t)strlen(prompt),NULL,0,true,true);
    if (count>=0 || count < -100000) goto done;
    count=-count;
    tokens=malloc((size_t)count*sizeof(*tokens));
    if (!tokens || llama_tokenize(vocab,prompt,(int32_t)strlen(prompt),tokens,count,true,true)!=count) goto done;
    printf("PROMPT_IDS");
    for (int32_t i=0;i<count;++i) printf(" %d",tokens[i]);
    putchar('\n'); fflush(stdout);
    struct llama_context_params cp=llama_context_default_params();
    cp.n_ctx=(uint32_t)count+32; cp.n_batch=512; cp.n_ubatch=512;
    cp.n_threads=6; cp.n_threads_batch=6;
    cp.type_k=GGML_TYPE_F32; cp.type_v=GGML_TYPE_F32;
    cp.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_DISABLED; cp.no_perf=true;
    ctx=llama_init_from_model(model,cp);
    if (!ctx) goto done;
    const int32_t vocabulary=llama_vocab_n_tokens(vocab);
    if (vocabulary<=0) goto done;
    for (int round=0;round<3;++round) {
        llama_memory_clear(llama_get_memory(ctx),false);
        for (int32_t offset=0;offset<count;) {
            int32_t n=count-offset; if (n>512) n=512;
            int8_t outputs[512]={0}; outputs[n-1]=(int8_t)(offset+n==count);
            struct llama_batch batch={0}; batch.n_tokens=n; batch.token=tokens+offset; batch.logits=outputs;
            if (llama_decode(ctx,batch)) goto done;
            offset+=n;
        }
        llama_synchronize(ctx);
        if (!llama_get_logits_ith(ctx,-1)) goto done;
        llama_token generated[32]; int used=0, eos=0;
        char text[8192]; size_t length=0;
        Sleep(2000); /* Isolate the decode interval from prefill sampling. */
        double start=seconds();
        for (int i=0;i<32;++i) {
            const float *logits=llama_get_logits_ith(ctx,-1);
            if (!logits) goto done;
            llama_token next=0;
            for (int32_t j=0;j<vocabulary;++j) {
                if (!isfinite(logits[j])) goto done;
                if (logits[j]>logits[next]) next=j;
            }
            generated[used++]=next;
            if (llama_vocab_is_eog(vocab,next)) { eos=1; break; }
            int32_t bytes=llama_token_to_piece(vocab,next,text+length,(int32_t)(sizeof(text)-length),0,false);
            if (bytes<0) goto done;
            length+=(size_t)bytes;
            if (i<31) {
                struct llama_batch batch=llama_batch_get_one(&next,1);
                if (llama_decode(ctx,batch)) goto done;
                llama_synchronize(ctx);
            }
        }
        double end=seconds();
        Sleep(2000); /* Keep post-decode work outside its PDH brackets. */
        if (!(start>0 && end>start)) goto done;
        printf("{\"round\":%d,\"prompt\":%d,\"generated\":%d,\"eos\":%s,\"start\":%.9f,\"end\":%.9f,\"decode_ms\":%.6f,\"ids\":[",round,count,used,eos?"true":"false",start,end,(end-start)*1000);
        for (int i=0;i<used;++i) printf("%s%d",i?",":"",generated[i]);
        printf("],\"text_hex\":\"");
        for (size_t i=0;i<length;++i) printf("%02x",(unsigned)(unsigned char)text[i]);
        printf("\"}\n"); fflush(stdout);
    }
    status=0;
done:
    free(tokens);
    if (ctx) llama_free(ctx);
    if (model) llama_model_free(model);
    llama_backend_free();
    return status;
}
