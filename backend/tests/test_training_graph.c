#include "../core/llm_cpu.c"
#include "training_graph_device.h"
#include "training_internal.h"
#include "model.h"
#define CHECK(x) do { if (!(x)) { fprintf(stderr,"graph:%d: %s (%s)\n",__LINE__,#x,error); return 1; } } while (0)
static const uint32_t tokens[]={1,100,101,102,100,101,102,100,101,102,100,101};
static const uint32_t labels[]={100,101,102,100,101,102,100,101,102,100,101,102};
static nya_train_adamw_config settings(nya_train_adamw o)
{ return (nya_train_adamw_config){o.learning_rate,o.beta1,o.beta2,o.epsilon,o.weight_decay,o.max_grad_norm}; }
static uint64_t u64(const unsigned char *b)
{ uint64_t v=0; for (size_t i=0;i<8;++i) v|=(uint64_t)b[i]<<(8*i); return v; }
static float f32(const unsigned char *b)
{ uint32_t v=(uint32_t)b[0]|(uint32_t)b[1]<<8|(uint32_t)b[2]<<16|(uint32_t)b[3]<<24; float f;memcpy(&f,&v,4);return f; }
static int same_file(FILE *a,FILE *b)
{
    rewind(a);rewind(b);int x,y;
    do { x=fgetc(a);y=fgetc(b);if(x!=y)return 0; } while(x!=EOF);
    return !ferror(a) && !ferror(b);
}
static int compare(FILE *cpu,FILE *gpu,double *maximum)
{
    char error[160]={0};unsigned char a[48],b[48];rewind(cpu);rewind(gpu);
    CHECK(fread(a,1,48,cpu)==48 && fread(b,1,48,gpu)==48 && !memcmp(a,b,48));
    uint64_t count=u64(a+8);
    for (uint64_t p=0;p<count;++p) {
        CHECK(fread(a,1,16,cpu)==16 && fread(b,1,16,gpu)==16 && !memcmp(a,b,16));
        uint64_t elements=u64(a)*u64(a+8);
        for (size_t state=0;state<4;++state) for (uint64_t i=0;i<elements;++i) {
            CHECK(fread(a,1,4,cpu)==4 && fread(b,1,4,gpu)==4);
            float expected=f32(a),actual=f32(b);double delta=fabs((double)expected-actual)/(1+fabs(expected));
            if (!isfinite(actual) || delta>1e-5) {
                fprintf(stderr,"parameter=%llu state=%zu element=%llu CPU=%.9g GPU=%.9g error=%.9g\n",
                    (unsigned long long)p,state,(unsigned long long)i,(double)expected,(double)actual,delta);return 1;
            }
            if (delta>*maximum) *maximum=delta;
        }
    }
    CHECK(fread(a,1,8,cpu)==8 && fread(b,1,8,gpu)==8 && fgetc(cpu)==EOF && fgetc(gpu)==EOF);return 0;
}
static int interface(void)
{
    char error[160]={0};CHECK(!nya_train_graph_create_resident(1024,NULL,0));
    CHECK(nya_train_graph_forward_resident(NULL)<0 && nya_train_decoder_prepare_resident(NULL,NULL,error,sizeof(error))<0);
    CHECK(nya_train_device_slice(NULL,0,0,0,0,0,0,0)<0);
    nya_train_graph *g=nya_train_graph_create(1024);CHECK(g && nya_train_graph_forward_resident(g)<0);nya_train_graph_free(g);
    return 0;
}
static int one_step(nya_train_decoder *decoder,nya_train_session *s,float *loss,int evaluation)
{
    char error[160]={0};nya_train_graph *g=s?nya_train_graph_create_resident(16*1024*1024,s,evaluation):
        evaluation?nya_train_graph_create_for_evaluation(16*1024*1024,NULL):nya_train_graph_create(16*1024*1024);
    CHECK(g); nya_train_tensor *logits=nya_train_decoder_forward(decoder,g,tokens,12);
    nya_train_tensor *objective=nya_train_cross_entropy(logits,labels,NULL,12);
    snprintf(error,sizeof(error),"%s",nya_train_error(g));CHECK(objective);
    nya_train_device_stats before={0},after={0};
    if(s) nya_train_device_get_stats(nya_train_session_device(s),&before);
    int result=evaluation?(s?nya_train_graph_forward_resident(g):0):nya_train_backward(objective);
    snprintf(error,sizeof(error),"%s",nya_train_error(g));CHECK(!result);
    if(s) {
        nya_train_device_get_stats(nya_train_session_device(s),&after);
        CHECK(after.uploads==before.uploads && after.downloads==before.downloads && after.used_bytes==before.used_bytes && after.synchronizations==before.synchronizations);
        CHECK(!nya_train_graph_forward_resident(g));
        nya_train_device_stats repeat; nya_train_device_get_stats(nya_train_session_device(s),&repeat);
        CHECK(repeat.kernel_launches==after.kernel_launches);
    }
    const float *value=nya_train_data(objective);snprintf(error,sizeof(error),"%s",nya_train_error(g));CHECK(value && isfinite(*value));*loss=*value;
    if(evaluation && !s) CHECK(nya_train_backward(objective)<0);
    /* Evaluation backward rejection marks the graph invalid; only exercise it
       on CPU here. Resident evaluation state preservation is checked below. */
    nya_train_graph_free(g);return 0;
}
static int lifecycle(const char *prefix)
{
    char error[256]={0},path[1200];nya_train_decoder_config c;nya_train_decoder_defaults(&c);
    c.embedding_length=16;c.feed_forward_length=32;c.block_count=1;c.head_count=2;c.kv_head_count=1;c.context_length=32;
    nya_train_decoder *cpu=nya_train_decoder_create(&c,error,sizeof(error)),*gpu=nya_train_decoder_create(&c,error,sizeof(error));CHECK(cpu && gpu);
    size_t count,gcount;nya_train_parameter *const *cp=nya_train_decoder_parameters(cpu,&count),*const *gp=nya_train_decoder_parameters(gpu,&gcount);CHECK(count==gcount);
    nya_train_adamw optimizer;nya_train_adamw_defaults(&optimizer);optimizer.learning_rate=0.015f;optimizer.weight_decay=0;
    nya_train_adamw initial=optimizer;
    nya_train_session *s=nya_train_session_create("cuda",16*1024*1024,&optimizer,gp,count,error,sizeof(error));CHECK(s);
    CHECK(!nya_train_decoder_prepare_resident(gpu,s,error,sizeof(error)));
    FILE *checkpoint=tmpfile(),*continuous=tmpfile();CHECK(checkpoint && continuous);
    float first=0,last=0;double maximum=0;
    for(unsigned step=0;step<100;++step) {
        for(size_t i=0;i<count;++i)nya_train_zero_grad(cp[i]);
        CHECK(!nya_train_session_zero_grad(s));float expected,actual;
        CHECK(!one_step(cpu,NULL,&expected,0) && !one_step(gpu,s,&actual,0));
        CHECK(fabs((double)expected-actual)<=1e-5*(1+fabs(expected)));
        if(step==0)first=actual;
        last=actual;
        CHECK(!nya_train_adamw_step(&optimizer,cp,count,error,sizeof(error)) && !nya_train_session_step(s,settings(optimizer)));
        nya_train_session_metrics metrics;CHECK(!nya_train_session_observe(s,&metrics) && metrics.step==optimizer.step);
        FILE *a=tmpfile(),*b=tmpfile();CHECK(a && b);
        CHECK(!nya_train_checkpoint_write(a,&optimizer,cp,count) && !nya_train_session_checkpoint_write(s,b) && !compare(a,b,&maximum));
        fclose(a);fclose(b);
        if(step==49)CHECK(!nya_train_session_checkpoint_write(s,checkpoint));
    }
    CHECK(first>5 && last<0.02f && !nya_train_session_checkpoint_write(s,continuous));
    CHECK(!nya_train_session_detach(s,&initial));nya_train_session_free(s);
    /* Exact restart uses a fresh decoder and session, not retained GPU buffers. */
    nya_train_decoder_free(gpu);gpu=nya_train_decoder_create(&c,error,sizeof(error));CHECK(gpu);
    gp=nya_train_decoder_parameters(gpu,&gcount);rewind(checkpoint);CHECK(!nya_train_checkpoint_read(checkpoint,&initial,gp,gcount) && initial.step==50);
    s=nya_train_session_create("cuda",16*1024*1024,&initial,gp,gcount,error,sizeof(error));CHECK(s);
    CHECK(!nya_train_decoder_prepare_resident(gpu,s,error,sizeof(error)));
    for(unsigned step=50;step<100;++step) {
        CHECK(!nya_train_session_zero_grad(s));float value;CHECK(!one_step(gpu,s,&value,0) && !nya_train_session_step(s,settings(initial)));
        nya_train_session_metrics m;CHECK(!nya_train_session_observe(s,&m) && m.step==step+1);
    }
    FILE *resumed=tmpfile();CHECK(resumed && !nya_train_session_checkpoint_write(s,resumed) && same_file(continuous,resumed));
    /* Forward-only evaluation leaves every persistent array/counter unchanged. */
    nya_train_graph *eval=nya_train_graph_create_resident(16*1024*1024,s,1);CHECK(eval);
    nya_train_tensor *eval_loss=nya_train_cross_entropy(nya_train_decoder_forward(gpu,eval,tokens,12),labels,NULL,12);CHECK(eval_loss && nya_train_data(eval_loss));
    nya_train_graph_free(eval);FILE *after_eval=tmpfile();CHECK(after_eval && !nya_train_session_checkpoint_write(s,after_eval) && same_file(resumed,after_eval));fclose(after_eval);
    CHECK(!nya_train_graph_create_resident(1,s,1) && !nya_train_session_discard_evaluation(s));
    eval=nya_train_graph_create_resident(65536,s,1);CHECK(eval);
    float huge=0x1.fffffep127f;
    nya_train_tensor *invalid=nya_train_scale(nya_train_input(eval,1,1,&huge),2);CHECK(invalid && !nya_train_data(invalid));
    nya_train_graph_free(eval);CHECK(!nya_train_session_discard_evaluation(s));
    after_eval=tmpfile();CHECK(after_eval && !nya_train_session_checkpoint_write(s,after_eval) && same_file(resumed,after_eval));fclose(after_eval);
    CHECK(!nya_train_session_detach(s,&initial) && initial.step==100);nya_train_session_free(s);
    snprintf(path,sizeof(path),"%s.trained.gguf",prefix);FILE *exported=fopen(path,"wb");CHECK(exported && !nya_train_decoder_export(gpu,exported,error,sizeof(error)) && !fclose(exported));
    nya_model_registry registry;nya_model_registry_init(&registry);const nya_model *model;
    CHECK(nya_model_load(&registry,path,&model)==NYA_MODEL_OK && model->generation_supported);
    nya_generation_request request={0};nya_generation_response response;
    request.prompt="a";request.max_tokens=8;request.max_output_bytes=64;request.top_p=1;
    CHECK(!nya_generation_run((nya_model *)model,&request,&response,error,sizeof(error)) && !strcmp(response.text,"bcabcabc"));
    nya_generation_response_free(&response);
    printf("Full GPU decoder: loss %.9g -> %.9g, maximum CPU state error %.9g; exact resume/evaluation; exported generation bcabcabc\n",(double)first,(double)last,maximum);
    nya_model_registry_shutdown(&registry);nya_train_decoder_free(cpu);nya_train_decoder_free(gpu);fclose(checkpoint);fclose(continuous);fclose(resumed);
    return 0;
}
static int failures(int injected)
{
    char error[160]={0};float initial=2;nya_train_parameter *p=nya_train_parameter_create(1,1,&initial);CHECK(p);
    nya_train_adamw o;nya_train_adamw_defaults(&o);
    nya_train_session *s=nya_train_session_create("cuda",1024*1024,&o,&p,1,error,sizeof(error));CHECK(s);
    nya_train_graph *g=nya_train_graph_create_resident(65536,s,0);
    if(injected) {
        if(g) {
            nya_train_tensor *loss=nya_train_scale(nya_train_leaf(g,p),2);
            CHECK(!loss || nya_train_backward(loss)<0);nya_train_graph_free(g);
        }
        CHECK(nya_train_session_step(s,settings(o))<0);
    } else {
        CHECK(g && !nya_train_graph_create_resident(65536,s,0));
        CHECK(nya_train_session_zero_grad(s)<0 && nya_train_session_step(s,settings(o))<0 && nya_train_session_detach(s,&o)<0);
        nya_train_tensor *leaf=nya_train_leaf(g,p);CHECK(leaf && !nya_train_slice_columns(leaf,1,1));
        nya_train_graph_free(g);CHECK(nya_train_session_discard_evaluation(s)<0 && nya_train_session_step(s,settings(o))<0 && !nya_train_session_zero_grad(s));
        g=nya_train_graph_create_resident(65536,s,0);CHECK(g);
        nya_train_tensor *loss=nya_train_scale(nya_train_leaf(g,p),3);CHECK(loss && !nya_train_backward(loss) && nya_train_data(loss)[0]==6);
        nya_train_graph_free(g);CHECK(!nya_train_session_step(s,settings(o)));
        nya_train_session_metrics m;CHECK(!nya_train_session_observe(s,&m) && m.step==1);
    }
    nya_train_session_free(s);CHECK(nya_train_parameter_data(p)[0]==initial);nya_train_parameter_free(p);return 0;
}
/* Cancellation must retain the CPU's separately rounded multiply/add. A
   fused multiply-add leaves a tiny residual that AdamW can amplify. */
static int cancellation(void)
{
    char error[160]={0};float value=2,coefficient=1-0x1p-23f;
    nya_train_parameter *p=nya_train_parameter_create(1,1,&value);CHECK(p);
    nya_train_graph *g=nya_train_graph_create(65536);CHECK(g);
    CHECK(!nya_train_backward(nya_train_scale(nya_train_leaf(g,p),-1)));
    nya_train_graph_free(g);CHECK(nya_train_parameter_gradient(p)[0]==-1);
    nya_train_adamw o;nya_train_adamw_defaults(&o);
    nya_train_session *s=nya_train_session_create("cuda",1024*1024,&o,&p,1,error,sizeof(error));CHECK(s);
    g=nya_train_graph_create_resident(65536,s,0);CHECK(g);
    nya_train_tensor *product=nya_train_mul(nya_train_leaf(g,p),nya_train_input(g,1,1,&coefficient));
    CHECK(product && !nya_train_backward(nya_train_scale(product,1+0x1p-23f)));
    nya_train_graph_free(g);CHECK(!nya_train_session_detach(s,&o));nya_train_session_free(s);
    CHECK(nya_train_parameter_gradient(p)[0]==0);nya_train_parameter_free(p);return 0;
}
static int slice_and_lifetime(void)
{
    char error[160]={0};float values[]={1,2,3,4,5,6},upstream[]={2,-3},result[6];
    nya_train_device *d=nya_train_device_create("cuda",65536);CHECK(d);
    nya_train_buffer x=nya_train_device_alloc(d,sizeof(values)),y=nya_train_device_alloc(d,sizeof(upstream)),dx=nya_train_device_alloc(d,sizeof(values));
    CHECK(x && y && dx && !nya_train_device_write(d,x,0,values,sizeof(values)));
    CHECK(nya_train_device_slice(d,y,x,2,3,3,1,0)<0 && nya_train_device_slice(d,x,x,2,3,1,1,0)<0);
    CHECK(nya_train_device_slice(d,y,x,SIZE_MAX,3,1,1,0)<0 && nya_train_device_slice(d,y,x,2,3,1,2,0)<0);
    CHECK(nya_train_device_slice(d,y,x,2,3,1,1,2)<0 && nya_train_device_slice(d,y,x,0,3,1,1,0)<0);
    CHECK(!nya_train_device_slice(d,y,x,2,3,2,1,0) && !nya_train_device_read(d,y,0,result,sizeof(upstream)) && result[0]==3 && result[1]==6);
    CHECK(!nya_train_device_write(d,y,0,upstream,sizeof(upstream)) && !nya_train_device_write(d,dx,0,values,sizeof(values)));
    CHECK(!nya_train_device_slice(d,dx,y,2,3,2,1,1) && !nya_train_device_read(d,dx,0,result,sizeof(result)));
    for(size_t i=0;i<6;++i)CHECK(result[i]==values[i]+(i==2?2:i==5?-3:0));
    nya_train_device_free(d);
    nya_train_parameter *p=nya_train_parameter_create(2,3,values);CHECK(p);
    nya_train_adamw o;nya_train_adamw_defaults(&o);
    nya_train_session *s=nya_train_session_create("cuda",1024*1024,&o,&p,1,error,sizeof(error));CHECK(s);
    CHECK(!nya_train_graph_create_resident(1,s,0) && !nya_train_session_zero_grad(s));
    nya_train_graph *g=nya_train_graph_create_resident(65536,s,1);CHECK(g);
    nya_train_tensor *out=nya_train_slice_columns(nya_train_leaf(g,p),1,2);CHECK(out);
    /* The graph retains session/parameter storage after caller ownership ends. */
    nya_train_session_free(s);nya_train_parameter_free(p);
    const float *data=nya_train_data(out);CHECK(data && data[0]==2 && data[1]==3 && data[2]==5 && data[3]==6);
    nya_train_graph_free(g);return 0;
}
static nya_train_tensor *model_objective(nya_train_decoder *m,nya_train_graph *g,const uint32_t *ids,
    const uint32_t *target,const uint32_t *rejected,size_t n,int dpo,double rc,double rr)
{
    unsigned char mask[12]={0};for(size_t i=2;i<n;++i)mask[i]=1;
    nya_train_tensor *chosen=nya_train_decoder_forward(m,g,ids,n);
    if(!dpo)return nya_train_cross_entropy(chosen,target,mask,n);
    return nya_train_dpo(nya_train_logprob(chosen,target,mask,n),
        nya_train_logprob(nya_train_decoder_forward(m,g,rejected,n),ids,mask,n),rc,rr,0.2f);
}
static int model_probe(const char *path,const char *prefix,size_t rank,int real)
{
    char error[256]={0},export_path[1200];nya_model_registry registry;nya_model_registry_init(&registry);const nya_model *base;
    CHECK(nya_model_load(&registry,path,&base)==NYA_MODEL_OK && base->generation_supported);
    nya_train_decoder *cpu=nya_train_decoder_from_model((nya_model *)base,rank,4,256*1024*1024,error,sizeof(error)),
        *gpu=nya_train_decoder_from_model((nya_model *)base,rank,4,256*1024*1024,error,sizeof(error));CHECK(cpu && gpu);
    size_t count,gcount;nya_train_parameter *const *cp=nya_train_decoder_parameters(cpu,&count),*const *gp=nya_train_decoder_parameters(gpu,&gcount);CHECK(count==gcount);
    nya_train_executor *executor=nya_train_executor_create(6);CHECK(executor);
    uint32_t ids[12]={1,3,4,5},target[12]={3,4,5,3},rejected[12]={1,5,4,3};size_t n=4;
    if(real) {
        uint32_t *encoded=NULL;size_t encoded_count=0;
        CHECK(!nya_train_decoder_tokenize(cpu,"The small library beside the river opens each morning.",&encoded,&encoded_count,error,sizeof(error)) && encoded_count>=9);
        n=8;for(size_t i=0;i<n;++i){ids[i]=encoded[i];target[i]=encoded[i+1];rejected[i]=encoded[n-i];}free(encoded);
    }
    unsigned char mask[12]={0};for(size_t i=2;i<n;++i)mask[i]=1;
    nya_train_graph *reference=nya_train_graph_create_for_evaluation(512*1024*1024,executor);CHECK(reference);
    nya_train_tensor *chosen=nya_train_logprob(nya_train_decoder_forward(cpu,reference,ids,n),target,mask,n),
        *reject=nya_train_logprob(nya_train_decoder_forward(cpu,reference,rejected,n),ids,mask,n);CHECK(chosen && reject);
    double rc=nya_train_data(chosen)[0],rr=nya_train_data(reject)[0];nya_train_graph_free(reference);
    nya_train_adamw o;nya_train_adamw_defaults(&o);o.learning_rate=0.003f;
    nya_train_session *session=nya_train_session_create("cuda",real?(size_t)3*1024*1024*1024:32*1024*1024,&o,gp,count,error,sizeof(error));CHECK(session);
    CHECK(!nya_train_decoder_prepare_resident(gpu,session,error,sizeof(error)));
    nya_train_device_stats prepared,repeated;nya_train_device_get_stats(nya_train_session_device(session),&prepared);
    CHECK(!nya_train_decoder_prepare_resident(gpu,session,error,sizeof(error)));
    nya_train_device_get_stats(nya_train_session_device(session),&repeated);
    CHECK(repeated.uploads==prepared.uploads && repeated.used_bytes==prepared.used_bytes);
    double maximum=0;
    for(unsigned step=0;step<3;++step) {
        for(size_t i=0;i<count;++i)nya_train_zero_grad(cp[i]);
        CHECK(!nya_train_session_zero_grad(session));
        nya_train_graph *cg=nya_train_graph_create_with_executor(512*1024*1024,executor),
            *gg=nya_train_graph_create_resident(512*1024*1024,session,0);CHECK(cg && gg);
        nya_train_tensor *cl=model_objective(cpu,cg,ids,target,rejected,n,step==2,rc,rr),
            *gl=model_objective(gpu,gg,ids,target,rejected,n,step==2,rc,rr);
        snprintf(error,sizeof(error),"CPU: %s; GPU: %s",nya_train_error(cg),nya_train_error(gg));CHECK(cl && gl);
        CHECK(!nya_train_backward(cl));nya_train_device_stats before,after;nya_train_device_get_stats(nya_train_session_device(session),&before);
        int result=nya_train_backward(gl);snprintf(error,sizeof(error),"%s",nya_train_error(gg));CHECK(!result);
        nya_train_device_get_stats(nya_train_session_device(session),&after);
        CHECK(after.uploads==before.uploads && after.downloads==before.downloads && after.synchronizations==before.synchronizations && after.used_bytes==before.used_bytes);
        const float *actual=nya_train_data(gl);snprintf(error,sizeof(error),"%s",nya_train_error(gg));CHECK(actual);
        float expected=nya_train_data(cl)[0];CHECK(fabs((double)expected-*actual)<=1e-5*(1+fabs(expected)));
        printf("model=%s rank=%zu step=%u objective=%s loss=%.9g buffers=%zu graph_bytes=%zu inner_transfers=0 inner_fences=0\n",
            path,rank,step,step==2?"DPO":"masked-CE",(double)*actual,after.buffers,nya_train_memory_used(gg));fflush(stdout);
        nya_train_graph_free(cg);nya_train_graph_free(gg);
        CHECK(!nya_train_adamw_step(&o,cp,count,error,sizeof(error)) && !nya_train_session_step(session,settings(o)));
        nya_train_session_metrics metrics;CHECK(!nya_train_session_observe(session,&metrics) && metrics.step==o.step);
        FILE *a=tmpfile(),*b=tmpfile();CHECK(a && b && !nya_train_checkpoint_write(a,&o,cp,count) && !nya_train_session_checkpoint_write(session,b) && !compare(a,b,&maximum));fclose(a);fclose(b);
    }
    CHECK(!nya_train_session_detach(session,&o));nya_train_session_free(session);
    snprintf(export_path,sizeof(export_path),"%s.rank-%zu.gguf",prefix,rank);FILE *file=fopen(export_path,"wb");CHECK(file && !nya_train_decoder_export(gpu,file,error,sizeof(error)) && !fclose(file));
    const nya_model *exported;CHECK(nya_model_load(&registry,export_path,&exported)==NYA_MODEL_OK && exported->generation_supported);
    nya_train_graph *eval=nya_train_graph_create_for_evaluation(512*1024*1024,executor);CHECK(eval);
    nya_train_tensor *logits=nya_train_decoder_forward(gpu,eval,ids,n);CHECK(logits);
    /* Compare the independently CPU-trained model after the same merge/export
       boundary. A packed base plus separate LoRA and merged F32 projection
       have different rounding; retain that diagnostic separately below. */
    const nya_model *cpu_exported=NULL;nya_llm_run_state cpu_state={0};nya_llm_context *cpu_context=NULL;
    if(real) {
        snprintf(export_path,sizeof(export_path),"%s.cpu.rank-%zu.gguf",prefix,rank);
        file=fopen(export_path,"wb");CHECK(file && !nya_train_decoder_export(cpu,file,error,sizeof(error)) && !fclose(file));
        CHECK(nya_model_load(&registry,export_path,&cpu_exported)==NYA_MODEL_OK && cpu_exported->generation_supported);
        cpu_context=(nya_llm_context *)cpu_exported->generation_context;
        CHECK(!nya_llm_state_create(cpu_context,n,&cpu_state,error,sizeof(error)));
    }
    nya_llm_context *context=(nya_llm_context *)exported->generation_context;nya_llm_run_state state;
    CHECK(!nya_llm_state_create(context,n,&state,error,sizeof(error)));
    double export_parity=0,logit_error=0,absolute_error=0;size_t worst_row=0,worst_column=0;double worst_expected=0,worst_actual=0;
    for(size_t i=0;i<n;++i) {
        CHECK(!nya_llm_forward(context,&state,ids[i],i,1));
        if(real)CHECK(!nya_llm_forward(cpu_context,&cpu_state,ids[i],i,1));
        for(size_t j=0;j<nya_train_columns(logits);++j) {
            double v=nya_train_data(logits)[i*nya_train_columns(logits)+j];
            double absolute=fabs(v-state.logits[j]),delta=absolute/(1+fabs(v));
            CHECK(isfinite(state.logits[j]));
            double aligned=real?cpu_state.logits[j]:v;
            double aligned_delta=fabs(aligned-state.logits[j])/(1+fabs(aligned));
            CHECK(isfinite(aligned) && aligned_delta<=1e-5);
            if(aligned_delta>export_parity)export_parity=aligned_delta;
            if(absolute>absolute_error)absolute_error=absolute;
            if(delta>logit_error){logit_error=delta;worst_row=i;worst_column=j;worst_expected=v;worst_actual=state.logits[j];}
        }
    }
    printf("model state maximum_scaled_error=%.9g exported_inference_maximum_scaled_error=%.9g\n",maximum,logit_error);fflush(stdout);
    printf("export worst row=%zu column=%zu training=%.9g inference=%.9g maximum_absolute_error=%.9g\n",worst_row,worst_column,worst_expected,worst_actual,absolute_error);fflush(stdout);
    printf("same-representation exported inference maximum_scaled_error=%.9g\n",export_parity);fflush(stdout);
    if(real)nya_llm_state_free(&cpu_state);
    nya_llm_state_free(&state);nya_train_graph_free(eval);nya_train_executor_free(executor);nya_train_decoder_free(cpu);nya_train_decoder_free(gpu);nya_model_registry_shutdown(&registry);return 0;
}
int main(int argc,char **argv)
{
    char error[160]={0};CHECK(!interface());if(argc==1)return 0;
    nya_train_device *probe=nya_train_device_create("cuda",1024*1024);if(!probe)return 77;nya_train_device_free(probe);
    if(!strcmp(argv[1],"--failure"))return failures(1);
    if(argc==4 && !strcmp(argv[1],"--model"))return model_probe(argv[2],argv[3],2,1);
    if(argc==4 && !strcmp(argv[1],"--gemma"))return model_probe(argv[2],argv[3],0,0) || model_probe(argv[2],argv[3],2,0);
    CHECK(argc==3 && !strcmp(argv[1],"--cuda"));
    CHECK(!failures(0) && !cancellation() && !slice_and_lifetime() && !lifecycle(argv[2]));return 0;
}
