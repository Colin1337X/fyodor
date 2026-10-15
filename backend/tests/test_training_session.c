#include "training_session.h"
#include "pretraining.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define CHECK(x) do { if (!(x)) { fprintf(stderr,"session:%d: %s\n",__LINE__,#x); return 1; } } while (0)

static nya_train_adamw_config settings(unsigned step)
{ return (nya_train_adamw_config){0.025f/(1+0.03f*(float)step),0.9f,0.99f,1e-8f,0.013f,0.2f}; }
static void configure(nya_train_adamw *o,unsigned step)
{
    nya_train_adamw_config c=settings(step);
    o->learning_rate=c.learning_rate; o->beta1=c.beta1; o->beta2=c.beta2;
    o->epsilon=c.epsilon; o->weight_decay=c.weight_decay; o->max_grad_norm=c.max_grad_norm;
}
static int parameters(nya_train_parameter **p)
{
    float w[20],b[4];
    for (size_t i=0;i<20;++i) w[i]=(float)((int)i-9)/64;
    for (size_t i=0;i<4;++i) b[i]=(float)i/128;
    p[0]=nya_train_parameter_create(4,5,w); p[1]=nya_train_parameter_create(1,4,b);
    CHECK(p[0] && p[1]); return 0;
}
static int same_file(FILE *a,FILE *b)
{
    rewind(a); rewind(b); int x,y;
    do { x=fgetc(a); y=fgetc(b); if (x!=y) return 0; } while (x!=EOF);
    return !ferror(a) && !ferror(b);
}
static int cpu_snapshot(FILE *f,nya_train_adamw *o,nya_train_parameter **p)
{ CHECK(f && !nya_train_checkpoint_write(f,o,p,2)); return 0; }
static int interfaces(void)
{
    nya_train_adamw o; nya_train_adamw_defaults(&o); nya_train_adamw original=o;
    nya_train_parameter *p[2]; CHECK(!parameters(p)); char error[160]={0};
    CHECK(!nya_train_session_create(NULL,1024,&o,p,2,error,sizeof(error)) && error[0]);
    CHECK(!nya_train_session_create("absent",1024,&o,p,2,error,sizeof(error)));
    CHECK(!nya_train_session_create("cuda",0,&o,p,2,error,sizeof(error)));
    CHECK(!nya_train_session_create("cuda",1024,NULL,p,2,error,sizeof(error)));
    CHECK(!nya_train_session_create("cuda",1024,&o,NULL,2,error,sizeof(error)));
    CHECK(!nya_train_session_create("cuda",1024,&o,p,0,error,sizeof(error)));
    CHECK(!nya_train_session_create("cuda",1024,&o,p,SIZE_MAX,error,sizeof(error)));
    nya_train_parameter *duplicates[]={p[0],p[0]};
    CHECK(!nya_train_session_create("cuda",1024,&o,duplicates,2,error,sizeof(error)));
    o.beta2=1; CHECK(!nya_train_session_create("cuda",1024,&o,p,2,error,sizeof(error))); o=original;
    float *host=nya_train_parameter_data(p[0]),saved=host[19]; host[19]=NAN;
    CHECK(!nya_train_session_create("cuda",1024,&o,p,2,error,sizeof(error))); host[19]=saved;
    for (int evaluation=0;evaluation<2;++evaluation) {
        nya_train_graph *g=evaluation?nya_train_graph_create_for_evaluation(4096,NULL):nya_train_graph_create(4096);
        CHECK(g && nya_train_leaf(g,p[1]));
        CHECK(!nya_train_session_create("cuda",1024,&o,p,2,error,sizeof(error)));
        nya_train_graph_free(g);
    }
    nya_train_session_metrics m={1,2,3},prior=m;
    CHECK(nya_train_session_observe(NULL,&m)<0 && !memcmp(&m,&prior,sizeof(m)));
    CHECK(nya_train_session_detach(NULL,&o)<0 && !memcmp(&o,&original,sizeof(o)));
    CHECK(!nya_train_session_device(NULL) && !nya_train_session_status(NULL));
    nya_train_adamw_tensor t={0};
    CHECK(nya_train_session_tensor(NULL,0,&t)<0 && nya_train_session_zero_grad(NULL)<0);
    CHECK(nya_train_session_step(NULL,settings(0))<0 && nya_train_session_checkpoint_write(NULL,NULL)<0);
    CHECK(nya_train_session_error(NULL)[0]); nya_train_session_free(NULL);
    /* CPU graph retention is independently covered by the host sanitizers. */
    float initial=2;
    nya_train_parameter *scalar=nya_train_parameter_create(1,1,&initial);
    nya_train_graph *g=nya_train_graph_create(4096); CHECK(scalar && g);
    nya_train_tensor *leaf=nya_train_leaf(g,scalar),*loss=nya_train_mul(leaf,leaf); CHECK(loss);
    nya_train_parameter_free(scalar);
    CHECK(nya_train_data(leaf)[0]==2 && !nya_train_backward(loss)); nya_train_graph_free(g);
    nya_train_parameter_free(p[0]); nya_train_parameter_free(p[1]); return 0;
}

static int ownership(void)
{
    nya_train_parameter *p[2]; CHECK(!parameters(p));
    nya_train_adamw o; nya_train_adamw_defaults(&o); char error[160];
    FILE *before=tmpfile(),*after=tmpfile(); CHECK(!cpu_snapshot(before,&o,p));
    CHECK(!nya_train_session_create("cuda",1,&o,p,2,error,sizeof(error)));
    CHECK(nya_train_parameter_data(p[0]) && nya_train_parameter_gradient(p[1]));
    nya_train_session *s=nya_train_session_create("cuda",1024*1024,&o,p,2,error,sizeof(error)); CHECK(s);
    CHECK(!nya_train_parameter_data(p[0]) && !nya_train_parameter_gradient(p[1]));
    CHECK(!nya_train_session_create("cuda",1024*1024,&o,p,2,error,sizeof(error)));
    CHECK(nya_train_adamw_step(&o,p,2,error,sizeof(error))<0 && o.step==0);
    CHECK(nya_train_checkpoint_write(after,&o,p,2)<0 && ftell(after)==0);
    rewind(before); CHECK(nya_train_checkpoint_read(before,&o,p,2)<0 && ftell(before)==0);
    for (int evaluation=0;evaluation<2;++evaluation) {
        nya_train_graph *g=evaluation?nya_train_graph_create_for_evaluation(4096,NULL):nya_train_graph_create(4096);
        CHECK(g && !nya_train_leaf(g,p[1]) && strstr(nya_train_error(g),"resident")); nya_train_graph_free(g);
    }
    nya_train_adamw_tensor tensor; CHECK(!nya_train_session_tensor(s,1,&tensor));
    CHECK(nya_train_session_tensor(s,2,&tensor)<0 && nya_train_session_tensor(s,0,NULL)<0);
    float gradient[]={1,2,3,4},actual[4]; nya_train_device *d=nya_train_session_device(s); CHECK(d);
    CHECK(!nya_train_device_write(d,tensor.gradient,0,gradient,sizeof(gradient)));
    nya_train_zero_grad(p[1]);
    CHECK(!nya_train_device_read(d,tensor.gradient,0,actual,sizeof(actual)) && !memcmp(actual,gradient,sizeof(actual)));
    CHECK(!nya_train_session_step(s,settings(0)));
    CHECK(nya_train_session_step(s,settings(1))<0 && nya_train_session_zero_grad(s)<0);
    nya_train_session_metrics m; CHECK(!nya_train_session_observe(s,&m) && m.step==1 && m.status==0);
    /* Abort must preserve all original CPU arrays and optimizer state. */
    nya_train_session_free(s); CHECK(!cpu_snapshot(after,&o,p) && same_file(before,after));
    fclose(before); fclose(after);
    for (int iteration=0;iteration<12;++iteration) {
        s=nya_train_session_create("cuda",1024*1024,&o,p,2,error,sizeof(error)); CHECK(s);
        nya_train_scope scope=nya_train_device_scratch_begin(nya_train_session_device(s)); CHECK(scope);
        CHECK(nya_train_device_alloc(nya_train_session_device(s),8192));
        CHECK(!nya_train_device_scratch_end(nya_train_session_device(s),scope));
        CHECK(!nya_train_session_detach(s,&o) && !nya_train_session_device(s) && !nya_train_session_status(s));
        CHECK(nya_train_session_detach(s,&o)<0 && nya_train_session_step(s,settings(0))<0);
        nya_train_session_free(s);
    }
    s=nya_train_session_create("cuda",1024*1024,&o,p,2,error,sizeof(error)); CHECK(s);
    nya_train_parameter_free(p[0]); nya_train_parameter_free(p[1]);
    CHECK(!nya_train_session_detach(s,&o)); nya_train_session_free(s);
    return 0;
}

static int transaction_failures(void)
{
    nya_train_parameter *p[2]; CHECK(!parameters(p)); nya_train_adamw o; nya_train_adamw_defaults(&o);
    char error[160]; nya_train_session *s=nya_train_session_create("cuda",1024*1024,&o,p,2,error,sizeof(error)); CHECK(s);
    nya_train_device *d=nya_train_session_device(s); nya_train_adamw_tensor t; CHECK(!nya_train_session_tensor(s,1,&t));
    float bad=NAN,good=0,values[4];
    CHECK(!nya_train_device_write(d,t.gradient,12,&bad,4) && !nya_train_session_step(s,settings(0)));
    nya_train_session_metrics m;
    CHECK(nya_train_session_observe(s,&m)<0 && m.step==0 && m.status==1);
    FILE *f=tmpfile(); CHECK(f && nya_train_session_checkpoint_write(s,f)<0 && ftell(f)==0);
    nya_train_adamw untouched=o;
    CHECK(nya_train_session_detach(s,&o)<0 && !memcmp(&o,&untouched,sizeof(o)) && !nya_train_parameter_data(p[0]));
    CHECK(!nya_train_session_zero_grad(s));
    CHECK(!nya_train_device_read(d,t.gradient,0,values,sizeof(values)));
    for (size_t i=0;i<4;++i) CHECK(values[i]==0);
    /* Snapshot validation catches invalid state even without an optimizer step
       or a previously queued finite check, and writes no checkpoint prefix. */
    nya_train_buffer ids[]={t.values,t.gradient,t.moment,t.variance};
    for (size_t a=0;a<4;++a) {
        float old=0; CHECK(!nya_train_device_read(d,ids[a],12,&old,4));
        bad=a==3?-1:NAN; CHECK(!nya_train_device_write(d,ids[a],12,&bad,4));
        CHECK(nya_train_session_checkpoint_write(s,f)<0 && ftell(f)==0);
        CHECK(nya_train_session_detach(s,&o)<0 && !memcmp(&o,&untouched,sizeof(o)) && !nya_train_parameter_data(p[1]));
        CHECK(!nya_train_device_write(d,ids[a],12,&old,4));
    }
    uint32_t tag=73;
    CHECK(!nya_train_device_write(d,nya_train_session_status(s),0,&tag,4));
    CHECK(!nya_train_session_step(s,settings(0)) && nya_train_session_observe(s,&m)<0 && m.status==73 && m.step==0);
    CHECK(!nya_train_session_zero_grad(s));
    CHECK(!nya_train_device_write(d,t.gradient,0,&good,4));
    CHECK(!nya_train_session_step(s,settings(0)) && !nya_train_session_checkpoint_write(s,f));
    rewind(f); nya_train_parameter *copy[2]; CHECK(!parameters(copy));
    nya_train_adamw restored; CHECK(!nya_train_checkpoint_read(f,&restored,copy,2) && restored.step==1);
    CHECK(!nya_train_session_detach(s,&o) && o.step==1 && o.learning_rate==settings(0).learning_rate);
    FILE *after=tmpfile(); CHECK(!cpu_snapshot(after,&o,p) && same_file(f,after));
    fclose(after); fclose(f); nya_train_session_free(s);
    for (size_t i=0;i<2;++i) { nya_train_parameter_free(p[i]); nya_train_parameter_free(copy[i]); }
    return 0;
}

static int failed_detach_preserves_host(void)
{
    nya_train_parameter *p[2]; CHECK(!parameters(p)); nya_train_adamw o; nya_train_adamw_defaults(&o);
    FILE *before=tmpfile(),*after=tmpfile(); CHECK(!cpu_snapshot(before,&o,p));
    char error[160]; nya_train_session *s=nya_train_session_create("cuda",1024*1024,&o,p,2,error,sizeof(error)); CHECK(s);
    nya_train_device *d=nya_train_session_device(s); nya_train_adamw_tensor first,last;
    CHECK(!nya_train_session_tensor(s,0,&first) && !nya_train_session_tensor(s,1,&last));
    float changed=123,invalid=-1;
    CHECK(!nya_train_device_write(d,first.values,0,&changed,4) && !nya_train_device_write(d,last.variance,12,&invalid,4));
    CHECK(nya_train_session_detach(s,&o)<0 && o.step==0);
    CHECK(nya_train_session_checkpoint_write(s,after)<0 && ftell(after)==0);
    nya_train_session_free(s);
    CHECK(!cpu_snapshot(after,&o,p) && same_file(before,after)); fclose(before); fclose(after);
    o.step=UINT64_MAX;
    s=nya_train_session_create("cuda",1024*1024,&o,p,2,error,sizeof(error)); CHECK(s);
    CHECK(!nya_train_session_step(s,settings(0)));
    nya_train_session_metrics m; CHECK(nya_train_session_observe(s,&m)<0 && m.status && m.step==UINT64_MAX);
    CHECK(!nya_train_session_zero_grad(s) && !nya_train_session_detach(s,&o) && o.step==UINT64_MAX);
    nya_train_session_free(s); nya_train_parameter_free(p[0]); nya_train_parameter_free(p[1]); return 0;
}

static void data(float *x,uint32_t *labels,unsigned group)
{
    for (size_t i=0;i<35;++i) x[i]=(float)((int)((i*11+group*7)%23)-11)/8;
    for (size_t i=0;i<7;++i) labels[i]=(uint32_t)((i+group)%4);
}
static int cpu_group(nya_train_parameter **p,unsigned group,float *loss)
{
    float x[35]; uint32_t labels[7]; data(x,labels,group);
    nya_train_graph *g=nya_train_graph_create(65536); CHECK(g);
    nya_train_tensor *logits=nya_train_add(nya_train_linear(nya_train_input(g,7,5,x),nya_train_leaf(g,p[0])),nya_train_leaf(g,p[1]));
    nya_train_tensor *objective=nya_train_cross_entropy(logits,labels,NULL,7); CHECK(objective);
    *loss=nya_train_data(objective)[0]; CHECK(!nya_train_backward(objective)); nya_train_graph_free(g); return 0;
}
static int gpu_group(nya_train_session *s,unsigned group,float *loss)
{
    nya_train_device *d=nya_train_session_device(s); nya_train_adamw_tensor w,b;
    CHECK(!nya_train_session_tensor(s,0,&w) && !nya_train_session_tensor(s,1,&b));
    nya_train_scope scope=nya_train_device_scratch_begin(d); CHECK(scope);
    nya_train_buffer x=nya_train_device_alloc(d,35*4),labels=nya_train_device_alloc(d,7*4),linear=nya_train_device_alloc(d,28*4),
        logits=nya_train_device_alloc(d,28*4),objective=nya_train_device_alloc(d,4),state=nya_train_device_alloc(d,nya_train_loss_state_bytes(7)),
        dlogits=nya_train_device_alloc(d,28*4),dlinear=nya_train_device_alloc(d,28*4),seed=nya_train_device_alloc(d,4);
    CHECK(x && labels && linear && logits && objective && state && dlogits && dlinear && seed);
    float input[35],one=1; uint32_t target[7]; data(input,target,group);
    CHECK(!nya_train_device_write(d,x,0,input,sizeof(input)) && !nya_train_device_write(d,labels,0,target,sizeof(target)) &&
        !nya_train_device_write(d,seed,0,&one,4));
    nya_train_device_stats before,after; nya_train_device_get_stats(d,&before);
    CHECK(!nya_train_device_linear(d,linear,w.values,0,4,5,x,7));
    CHECK(!nya_train_device_binary(d,logits,(nya_train_view){linear,7,4},(nya_train_view){b.values,1,4},NYA_TRAIN_BINARY_ADD));
    CHECK(!nya_train_device_loss(d,objective,state,(nya_train_view){logits,7,4},labels,0,NYA_TRAIN_LOSS_CE));
    CHECK(!nya_train_device_check_finite(d,nya_train_session_status(s),objective,1,9));
    CHECK(!nya_train_device_loss_backward(d,dlogits,state,(nya_train_view){logits,7,4},labels,0,seed));
    CHECK(!nya_train_device_binary_backward(d,dlinear,b.gradient,(nya_train_view){linear,7,4},(nya_train_view){b.values,1,4},dlogits,NYA_TRAIN_BINARY_ADD));
    CHECK(!nya_train_device_linear_dw(d,w.gradient,x,dlinear,4,5,7));
    nya_train_device_get_stats(d,&after);
    CHECK(after.uploads==before.uploads && after.downloads==before.downloads && after.synchronizations==before.synchronizations && after.used_bytes==before.used_bytes);
    CHECK(!nya_train_device_read(d,objective,0,loss,4) && !nya_train_device_scratch_end(d,scope)); return 0;
}
static int close_checkpoint(FILE *cpu,FILE *gpu)
{
    rewind(cpu); rewind(gpu); unsigned char a[48],b[48];
    CHECK(fread(a,1,48,cpu)==48 && fread(b,1,48,gpu)==48 && !memcmp(a,b,48));
    for (size_t p=0;p<2;++p) {
        CHECK(fread(a,1,16,cpu)==16 && fread(b,1,16,gpu)==16 && !memcmp(a,b,16));
        for (size_t i=0;i<(p==0?20U:4U)*4;++i) {
            CHECK(fread(a,1,4,cpu)==4 && fread(b,1,4,gpu)==4);
            uint32_t av=(uint32_t)a[0]|(uint32_t)a[1]<<8|(uint32_t)a[2]<<16|(uint32_t)a[3]<<24;
            uint32_t bv=(uint32_t)b[0]|(uint32_t)b[1]<<8|(uint32_t)b[2]<<16|(uint32_t)b[3]<<24;
            float af,bf; memcpy(&af,&av,4); memcpy(&bf,&bv,4);
            CHECK(isfinite(bf) && fabs((double)af-bf)<=1e-6*(1+fabs(af)));
        }
    }
    return 0;
}
static int trajectory(void)
{
    nya_train_parameter *cpu[2],*gpu[2]; CHECK(!parameters(cpu) && !parameters(gpu));
    nya_train_adamw o; nya_train_adamw_defaults(&o); configure(&o,0); nya_train_adamw original=o;
    char error[160]; nya_train_session *s=nya_train_session_create("cuda",1024*1024,&o,gpu,2,error,sizeof(error)); CHECK(s);
    FILE *checkpoint=tmpfile(),*continuous=tmpfile(); CHECK(checkpoint && continuous);
    double initial=0,final=0;
    for (unsigned step=0;step<40;++step) {
        nya_train_zero_grad(cpu[0]); nya_train_zero_grad(cpu[1]); CHECK(!nya_train_session_zero_grad(s));
        double combined=0;
        for (unsigned micro=0;micro<2;++micro) {
            float expected,actual; CHECK(!cpu_group(cpu,micro,&expected) && !gpu_group(s,micro,&actual));
            CHECK(fabs((double)expected-actual)<=1e-6*(1+fabs(expected))); combined+=actual;
        }
        if (step==0) initial=combined;
        final=combined;
        configure(&o,step); CHECK(!nya_train_adamw_step(&o,cpu,2,error,sizeof(error)));
        nya_train_device_stats before,after; nya_train_device_get_stats(nya_train_session_device(s),&before);
        CHECK(!nya_train_session_step(s,settings(step)));
        nya_train_device_get_stats(nya_train_session_device(s),&after);
        CHECK(after.kernel_launches==before.kernel_launches+5 && after.used_bytes==before.used_bytes &&
            after.uploads==before.uploads && after.downloads==before.downloads && after.synchronizations==before.synchronizations);
        nya_train_session_metrics metrics; CHECK(!nya_train_session_observe(s,&metrics) && metrics.step==o.step);
        FILE *a=tmpfile(),*b=tmpfile();
        CHECK(!cpu_snapshot(a,&o,cpu) && b && !nya_train_session_checkpoint_write(s,b) && !close_checkpoint(a,b)); fclose(a); fclose(b);
        if (step==9) CHECK(!nya_train_session_checkpoint_write(s,checkpoint));
    }
    CHECK(final<initial*0.7 && !nya_train_session_checkpoint_write(s,continuous));
    CHECK(!nya_train_session_detach(s,&original) && original.step==40); nya_train_session_free(s);
    FILE *detached=tmpfile(); CHECK(!cpu_snapshot(detached,&original,gpu) && same_file(continuous,detached)); fclose(detached);
    /* A fresh session consumes the existing CPU checkpoint parser. After
       continuation every state byte and optimizer setting must match exactly. */
    rewind(checkpoint); CHECK(!nya_train_checkpoint_read(checkpoint,&original,gpu,2) && original.step==10);
    s=nya_train_session_create("cuda",1024*1024,&original,gpu,2,error,sizeof(error)); CHECK(s);
    for (unsigned step=10;step<40;++step) {
        CHECK(!nya_train_session_zero_grad(s)); float loss;
        CHECK(!gpu_group(s,0,&loss) && !gpu_group(s,1,&loss) && !nya_train_session_step(s,settings(step)));
        nya_train_session_metrics m; CHECK(!nya_train_session_observe(s,&m) && m.step==step+1);
    }
    FILE *resumed=tmpfile(); CHECK(resumed && !nya_train_session_checkpoint_write(s,resumed) && same_file(continuous,resumed));
    CHECK(!nya_train_session_detach(s,&original)); nya_train_session_free(s);
    fclose(resumed); fclose(continuous); fclose(checkpoint);
    for (size_t i=0;i<2;++i) { nya_train_parameter_free(cpu[i]); nya_train_parameter_free(gpu[i]); }
    printf("40-step accumulated CPU/GPU trajectory, loss %.9g -> %.9g, exact checkpoint continuation passed\n",initial,final); return 0;
}

static int export_guard(void)
{
    nya_train_decoder_config c; nya_train_decoder_defaults(&c);
    c.embedding_length=8; c.feed_forward_length=16; c.block_count=1; c.head_count=2; c.kv_head_count=1;
    char error[160]; nya_train_decoder *decoder=nya_train_decoder_create(&c,error,sizeof(error)); CHECK(decoder);
    size_t count; nya_train_parameter *const *p=nya_train_decoder_parameters(decoder,&count);
    nya_train_adamw o; nya_train_adamw_defaults(&o);
    nya_train_session *s=nya_train_session_create("cuda",2*1024*1024,&o,p,count,error,sizeof(error)); CHECK(s);
    FILE *f=tmpfile(); CHECK(f && nya_train_decoder_export(decoder,f,error,sizeof(error))<0 && ftell(f)==0 && strstr(error,"detach"));
    CHECK(!nya_train_session_detach(s,&o) && !nya_train_decoder_export(decoder,f,error,sizeof(error)));
    fclose(f); nya_train_session_free(s); nya_train_decoder_free(decoder); return 0;
}
static int failure(int creation)
{
    nya_train_parameter *p[2]; CHECK(!parameters(p)); nya_train_adamw o; nya_train_adamw_defaults(&o);
    FILE *before=tmpfile(),*after=tmpfile(); CHECK(!cpu_snapshot(before,&o,p)); char error[160];
    nya_train_session *s=nya_train_session_create("cuda",1024*1024,&o,p,2,error,sizeof(error));
    if (creation) CHECK(!s && error[0]);
    else {
        CHECK(s && nya_train_session_step(s,settings(0))<0);
        nya_train_session_metrics m={91,17,8},prior=m;
        CHECK(nya_train_session_observe(s,&m)<0 && !memcmp(&m,&prior,sizeof(m)));
        CHECK(nya_train_session_checkpoint_write(s,after)<0 && ftell(after)==0);
        CHECK(nya_train_session_detach(s,&o)<0 && o.step==0 && !nya_train_parameter_data(p[0]));
        nya_train_session_free(s);
    }
    CHECK(!cpu_snapshot(after,&o,p) && same_file(before,after)); fclose(before); fclose(after);
    nya_train_parameter_free(p[0]); nya_train_parameter_free(p[1]); return 0;
}
int main(int argc,char **argv)
{
    if (argc==1) return interfaces();
    nya_train_device *probe=nya_train_device_create("cuda",1024*1024);
    if (!probe) return 77;
    nya_train_device_free(probe);
    if (!strcmp(argv[1],"--create-failure")) return failure(1);
    if (!strcmp(argv[1],"--update-failure")) return failure(0);
    CHECK(!strcmp(argv[1],"--cuda"));
    CHECK(!interfaces() && !ownership() && !transaction_failures() && !failed_detach_preserves_host() && !trajectory() && !export_guard());
    puts("Resident ownership, CPU access guards, transactional detach and export checks passed"); return 0;
}
