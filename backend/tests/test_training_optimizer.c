#include "training_device.h"
#include "training.h"
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define CHECK(x) do { if (!(x)) { fprintf(stderr,"line %d: %s (%s)\n",__LINE__,#x,nya_train_device_error(d)); return 1; } } while (0)
static int close_value(double expected,float actual)
{
    if (isfinite(actual) && fabs(expected-actual)<=1e-6*(1+fabs(expected))) return 1;
    fprintf(stderr,"expected %.12g actual %.9g\n",expected,(double)actual); return 0;
}
static nya_train_adamw_config config_of(nya_train_adamw o)
{ return (nya_train_adamw_config){o.learning_rate,o.beta1,o.beta2,o.epsilon,o.weight_decay,o.max_grad_norm}; }
static int tensor(nya_train_device *d,nya_train_adamw_tensor *p,size_t count)
{
    *p=(nya_train_adamw_tensor){nya_train_device_alloc(d,count*4+4),nya_train_device_alloc(d,count*4+4),
        nya_train_device_alloc(d,count*4+4),nya_train_device_alloc(d,count*4+4),count};
    CHECK(p->values && p->gradient && p->moment && p->variance);
    float sentinel=12345; nya_train_buffer ids[]={p->values,p->gradient,p->moment,p->variance};
    for (size_t j=0;j<4;++j) CHECK(!nya_train_device_write(d,ids[j],count*4,&sentinel,4));
    return 0;
}
static int compare_checkpoint(nya_train_device *d,nya_train_adamw *optimizer,
    nya_train_parameter **parameters,nya_train_adamw_tensor *tensors,size_t count)
{
    FILE *file=tmpfile(); CHECK(file && !nya_train_checkpoint_write(file,optimizer,parameters,count) && !fseek(file,48,SEEK_SET));
    for (size_t p=0;p<count;++p) {
        size_t n=tensors[p].count; float *actual=malloc((n+1)*4); CHECK(actual && !fseek(file,16,SEEK_CUR));
        nya_train_buffer ids[]={tensors[p].values,tensors[p].gradient,tensors[p].moment,tensors[p].variance};
        for (size_t array=0;array<4;++array) {
            CHECK(!nya_train_device_read(d,ids[array],0,actual,(n+1)*4) && actual[n]==12345);
            for (size_t i=0;i<n;++i) {
                unsigned char bytes[4]; CHECK(fread(bytes,1,4,file)==4);
                uint32_t bits=(uint32_t)bytes[0]|(uint32_t)bytes[1]<<8|(uint32_t)bytes[2]<<16|(uint32_t)bytes[3]<<24;
                float expected; memcpy(&expected,&bits,4); CHECK(close_value(expected,actual[i]));
                if (array==1) CHECK(!memcmp(&expected,&actual[i],4));
            }
        }
        free(actual);
    }
    CHECK(!fclose(file)); return 0;
}
static int trajectory(nya_train_device *d,int variant)
{
    const size_t counts[]={1,3,255,256,257,2047,2048,2049,8193};
    nya_train_scope scope=nya_train_device_scratch_begin(d); CHECK(scope);
    nya_train_adamw_tensor tensors[9]; nya_train_parameter *parameters[9];
    for (size_t p=0;p<9;++p) {
        size_t n=counts[p]; float *initial=malloc(n*4); CHECK(initial);
        for (size_t i=0;i<n;++i) initial[i]=(float)((int)((i+p*19)%61)-30)/16;
        parameters[p]=nya_train_parameter_create(1,n,initial); CHECK(parameters[p] && !tensor(d,&tensors[p],n));
        CHECK(!nya_train_device_write(d,tensors[p].values,0,initial,n*4)); free(initial);
    }
    nya_train_buffer step=nya_train_device_alloc(d,8),status=nya_train_device_alloc(d,4),norm=nya_train_device_alloc(d,8);
    nya_train_optimizer_plan plan=nya_train_device_adamw_plan(d,tensors,9); CHECK(step && status && norm && plan);
    nya_train_adamw optimizer; nya_train_adamw_defaults(&optimizer);
    optimizer.learning_rate=0.017f; optimizer.weight_decay=variant==2?0:0.031f;
    optimizer.max_grad_norm=variant==0?0:variant==1?0.1f:100000;
    if (variant==2) { optimizer.beta1=0; optimizer.beta2=0; }
    if (variant>=3) { optimizer.beta1=nextafterf(1,0); optimizer.beta2=nextafterf(1,0); }
    if (variant==4) optimizer.step=(UINT64_C(1)<<53)+7;
    CHECK(!nya_train_device_write(d,step,0,&optimizer.step,8));
    for (unsigned iteration=0;iteration<20;++iteration) {
        nya_train_graph *graph=nya_train_graph_create(4*1024*1024); CHECK(graph);
        nya_train_tensor *objective=NULL; double square=0;
        for (size_t p=0;p<9;++p) {
            size_t n=counts[p]; float *gradient=malloc(n*4); CHECK(gradient);
            for (size_t i=0;i<n;++i) {
                gradient[i]=(float)((int)((i+p*5+iteration*3)%71)-35)/32;
                square+=(double)gradient[i]*gradient[i];
            }
            nya_train_zero_grad(parameters[p]);
            nya_train_tensor *term=nya_train_linear(nya_train_leaf(graph,parameters[p]),nya_train_input(graph,1,n,gradient)); CHECK(term);
            objective=objective?nya_train_add(objective,term):term; CHECK(objective);
            CHECK(!nya_train_device_write(d,tensors[p].gradient,0,gradient,n*4)); free(gradient);
        }
        CHECK(!nya_train_backward(objective)); nya_train_graph_free(graph);
        optimizer.learning_rate*=0.97f;
        nya_train_adamw_config config=config_of(optimizer);
        char error[256]; CHECK(!nya_train_adamw_step(&optimizer,parameters,9,error,sizeof(error)));
        nya_train_device_stats before,after; nya_train_device_get_stats(d,&before);
        CHECK(!nya_train_device_adamw(d,plan,config,step,status,norm,17));
        nya_train_device_get_stats(d,&after);
        CHECK(after.kernel_launches==before.kernel_launches+5 && after.uploads==before.uploads &&
            after.downloads==before.downloads && after.synchronizations==before.synchronizations && after.used_bytes==before.used_bytes);
        uint64_t actual_step=0; uint32_t tag=99; double actual_norm=0;
        CHECK(!nya_train_device_read(d,step,0,&actual_step,8) && actual_step==optimizer.step);
        CHECK(!nya_train_device_read(d,status,0,&tag,4) && !tag);
        CHECK(!nya_train_device_read(d,norm,0,&actual_norm,8) && fabs(actual_norm-sqrt(square))<=1e-12*(1+sqrt(square)));
        CHECK(!compare_checkpoint(d,&optimizer,parameters,tensors,9));
    }
    for (size_t p=0;p<9;++p) nya_train_parameter_free(parameters[p]);
    CHECK(!nya_train_device_scratch_end(d,scope)); return 0;
}
static int transactions(nya_train_device *d)
{
    nya_train_scope scope=nya_train_device_scratch_begin(d); CHECK(scope);
    nya_train_adamw_tensor tensors[2]; CHECK(!tensor(d,&tensors[0],3) && !tensor(d,&tensors[1],4097));
    nya_train_buffer step=nya_train_device_alloc(d,8),status=nya_train_device_alloc(d,4),norm=nya_train_device_alloc(d,8);
    nya_train_optimizer_plan plan=nya_train_device_adamw_plan(d,tensors,2); CHECK(step && status && norm && plan);
    nya_train_adamw optimizer; nya_train_adamw_defaults(&optimizer); optimizer.max_grad_norm=0;
    float *snapshot=malloc(4097*4),*actual=malloc(4097*4); CHECK(snapshot && actual);
    for (int failure=0;failure<10;++failure) {
        uint64_t expected_step=failure==8?UINT64_MAX:7; double expected_norm=123;
        uint32_t oldtag=failure==9?91u:0u;
        CHECK(!nya_train_device_write(d,step,0,&expected_step,8) && !nya_train_device_write(d,status,0,&oldtag,4) &&
            !nya_train_device_write(d,norm,0,&expected_norm,8));
        for (size_t p=0;p<2;++p) {
            nya_train_buffer ids[]={tensors[p].values,tensors[p].gradient,tensors[p].moment,tensors[p].variance};
            for (size_t array=0;array<4;++array) {
                for (size_t i=0;i<tensors[p].count;++i) snapshot[i]=(float)(array+1)/8;
                if (p==1) {
                    if (failure<=2 && array==1) snapshot[4096]=failure==0?NAN:failure==1?INFINITY:-INFINITY;
                    if (failure==3 && array==0) snapshot[4096]=NAN;
                    if (failure==4 && array==2) snapshot[4096]=INFINITY;
                    if (failure==5 && array==3) snapshot[4096]=-FLT_MAX;
                    if (failure==6 && array==1) snapshot[4096]=FLT_MAX;
                    if (failure==7 && array==0) snapshot[4096]=FLT_MAX;
                }
                CHECK(!nya_train_device_write(d,ids[array],0,snapshot,tensors[p].count*4));
            }
        }
        nya_train_adamw_config config=config_of(optimizer);
        if (failure==7) { config.learning_rate=FLT_MAX; config.weight_decay=FLT_MAX; }
        CHECK(!nya_train_device_adamw(d,plan,config,step,status,norm,17));
        /* A later queued attempt must preserve the first failure and all state. */
        CHECK(!nya_train_device_adamw(d,plan,config,step,status,norm,23));
        uint64_t actual_step=0; uint32_t tag=0; double actual_norm=0;
        CHECK(!nya_train_device_read(d,step,0,&actual_step,8) && actual_step==expected_step);
        CHECK(!nya_train_device_read(d,status,0,&tag,4) && tag==(oldtag?oldtag:17u));
        CHECK(!nya_train_device_read(d,norm,0,&actual_norm,8) && actual_norm==expected_norm);
        for (size_t p=0;p<2;++p) {
            nya_train_buffer ids[]={tensors[p].values,tensors[p].gradient,tensors[p].moment,tensors[p].variance};
            for (size_t array=0;array<4;++array) {
                CHECK(!nya_train_device_read(d,ids[array],0,actual,tensors[p].count*4));
                for (size_t i=0;i<tensors[p].count;++i) {
                    float expected=(float)(array+1)/8;
                    if (p==1 && i==4096) {
                        if (failure<=2 && array==1) expected=failure==0?NAN:failure==1?INFINITY:-INFINITY;
                        if (failure==3 && array==0) expected=NAN;
                        if (failure==4 && array==2) expected=INFINITY;
                        if (failure==5 && array==3) expected=-FLT_MAX;
                        if (failure==6 && array==1) expected=FLT_MAX;
                        if (failure==7 && array==0) expected=FLT_MAX;
                    }
                    CHECK(!memcmp(&expected,&actual[i],4));
                }
            }
        }
        nya_train_device_stats stats; nya_train_device_get_stats(d,&stats); CHECK(!stats.failed);
    }
    /* Reset corrected state and queue several successful steps without readback. */
    CHECK(!nya_train_device_zero(d,step) && !nya_train_device_zero(d,status));
    for (size_t p=0;p<2;++p) CHECK(!nya_train_device_zero(d,tensors[p].values) && !nya_train_device_zero(d,tensors[p].moment) && !nya_train_device_zero(d,tensors[p].variance));
    for (size_t k=0;k<5;++k) CHECK(!nya_train_device_adamw(d,plan,config_of(optimizer),step,status,norm,31));
    uint64_t actual_step=0; uint32_t tag=99;
    CHECK(!nya_train_device_read(d,step,0,&actual_step,8) && actual_step==5 && !nya_train_device_read(d,status,0,&tag,4) && !tag);
    free(snapshot); free(actual); CHECK(!nya_train_device_scratch_end(d,scope)); return 0;
}
static int descriptors(nya_train_device *d)
{
    nya_train_scope scope=nya_train_device_scratch_begin(d); CHECK(scope);
    nya_train_adamw_tensor p; CHECK(!tensor(d,&p,3));
    nya_train_optimizer_plan stale=nya_train_device_adamw_plan(d,&p,1); CHECK(stale && !nya_train_device_scratch_end(d,scope));
    scope=nya_train_device_scratch_begin(d); CHECK(scope);
    CHECK(!tensor(d,&p,3));
    nya_train_buffer step=nya_train_device_alloc(d,8),status=nya_train_device_alloc(d,4),norm=nya_train_device_alloc(d,8),tiny=nya_train_device_alloc(d,1);
    nya_train_optimizer_plan plan=nya_train_device_adamw_plan(d,&p,1); CHECK(step && status && norm && tiny && plan);
    uint32_t id=0; nya_train_indices map=nya_train_device_indices(d,&id,1,1); CHECK(map);
    nya_train_device *other=nya_train_device_create("cuda",4096); CHECK(other);
    nya_train_adamw_tensor foreign; CHECK(!tensor(other,&foreign,3));
    nya_train_optimizer_plan foreign_plan=nya_train_device_adamw_plan(other,&foreign,1); CHECK(foreign_plan);
    nya_train_adamw o; nya_train_adamw_defaults(&o); nya_train_adamw_config config=config_of(o);
    nya_train_device_stats before,after; nya_train_device_get_stats(d,&before);
    const nya_train_buffer invalid[]={0,stale,foreign.values,foreign_plan,tiny,map,plan};
    for (size_t i=0;i<7;++i) {
        nya_train_adamw_tensor bad=p; bad.values=invalid[i]; CHECK(!nya_train_device_adamw_plan(d,&bad,1));
        bad=p; bad.gradient=invalid[i]; CHECK(!nya_train_device_adamw_plan(d,&bad,1));
        bad=p; bad.moment=invalid[i]; CHECK(!nya_train_device_adamw_plan(d,&bad,1));
        bad=p; bad.variance=invalid[i]; CHECK(!nya_train_device_adamw_plan(d,&bad,1));
        CHECK(nya_train_device_adamw(d,plan,config,invalid[i],status,norm,1));
        CHECK(nya_train_device_adamw(d,plan,config,step,invalid[i],norm,1));
        CHECK(nya_train_device_adamw(d,plan,config,step,status,invalid[i],1));
        if (invalid[i]!=plan) CHECK(nya_train_device_adamw(d,invalid[i],config,step,status,norm,1));
    }
    CHECK(!nya_train_device_adamw_plan(d,NULL,1) && !nya_train_device_adamw_plan(d,&p,0) && !nya_train_device_adamw_plan(d,&p,SIZE_MAX));
    nya_train_adamw_tensor pair[2]={p,p}; CHECK(!nya_train_device_adamw_plan(d,pair,2));
    nya_train_adamw_tensor bad=p; bad.count=0; CHECK(!nya_train_device_adamw_plan(d,&bad,1));
    bad.count=5; CHECK(!nya_train_device_adamw_plan(d,&bad,1)); bad.count=SIZE_MAX; CHECK(!nya_train_device_adamw_plan(d,&bad,1));
    const nya_train_buffer ids[]={p.values,p.gradient,p.moment,p.variance};
    for (size_t i=0;i<4;++i) {
        CHECK(nya_train_device_adamw(d,plan,config,ids[i],status,norm,1));
        CHECK(nya_train_device_adamw(d,plan,config,step,ids[i],norm,1));
        CHECK(nya_train_device_adamw(d,plan,config,step,status,ids[i],1));
        for (size_t j=0;j<4;++j) if (i!=j) {
            bad=p;
            if (j==0) bad.values=ids[i]; else if (j==1) bad.gradient=ids[i]; else if (j==2) bad.moment=ids[i]; else bad.variance=ids[i];
            CHECK(!nya_train_device_adamw_plan(d,&bad,1));
        }
    }
    CHECK(nya_train_device_adamw(d,plan,config,step,status,norm,0));
    CHECK(nya_train_device_adamw(d,plan,config,step,step,norm,1));
    CHECK(nya_train_device_adamw(d,plan,config,step,status,step,1));
    CHECK(nya_train_device_adamw(d,plan,config,step,norm,norm,1));
    const float invalid_settings[]={NAN,INFINITY,-INFINITY,-1};
    for (size_t i=0;i<4;++i) for (size_t field=0;field<6;++field) {
        nya_train_adamw_config c=config;
        if (field==0) c.learning_rate=invalid_settings[i]; else if (field==1) c.beta1=invalid_settings[i];
        else if (field==2) c.beta2=invalid_settings[i]; else if (field==3) c.epsilon=invalid_settings[i];
        else if (field==4) c.weight_decay=invalid_settings[i]; else c.max_grad_norm=invalid_settings[i];
        CHECK(nya_train_device_adamw(d,plan,c,step,status,norm,1));
    }
    nya_train_adamw_config c=config; c.learning_rate=0; CHECK(nya_train_device_adamw(d,plan,c,step,status,norm,1));
    c=config; c.epsilon=0; CHECK(nya_train_device_adamw(d,plan,c,step,status,norm,1));
    c=config; c.beta1=1; CHECK(nya_train_device_adamw(d,plan,c,step,status,norm,1));
    c=config; c.beta2=1; CHECK(nya_train_device_adamw(d,plan,c,step,status,norm,1));
    uint64_t sentinel=42;
    CHECK(nya_train_device_read(d,plan,0,&sentinel,8) && sentinel==42);
    CHECK(nya_train_device_write(d,plan,0,&sentinel,8) && nya_train_device_zero(d,plan));
    CHECK(nya_train_device_check_finite(d,status,plan,1,2));
    nya_train_device_get_stats(d,&after);
    CHECK(!after.failed && after.kernel_launches==before.kernel_launches && after.uploads==before.uploads &&
        after.downloads==before.downloads && after.synchronizations==before.synchronizations && after.used_bytes==before.used_bytes);
    CHECK(!nya_train_device_adamw(d,plan,config,step,status,norm,1) && !nya_train_device_finish(d));
    nya_train_device_free(other); CHECK(!nya_train_device_scratch_end(d,scope));
    /* Descriptor copies outlive caller mutations, and retired plans release
       their host metadata without fencing queued device consumers. */
    CHECK(!tensor(d,&p,3)); step=nya_train_device_alloc(d,8); status=nya_train_device_alloc(d,4); norm=nya_train_device_alloc(d,8);
    CHECK(step && status && norm); nya_train_device_get_stats(d,&before);
    for (size_t repeat=0;repeat<100;++repeat) {
        scope=nya_train_device_scratch_begin(d); CHECK(scope);
        nya_train_adamw_tensor copy=p; plan=nya_train_device_adamw_plan(d,&copy,1); CHECK(plan);
        memset(&copy,0,sizeof(copy));
        CHECK(!nya_train_device_adamw(d,plan,config,step,status,norm,1));
        CHECK(!nya_train_device_scratch_end(d,scope));
        CHECK(nya_train_device_adamw(d,plan,config,step,status,norm,1));
    }
    nya_train_device_get_stats(d,&after);
    CHECK(after.used_bytes==before.used_bytes && after.buffers==before.buffers && after.scratch_resets==before.scratch_resets+100);
    CHECK(!nya_train_device_read(d,step,0,&sentinel,8) && sentinel==100);
    return 0;
}
static int overflow_boundary(nya_train_device *d)
{
    nya_train_scope scope=nya_train_device_scratch_begin(d); CHECK(scope);
    nya_train_adamw_tensor p; CHECK(!tensor(d,&p,1));
    float value=FLT_MAX,grad=-1;
    CHECK(!nya_train_device_write(d,p.values,0,&value,4) && !nya_train_device_write(d,p.gradient,0,&grad,4));
    nya_train_optimizer_plan plan=nya_train_device_adamw_plan(d,&p,1);
    nya_train_buffer step=nya_train_device_alloc(d,8),status=nya_train_device_alloc(d,4),norm=nya_train_device_alloc(d,8); CHECK(plan && step && status && norm);
    nya_train_adamw_config config={1e30f,0,0,1e-8f,0,0};
    double proposed=(double)FLT_MAX+(double)config.learning_rate/(1+(double)config.epsilon);
    CHECK(proposed>FLT_MAX && (float)proposed==FLT_MAX);
    CHECK(!nya_train_device_adamw(d,plan,config,step,status,norm,37));
    uint32_t tag=0; uint64_t count=99;
    CHECK(!nya_train_device_read(d,status,0,&tag,4) && tag==37 && !nya_train_device_read(d,step,0,&count,8) && !count);
    float actual; CHECK(!nya_train_device_read(d,p.values,0,&actual,4) && actual==FLT_MAX);
    CHECK(!nya_train_device_read(d,p.moment,0,&actual,4) && actual==0 && !nya_train_device_read(d,p.variance,0,&actual,4) && actual==0);
    /* The last representable step succeeds; the following queued step rejects. */
    count=UINT64_MAX-1; value=0; config.learning_rate=0.01f;
    CHECK(!nya_train_device_write(d,p.values,0,&value,4) && !nya_train_device_write(d,step,0,&count,8) && !nya_train_device_zero(d,status));
    CHECK(!nya_train_device_adamw(d,plan,config,step,status,norm,41));
    CHECK(!nya_train_device_read(d,step,0,&count,8) && count==UINT64_MAX);
    CHECK(!nya_train_device_adamw(d,plan,config,step,status,norm,43));
    CHECK(!nya_train_device_read(d,status,0,&tag,4) && tag==43);
    CHECK(!nya_train_device_read(d,step,0,&count,8) && count==UINT64_MAX);
    CHECK(!nya_train_device_scratch_end(d,scope)); return 0;
}
static int classifier(nya_train_device *d)
{
    const float input[12]={-1,0,1,-1,1,0,1,0,1,1,1,0},initial[6]={0.25f,-0.125f,0.0625f,-0.125f,0.25f,-0.0625f};
    const uint32_t labels[4]={0,0,1,1}; float seed=1,first=0,last=0;
    nya_train_scope outer=nya_train_device_scratch_begin(d); CHECK(outer);
    nya_train_adamw_tensor p; CHECK(!tensor(d,&p,6));
    nya_train_optimizer_plan plan=nya_train_device_adamw_plan(d,&p,1);
    nya_train_buffer step=nya_train_device_alloc(d,8),status=nya_train_device_alloc(d,4),norm=nya_train_device_alloc(d,8);
    nya_train_buffer x=nya_train_device_alloc(d,48),bl=nya_train_device_alloc(d,16),dy=nya_train_device_alloc(d,4),zero=nya_train_device_alloc(d,24);
    nya_train_buffer logits=nya_train_device_alloc(d,32),dl=nya_train_device_alloc(d,32),state=nya_train_device_alloc(d,136),loss=nya_train_device_alloc(d,4);
    nya_train_buffer zero_logits=nya_train_device_alloc(d,32);
    CHECK(plan && step && status && norm && x && bl && dy && zero && logits && dl && state && loss && zero_logits);
    CHECK(!nya_train_device_write(d,p.values,0,initial,24) && !nya_train_device_write(d,x,0,input,48) &&
        !nya_train_device_write(d,bl,0,labels,16) && !nya_train_device_write(d,dy,0,&seed,4));
    nya_train_parameter *parameter=nya_train_parameter_create(2,3,initial); CHECK(parameter);
    nya_train_adamw optimizer; nya_train_adamw_defaults(&optimizer); optimizer.learning_rate=0.05f; optimizer.weight_decay=0;
    for (size_t i=0;i<50;++i) {
        nya_train_zero_grad(parameter);
        nya_train_graph *g=nya_train_graph_create(65536); CHECK(g);
        nya_train_tensor *y=nya_train_linear(nya_train_input(g,4,3,input),nya_train_leaf(g,parameter));
        nya_train_tensor *objective=nya_train_cross_entropy(y,labels,NULL,4); CHECK(objective);
        float expected=*nya_train_data(objective); CHECK(!nya_train_backward(objective));
        nya_train_graph_free(g);
        char error[256]; CHECK(!nya_train_adamw_step(&optimizer,&parameter,1,error,sizeof(error)));
        nya_train_device_stats before,after; nya_train_device_get_stats(d,&before);
        CHECK(!nya_train_device_unary(d,p.gradient,zero,6,NYA_TRAIN_UNARY_SCALE,0));
        CHECK(!nya_train_device_unary(d,dl,zero_logits,8,NYA_TRAIN_UNARY_SCALE,0));
        CHECK(!nya_train_device_linear(d,logits,p.values,0,2,3,x,4));
        CHECK(!nya_train_device_loss(d,loss,state,(nya_train_view){logits,4,2},bl,0,NYA_TRAIN_LOSS_CE));
        CHECK(!nya_train_device_check_finite(d,status,loss,1,11));
        CHECK(!nya_train_device_loss_backward(d,dl,state,(nya_train_view){logits,4,2},bl,0,dy));
        CHECK(!nya_train_device_linear_dw(d,p.gradient,x,dl,2,3,4));
        CHECK(!nya_train_device_check_finite(d,status,p.gradient,6,13));
        CHECK(!nya_train_device_adamw(d,plan,config_of(optimizer),step,status,norm,17));
        nya_train_device_get_stats(d,&after);
        CHECK(after.kernel_launches==before.kernel_launches+14 && after.used_bytes==before.used_bytes &&
            after.uploads==before.uploads && after.downloads==before.downloads && after.synchronizations==before.synchronizations);
        CHECK(!nya_train_device_read(d,loss,0,&last,4) && close_value(expected,last));
        if (!i) first=last;
        uint64_t actual_step=0; uint32_t tag=99;
        CHECK(!nya_train_device_read(d,step,0,&actual_step,8) && actual_step==i+1 && !nya_train_device_read(d,status,0,&tag,4) && !tag);
        CHECK(!compare_checkpoint(d,&optimizer,&parameter,&p,1));
    }
    CHECK(last<first*0.2f);
    nya_train_parameter_free(parameter); CHECK(!nya_train_device_scratch_end(d,outer)); return 0;
}
static int resume(nya_train_device *d)
{
    nya_train_scope scope=nya_train_device_scratch_begin(d); CHECK(scope);
    nya_train_adamw_tensor p; CHECK(!tensor(d,&p,2049));
    float *values=malloc(2049*4),*actual=malloc(2049*4); CHECK(values && actual);
    for (size_t i=0;i<2049;++i) values[i]=(float)((int)(i%71)-35)/64;
    CHECK(!nya_train_device_write(d,p.values,0,values,2049*4) && !nya_train_device_write(d,p.gradient,0,values,2049*4));
    nya_train_optimizer_plan plan=nya_train_device_adamw_plan(d,&p,1);
    nya_train_buffer step=nya_train_device_alloc(d,8),status=nya_train_device_alloc(d,4),norm=nya_train_device_alloc(d,8); CHECK(plan && step && status && norm);
    nya_train_adamw o; nya_train_adamw_defaults(&o);
    for (size_t i=0;i<10;++i) CHECK(!nya_train_device_adamw(d,plan,config_of(o),step,status,norm,17));
    uint64_t saved_step=0; CHECK(!nya_train_device_read(d,step,0,&saved_step,8) && saved_step==10);
    nya_train_device *restored=nya_train_device_create("cuda",1024*1024); CHECK(restored);
    nya_train_adamw_tensor q; CHECK(!tensor(restored,&q,2049));
    const nya_train_buffer sources[]={p.values,p.gradient,p.moment,p.variance},destinations[]={q.values,q.gradient,q.moment,q.variance};
    for (size_t j=0;j<4;++j) CHECK(!nya_train_device_read(d,sources[j],0,values,2049*4) && !nya_train_device_write(restored,destinations[j],0,values,2049*4));
    nya_train_optimizer_plan rp=nya_train_device_adamw_plan(restored,&q,1);
    nya_train_buffer rs=nya_train_device_alloc(restored,8),rf=nya_train_device_alloc(restored,4),rn=nya_train_device_alloc(restored,8);
    CHECK(rp && rs && rf && rn && !nya_train_device_write(restored,rs,0,&saved_step,8));
    for (size_t i=0;i<10;++i) {
        o.learning_rate*=0.9f;
        CHECK(!nya_train_device_adamw(d,plan,config_of(o),step,status,norm,17));
        CHECK(!nya_train_device_adamw(restored,rp,config_of(o),rs,rf,rn,17));
    }
    for (size_t j=0;j<4;++j) CHECK(!nya_train_device_read(d,sources[j],0,values,2049*4) &&
        !nya_train_device_read(restored,destinations[j],0,actual,2049*4) && !memcmp(values,actual,2049*4));
    uint64_t a=0,b=0; uint32_t tag=99; double na=0,nb=0;
    CHECK(!nya_train_device_read(d,step,0,&a,8) && !nya_train_device_read(restored,rs,0,&b,8) && a==20 && b==20);
    CHECK(!nya_train_device_read(d,norm,0,&na,8) && !nya_train_device_read(restored,rn,0,&nb,8) && !memcmp(&na,&nb,8));
    CHECK(!nya_train_device_read(d,status,0,&tag,4) && !tag && !nya_train_device_read(restored,rf,0,&tag,4) && !tag);
    nya_train_device_free(restored); free(values); free(actual);
    CHECK(!nya_train_device_scratch_end(d,scope)); return 0;
}
static int budget(nya_train_device *d)
{
    nya_train_device *small=nya_train_device_create("cuda",2048); CHECK(small);
    nya_train_adamw_tensor p; CHECK(!tensor(small,&p,3));
    nya_train_buffer step=nya_train_device_alloc(small,8),status=nya_train_device_alloc(small,4),norm=nya_train_device_alloc(small,8);
    nya_train_optimizer_plan plan=nya_train_device_adamw_plan(small,&p,1); CHECK(step && status && norm && plan);
    nya_train_device_stats before,after; nya_train_device_get_stats(small,&before);
    CHECK(!nya_train_device_adamw_plan(small,&p,1)); nya_train_device_get_stats(small,&after);
    CHECK(!after.failed && after.used_bytes==before.used_bytes && after.buffers==before.buffers &&
        after.kernel_launches==before.kernel_launches && after.uploads==before.uploads && after.synchronizations==before.synchronizations);
    nya_train_adamw o; nya_train_adamw_defaults(&o);
    CHECK(!nya_train_device_adamw(small,plan,config_of(o),step,status,norm,17));
    uint64_t actual_step=0; double actual_norm=99;
    CHECK(!nya_train_device_read(small,step,0,&actual_step,8) && actual_step==1);
    CHECK(!nya_train_device_read(small,norm,0,&actual_norm,8) && actual_norm==0);
    nya_train_device_free(small); return 0;
}
int main(int argc,char **argv)
{
    nya_train_device *d=NULL;
    nya_train_adamw_tensor t={0,0,0,0,1};
    CHECK(nya_train_adamw_plan_bytes(&t,1)==116 && !nya_train_adamw_plan_bytes(NULL,1));
    CHECK(!nya_train_adamw_plan_bytes(&t,0) && !nya_train_adamw_plan_bytes(&t,SIZE_MAX));
    t.count=SIZE_MAX; CHECK(!nya_train_adamw_plan_bytes(&t,1)); t.count=0; CHECK(!nya_train_adamw_plan_bytes(&t,1));
    CHECK(!nya_train_device_adamw_plan(NULL,NULL,0));
    CHECK(nya_train_device_adamw(NULL,0,(nya_train_adamw_config){0},0,0,0,0));
    if (argc==1) return 0;
    d=nya_train_device_create("cuda",32*1024*1024); if (!d) return 77;
    if (!strcmp(argv[1],"--failure")) {
        nya_train_adamw_tensor p; CHECK(!tensor(d,&p,3));
        nya_train_optimizer_plan plan=nya_train_device_adamw_plan(d,&p,1);
        nya_train_buffer step=nya_train_device_alloc(d,8),status=nya_train_device_alloc(d,4),norm=nya_train_device_alloc(d,8); CHECK(plan && step && status && norm);
        nya_train_adamw o; nya_train_adamw_defaults(&o);
        CHECK(nya_train_device_adamw(d,plan,config_of(o),step,status,norm,17));
        nya_train_device_stats stats; nya_train_device_get_stats(d,&stats);
        CHECK(stats.failed && stats.kernel_launches==strtoull(getenv("NYA_TRAIN_DEVICE_FAIL_LAUNCH_AFTER"),NULL,10));
        float sentinel=123; CHECK(nya_train_device_read(d,p.values,0,&sentinel,4) && sentinel==123 && nya_train_device_finish(d));
        nya_train_device_free(d); return 0;
    }
    for (int variant=0;variant<5;++variant) CHECK(!trajectory(d,variant));
    CHECK(!transactions(d) && !descriptors(d) && !overflow_boundary(d) && !classifier(d) && !resume(d) && !budget(d));
    nya_train_device_free(d);
    puts("resident AdamW: CPU parameter/gradient/moment trajectories, clipping, norm, transaction rejection and recovery passed");
    return 0;
}
