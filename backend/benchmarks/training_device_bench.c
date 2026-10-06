/* Resident matrix microbenchmark, not a complete training throughput test. */
#include "training_device.h"
#include "llm_internal.h"
#include "training_internal.h"
#include "train_clock.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define REQUIRE(x) do { if (!(x)) { fprintf(stderr,"line %d: %s; %s\n",__LINE__,#x,nya_train_device_error(d)); return 1; } } while (0)
static int dispatch(nya_train_device *d, unsigned op, nya_train_buffer y, nya_train_buffer w, nya_train_buffer x,
    nya_train_buffer dy, nya_train_buffer dx, nya_train_buffer dw, unsigned type, size_t o, size_t i, size_t n,
    nya_train_buffer status)
{
    int result=op==0 ? nya_train_device_linear(d,y,w,type,o,i,x,n) : op==1 ?
        nya_train_device_linear_dx(d,dx,w,type,o,i,dy,n) : nya_train_device_linear_dw(d,dw,x,dy,o,i,n);
    if (!result && status) result=nya_train_device_check_finite(d,status,op==0?y:op==1?dx:dw,
        op==0?n*o:op==1?n*i:o*i,op+1);
    return result;
}
static int run(nya_train_device *d, const nya_llm_tensor *w, size_t n, int recycle, nya_train_buffer status)
{
    nya_train_scope scope=recycle ? nya_train_device_scratch_begin(d) : 0;
    REQUIRE(!recycle || scope);
    size_t i=(size_t)w->dimensions[0], o=(size_t)w->dimensions[1];
    float *x=malloc(n*i*4), *dy=malloc(n*o*4);
    REQUIRE(x && dy);
    for (size_t k=0;k<n*i;++k) x[k]=(float)((int)(k%71)-35)/64;
    for (size_t k=0;k<n*o;++k) dy[k]=(float)((int)(k%67)-33)/64;
    nya_train_buffer bw=nya_train_device_alloc(d,w->data_size), bx=nya_train_device_alloc(d,n*i*4);
    nya_train_buffer by=nya_train_device_alloc(d,n*o*4), bd=nya_train_device_alloc(d,n*o*4);
    nya_train_buffer dx=nya_train_device_alloc(d,n*i*4), dw=nya_train_device_alloc(d,o*i*4);
    REQUIRE(bw && bx && by && bd && dx && dw);
    REQUIRE(!nya_train_device_write(d,bw,0,w->data,w->data_size) && !nya_train_device_write(d,bx,0,x,n*i*4) &&
        !nya_train_device_write(d,bd,0,dy,n*o*4));
    for (unsigned op=0;op<3;++op) {
        REQUIRE(!dispatch(d,op,by,bw,bx,bd,dx,dw,w->type,o,i,n,status));
        /* Independently check 64 spread elements per real-model result. The
           CTest suite checks every element on tile tails and long reductions. */
        for (size_t sample=0;sample<64;++sample) {
            size_t size=op==0?n*o:op==1?n*i:o*i, index=sample*(size-1)/63;
            size_t row=index/(op==0?o:i), col=index%(op==0?o:i), terms=op==0?i:op==1?o:n;
            double sum=0, magnitude=0;
            for (size_t k=0;k<terms;++k) {
                double p=op==0?(double)nya_llm_tensor_value(w,col*i+k)*x[row*i+k]:op==1?
                    (double)nya_llm_tensor_value(w,k*i+col)*dy[row*o+k]:(double)dy[k*o+row]*x[k*i+col];
                sum+=p; magnitude+=fabs(p);
            }
            float actual;
            REQUIRE(!nya_train_device_read(d,op==0?by:op==1?dx:dw,index*4,&actual,4));
            REQUIRE(isfinite(actual) && fabs(actual-sum)<=1e-6*(1+magnitude));
        }
        double times[5];
        for (unsigned pass=0;pass<7;++pass) {
            REQUIRE(!nya_train_device_zero(d,dx) && !nya_train_device_zero(d,dw) && !nya_train_device_finish(d));
            nya_train_device_stats before,after; nya_train_device_get_stats(d,&before);
            double start=tr_seconds();
            for (unsigned repeat=0;repeat<20;++repeat) REQUIRE(!dispatch(d,op,by,bw,bx,bd,dx,dw,w->type,o,i,n,status));
            REQUIRE(!nya_train_device_finish(d));
            double ms=(tr_seconds()-start)*1000/20;
            nya_train_device_get_stats(d,&after);
            REQUIRE(after.uploads==before.uploads && after.downloads==before.downloads && after.synchronizations==before.synchronizations+1 && after.kernel_launches==before.kernel_launches+20u*(status?2u:1u));
            if (pass>=2) times[pass-2]=ms;
        }
        printf("{\"tensor\":\"%s\",\"type\":%u,\"outputs\":%zu,\"inputs\":%zu,\"tokens\":%zu,\"operation\":%u,\"ms\":[",w->name,w->type,o,i,n,op);
        for (size_t k=0;k<5;++k) printf("%s%.9f",k?",":"",times[k]);
        printf("]}\n"); fflush(stdout);
    }
    if (recycle) REQUIRE(!nya_train_device_scratch_end(d,scope));
    free(x); free(dy);
    return 0;
}
static int startup(const char *mode)
{
    nya_train_device *d=NULL;
    double start=tr_seconds(), created, executed;
    float w=3, x=2, y=0;
    nya_compute_context *c=NULL;
    if (!strcmp(mode,"inference")) {
        c=nya_compute_create();
        REQUIRE(c && !strcmp(nya_compute_name(c),"cuda"));
        created=tr_seconds();
        REQUIRE(!nya_compute_matvec(c,&w,1,1,&x,&y) && y==6);
    } else if (!strcmp(mode,"training")) {
        d=nya_train_device_create("cuda",1024*1024); REQUIRE(d);
        created=tr_seconds();
        nya_train_buffer bw=nya_train_device_alloc(d,4), bx=nya_train_device_alloc(d,4), by=nya_train_device_alloc(d,4);
        REQUIRE(bw && bx && by && !nya_train_device_write(d,bw,0,&w,4) && !nya_train_device_write(d,bx,0,&x,4));
        REQUIRE(!nya_train_device_linear(d,by,bw,0,1,1,bx,1) && !nya_train_device_read(d,by,0,&y,4) && y==6);
    } else return 1;
    executed=tr_seconds();
    printf("{\"mode\":\"%s\",\"create_ms\":%.6f,\"first_work_ms\":%.6f",mode,(created-start)*1000,(executed-created)*1000);
#ifdef _WIN32
    printf(",\"cublas_loaded\":%d,\"cutlass_loaded\":%d",GetModuleHandleA("cublas64_13.dll")!=NULL,GetModuleHandleA("fyodor-cutlass.dll")!=NULL);
#endif
    printf("}\n");
    nya_train_device_free(d); nya_compute_free(c); return 0;
}
/* Real compressed gate/up/down weights; frozen FFN input-gradient parity.
   This is a component probe, not a complete transformer training step. */
static int ffn(const nya_llm_context *model,size_t n,int normalized,int embedded)
{
    nya_train_device *d=NULL;
    const nya_llm_tensor *w[]={model->layers[0].feed_forward_gate,model->layers[0].feed_forward_up,model->layers[0].feed_forward_down};
    REQUIRE(w[0] && w[1] && w[2]);
    size_t i=(size_t)w[0]->dimensions[0], h=(size_t)w[0]->dimensions[1];
    REQUIRE(i && h && w[1]->dimensions[0]==i && w[1]->dimensions[1]==h && w[2]->dimensions[0]==h && w[2]->dimensions[1]==i);
    REQUIRE(n<=SIZE_MAX/4/i && n<=SIZE_MAX/4/h);
    size_t count=n*i, hidden=n*h, table_rows=0, gradient_count=count;
    uint32_t *ids=embedded?malloc(n*4):NULL;
    if (embedded) {
        REQUIRE(normalized && ids && model->token_embedding && model->token_embedding->dimensions[0]==i);
        table_rows=(size_t)model->token_embedding->dimensions[1];
        REQUIRE(table_rows>=4 && table_rows<=UINT32_MAX && table_rows<=SIZE_MAX/4/i);
        gradient_count=table_rows*i;
        for (size_t k=0;k<n;++k) ids[k]=(uint32_t)(k%5?table_rows-1-k%3:1);
    }
    float *x=malloc(count*4), *dy=malloc(count*4), *expected=malloc(count*4), *actual=malloc((gradient_count>count?gradient_count:count)*4);
    REQUIRE(x && dy && expected && actual);
    for (size_t k=0;k<count;++k) { x[k]=(float)((int)(k%71)-35)/64; dy[k]=(float)((int)(k%67)-33)/64; }
    float *norm_weight=normalized?malloc(i*4):NULL;
    if (normalized) {
        REQUIRE(norm_weight && model->layers[0].feed_forward_norm &&
            model->layers[0].feed_forward_norm->dimensions[0]==i);
        for (size_t k=0;k<i;++k) norm_weight[k]=nya_llm_tensor_value(model->layers[0].feed_forward_norm,k);
    }
    nya_train_parameter *pn=normalized?nya_train_parameter_create(1,i,norm_weight):NULL;
    REQUIRE(!normalized || pn);
    nya_train_parameter *px=nya_train_parameter_create(embedded?table_rows:n,i,embedded?NULL:x);
    nya_train_executor *executor=nya_train_executor_create(6); REQUIRE(px && executor);
    if (embedded) for (size_t k=0;k<gradient_count;++k)
        nya_train_parameter_data(px)[k]=nya_llm_tensor_value(model->token_embedding,k);
    for (unsigned pass=0;pass<2;++pass) {
        nya_train_graph *g=nya_train_graph_create_with_executor(256*1024*1024,executor); REQUIRE(g);
        nya_train_tensor *input=nya_train_leaf(g,px);
        if (embedded) input=nya_train_embedding(input,ids,n);
        if (normalized) input=nya_train_rms_norm(input,nya_train_leaf(g,pn),model->norm_epsilon);
        nya_train_tensor *a=nya_train_linear_mapped(input,w[0]), *b=nya_train_linear_mapped(input,w[1]);
        nya_train_tensor *out=nya_train_linear_mapped(nya_train_mul(nya_train_silu(a),b),w[2]); REQUIRE(out);
        memcpy(expected,nya_train_data(out),count*4);
        REQUIRE(!nya_train_backward(nya_train_linear(nya_train_reshape(out,1,count),nya_train_input(g,1,count,dy))));
        nya_train_graph_free(g);
    }
    nya_train_executor_free(executor);
    d=nya_train_device_create("cuda",(embedded?768u:64u)*1024*1024); REQUIRE(d);
    nya_train_buffer weights[3];
    for (size_t k=0;k<3;++k) {
        weights[k]=nya_train_device_alloc(d,w[k]->data_size); REQUIRE(weights[k]);
        REQUIRE(!nya_train_device_write(d,weights[k],0,w[k]->data,w[k]->data_size));
    }
    nya_train_buffer input=nya_train_device_alloc(d,count*4), seed=nya_train_device_alloc(d,count*4);
    nya_train_buffer output=nya_train_device_alloc(d,count*4), dx=nya_train_device_alloc(d,count*4), status=nya_train_device_alloc(d,4);
    REQUIRE(input && seed && output && dx && status);
    if (!embedded) REQUIRE(!nya_train_device_write(d,input,0,x,count*4));
    REQUIRE(!nya_train_device_write(d,seed,0,dy,count*4));
    nya_train_buffer embedding_table=0, embedding_grad=0;
    nya_train_indices indices=0;
    if (embedded) {
        embedding_table=nya_train_device_alloc(d,gradient_count*4); embedding_grad=nya_train_device_alloc(d,gradient_count*4);
        indices=nya_train_device_indices(d,ids,n,table_rows); REQUIRE(embedding_table && embedding_grad && indices);
        REQUIRE(!nya_train_device_write(d,embedding_table,0,nya_train_parameter_data(px),gradient_count*4));
    }
    nya_train_buffer norm_scale=0, norm_grad=0;
    if (normalized) {
        norm_scale=nya_train_device_alloc(d,i*4); norm_grad=nya_train_device_alloc(d,i*4);
        REQUIRE(norm_scale && norm_grad && !nya_train_device_write(d,norm_scale,0,norm_weight,i*4));
    }
    double times[5];
    for (unsigned repetition=0;repetition<7;++repetition) {
        if (embedded) REQUIRE(!nya_train_device_zero(d,embedding_grad));
        if (normalized) REQUIRE(!nya_train_device_zero(d,norm_grad));
        REQUIRE(!nya_train_device_zero(d,dx) && !nya_train_device_zero(d,status) && !nya_train_device_finish(d));
        nya_train_device_stats before,after; nya_train_device_get_stats(d,&before);
        double start=tr_seconds();
        for (unsigned pass=0;pass<2;++pass) {
            nya_train_scope scope=nya_train_device_scratch_begin(d); REQUIRE(scope);
            if (embedded) {
                REQUIRE(!nya_train_device_zero(d,dx));
                REQUIRE(!nya_train_device_embedding(d,input,embedding_table,0,i,indices));
            }
            nya_train_buffer a=nya_train_device_alloc(d,hidden*4), b=nya_train_device_alloc(d,hidden*4), s=nya_train_device_alloc(d,hidden*4), m=nya_train_device_alloc(d,hidden*4);
            nya_train_buffer dm=nya_train_device_alloc(d,hidden*4), ds=nya_train_device_alloc(d,hidden*4), db=nya_train_device_alloc(d,hidden*4), da=nya_train_device_alloc(d,hidden*4);
            REQUIRE(a && b && s && m && dm && ds && db && da);
            nya_train_buffer normed=input, norm_dx=dx, inverse=0;
            if (normalized) {
                normed=nya_train_device_alloc(d,count*4); norm_dx=nya_train_device_alloc(d,count*4); inverse=nya_train_device_alloc(d,n*8);
                REQUIRE(normed && norm_dx && inverse);
                REQUIRE(!nya_train_device_rms_norm(d,normed,inverse,(nya_train_view){input,n,i},norm_scale,model->norm_epsilon));
                REQUIRE(!nya_train_device_check_finite(d,status,normed,count,3));
            }
            REQUIRE(!nya_train_device_linear(d,a,weights[0],w[0]->type,h,i,normed,n));
            REQUIRE(!nya_train_device_linear(d,b,weights[1],w[1]->type,h,i,normed,n));
            REQUIRE(!nya_train_device_unary(d,s,a,hidden,NYA_TRAIN_UNARY_SILU,0));
            REQUIRE(!nya_train_device_binary(d,m,(nya_train_view){s,n,h},(nya_train_view){b,n,h},NYA_TRAIN_BINARY_MUL));
            REQUIRE(!nya_train_device_linear(d,output,weights[2],w[2]->type,i,h,m,n));
            REQUIRE(!nya_train_device_linear_dx(d,dm,weights[2],w[2]->type,i,h,seed,n));
            REQUIRE(!nya_train_device_binary_backward(d,ds,db,(nya_train_view){s,n,h},(nya_train_view){b,n,h},dm,NYA_TRAIN_BINARY_MUL));
            REQUIRE(!nya_train_device_unary_backward(d,da,a,ds,hidden,NYA_TRAIN_UNARY_SILU,0));
            REQUIRE(!nya_train_device_linear_dx(d,norm_dx,weights[1],w[1]->type,h,i,db,n));
            REQUIRE(!nya_train_device_linear_dx(d,norm_dx,weights[0],w[0]->type,h,i,da,n));
            if (normalized) {
                REQUIRE(!nya_train_device_rms_norm_backward(d,dx,norm_grad,inverse,(nya_train_view){input,n,i},norm_scale,norm_dx));
                REQUIRE(!nya_train_device_check_finite(d,status,norm_grad,i,4));
            }
            REQUIRE(!nya_train_device_check_finite(d,status,output,count,1) && !nya_train_device_check_finite(d,status,dx,count,2));
            if (embedded) {
                REQUIRE(!nya_train_device_embedding_backward(d,embedding_grad,dx,i,indices));
                REQUIRE(!nya_train_device_check_finite(d,status,embedding_grad,gradient_count,5));
            }
            REQUIRE(!nya_train_device_scratch_end(d,scope));
        }
        REQUIRE(!nya_train_device_finish(d));
        double ms=(tr_seconds()-start)*1000;
        nya_train_device_get_stats(d,&after);
        REQUIRE(after.kernel_launches==before.kernel_launches+(embedded?64u:normalized?56u:40u) && after.uploads==before.uploads &&
            after.downloads==before.downloads && after.synchronizations==before.synchronizations+1 && after.used_bytes==before.used_bytes);
        if (repetition>=2) times[repetition-2]=ms;
    }
    uint32_t failure=1; REQUIRE(!nya_train_device_read(d,status,0,&failure,4) && !failure);
    double maximum[3]={0,0,0};
    for (unsigned gradient=0;gradient<(normalized?3u:2u);++gradient) {
        const float *reference=gradient==2?nya_train_parameter_gradient(pn):gradient?nya_train_parameter_gradient(px):expected;
        size_t elements=gradient==2?i:gradient?gradient_count:count;
        REQUIRE(!nya_train_device_read(d,gradient==2?norm_grad:gradient?(embedded?embedding_grad:dx):output,0,actual,elements*4));
        for (size_t k=0;k<elements;++k) {
            double error=fabs((double)actual[k]-reference[k])/(1+fabs(reference[k]));
            if (!isfinite(actual[k]) || error>1e-5) fprintf(stderr,"component gradient=%u element=%zu expected=%.9g actual=%.9g scaled_error=%.9g\n",gradient,k,(double)reference[k],(double)actual[k],error);
            REQUIRE(isfinite(actual[k]) && error<=1e-5);
            if (error>maximum[gradient]) maximum[gradient]=error;
        }
    }
    nya_train_device_stats stats; nya_train_device_get_stats(d,&stats);
    printf("{\"component\":\"%s\",\"tokens\":%zu,\"inputs\":%zu,\"hidden\":%zu,\"table_rows\":%zu,\"microbatches\":2,\"max_scaled_output_error\":%.9g,\"max_scaled_%s_gradient_error\":%.9g,\"max_scaled_norm_gradient_error\":%.9g,\"tolerance\":1e-5,\"peak_bytes\":%zu,\"live_bytes\":%zu,\"uploads\":%llu,\"downloads\":%llu,\"timing_accepted\":false,\"ms\":[",
        embedded?"embedding-rms-ffn":normalized?"silu-ffn-with-trainable-rms":"frozen-silu-ffn",n,i,h,table_rows,maximum[0],embedded?"embedding":"input",maximum[1],maximum[2],stats.peak_bytes,stats.used_bytes,(unsigned long long)stats.uploads,(unsigned long long)stats.downloads);
    for (size_t k=0;k<5;++k) printf("%s%.9f",k?",":"",times[k]);
    printf("]}\n");
    nya_train_device_free(d); nya_train_parameter_free(px);
    nya_train_parameter_free(pn); free(norm_weight);
    free(ids);
    free(x); free(dy); free(expected); free(actual);
    return 0;
}
/* Frozen real Q/K projections, RoPE, and shared input-gradient accumulation.
   Both branches and two microbatches must match CPU graph reverse traversal. */
static int qk_rope(const nya_llm_context *model,size_t n)
{
    nya_train_device *d=NULL;
    const nya_llm_tensor *weights[2]={model->layers[0].query,model->layers[0].key};
    size_t heads[2]={model->head_count,model->key_value_head_count};
    REQUIRE(heads[0]);
    size_t width=model->embedding_length,dim=width/heads[0];
    REQUIRE(width && heads[0] && heads[1] && dim && !(dim%2) && width%heads[0]==0);
    REQUIRE(isfinite(model->rope_frequency_base) && model->rope_frequency_base>=1);
    REQUIRE(n<=SIZE_MAX/4/width);
    float *x=malloc(n*width*4),*freq=malloc(dim/2*4),*actual=malloc(n*width*4);
    float *seed[2],*expected[2]; size_t cols[2];
    REQUIRE(x && freq && actual);
    for (size_t j=0;j<n*width;++j) x[j]=(float)((int)(j%71)-35)/64;
    for (size_t j=0;j<dim/2;++j) freq[j]=powf(model->rope_frequency_base,-2.0f*(float)j/(float)dim);
    for (size_t b=0;b<2;++b) {
        cols[b]=heads[b]*dim;
        REQUIRE(cols[b]<=width && weights[b] && weights[b]->dimensions[0]==width && weights[b]->dimensions[1]==cols[b]);
        seed[b]=malloc(n*cols[b]*4); expected[b]=malloc(n*cols[b]*4); REQUIRE(seed[b] && expected[b]);
        for (size_t j=0;j<n*cols[b];++j) seed[b][j]=(float)((int)(j%(b?61:67))-31)/64;
    }
    nya_train_parameter *px=nya_train_parameter_create(n,width,x);
    nya_train_executor *executor=nya_train_executor_create(6); REQUIRE(px && executor);
    for (unsigned pass=0;pass<2;++pass) {
        nya_train_graph *g=nya_train_graph_create_with_executor(128*1024*1024,executor); REQUIRE(g);
        nya_train_tensor *input=nya_train_leaf(g,px),*loss[2]; REQUIRE(input);
        for (size_t b=0;b<2;++b) {
            nya_train_tensor *out=nya_train_rope(nya_train_linear_mapped(input,weights[b]),heads[b],dim,freq,0); REQUIRE(out);
            memcpy(expected[b],nya_train_data(out),n*cols[b]*4);
            loss[b]=nya_train_linear(nya_train_reshape(out,1,n*cols[b]),nya_train_input(g,1,n*cols[b],seed[b])); REQUIRE(loss[b]);
        }
        REQUIRE(!nya_train_backward(nya_train_add(loss[0],loss[1])));
        nya_train_graph_free(g);
    }
    nya_train_executor_free(executor);
    d=nya_train_device_create("cuda",64*1024*1024); REQUIRE(d);
    nya_train_buffer bx=nya_train_device_alloc(d,n*width*4),bf=nya_train_device_alloc(d,dim/2*4);
    nya_train_buffer dx=nya_train_device_alloc(d,n*width*4),status=nya_train_device_alloc(d,4);
    REQUIRE(bx && bf && dx && status && !nya_train_device_write(d,bx,0,x,n*width*4) && !nya_train_device_write(d,bf,0,freq,dim/2*4));
    nya_train_buffer bw[2],bs[2],by[2];
    for (size_t b=0;b<2;++b) {
        bw[b]=nya_train_device_alloc(d,weights[b]->data_size); bs[b]=nya_train_device_alloc(d,n*cols[b]*4); by[b]=nya_train_device_alloc(d,n*cols[b]*4);
        REQUIRE(bw[b] && bs[b] && by[b] && !nya_train_device_write(d,bw[b],0,weights[b]->data,weights[b]->data_size) && !nya_train_device_write(d,bs[b],0,seed[b],n*cols[b]*4));
    }
    double times[5],maximum[3]={0};
    for (unsigned repetition=0;repetition<7;++repetition) {
        REQUIRE(!nya_train_device_zero(d,dx) && !nya_train_device_zero(d,status) && !nya_train_device_finish(d));
        nya_train_device_stats before,after; nya_train_device_get_stats(d,&before);
        double start=tr_seconds();
        for (unsigned pass=0;pass<2;++pass) {
            nya_train_scope scope=nya_train_device_scratch_begin(d); REQUIRE(scope);
            nya_train_buffer projected[2],dr[2];
            for (size_t b=0;b<2;++b) {
                projected[b]=nya_train_device_alloc(d,n*cols[b]*4); dr[b]=nya_train_device_alloc(d,n*cols[b]*4); REQUIRE(projected[b] && dr[b]);
                REQUIRE(!nya_train_device_linear(d,projected[b],bw[b],weights[b]->type,cols[b],width,bx,n));
                REQUIRE(!nya_train_device_rope(d,by[b],(nya_train_view){projected[b],n,cols[b]},heads[b],dim,bf,0));
                REQUIRE(!nya_train_device_check_finite(d,status,by[b],n*cols[b],(uint32_t)b+1));
            }
            for (size_t reverse=0;reverse<2;++reverse) {
                size_t b=1-reverse;
                REQUIRE(!nya_train_device_rope_backward(d,dr[b],(nya_train_view){bs[b],n,cols[b]},heads[b],dim,bf,0));
                REQUIRE(!nya_train_device_linear_dx(d,dx,bw[b],weights[b]->type,cols[b],width,dr[b],n));
            }
            REQUIRE(!nya_train_device_check_finite(d,status,dx,n*width,3));
            REQUIRE(!nya_train_device_scratch_end(d,scope));
        }
        REQUIRE(!nya_train_device_finish(d));
        double ms=(tr_seconds()-start)*1000;
        nya_train_device_get_stats(d,&after);
        REQUIRE(after.kernel_launches==before.kernel_launches+30 && after.uploads==before.uploads && after.downloads==before.downloads &&
            after.synchronizations==before.synchronizations+1 && after.used_bytes==before.used_bytes);
        if (repetition>=2) times[repetition-2]=ms;
        /* Validate every repetition, outside the measured interval. */
        uint32_t tag=99; REQUIRE(!nya_train_device_read(d,status,0,&tag,4) && !tag);
        for (size_t b=0;b<3;++b) {
            size_t count=n*(b<2?cols[b]:width); const float *reference=b<2?expected[b]:nya_train_parameter_gradient(px);
            REQUIRE(!nya_train_device_read(d,b<2?by[b]:dx,0,actual,count*4));
            for (size_t j=0;j<count;++j) {
                double error=fabs((double)actual[j]-reference[j])/(1+fabs(reference[j]));
                if (!isfinite(actual[j]) || error>1e-5) fprintf(stderr,"QK component=%zu index=%zu CPU=%.9g GPU=%.9g error=%.9g\n",b,j,(double)reference[j],(double)actual[j],error);
                REQUIRE(isfinite(actual[j]) && error<=1e-5);
                if (error>maximum[b]) maximum[b]=error;
            }
        }
    }
    nya_train_device_stats stats; nya_train_device_get_stats(d,&stats);
    printf("{\"component\":\"frozen-qk-rope\",\"tokens\":%zu,\"inputs\":%zu,\"heads\":[%zu,%zu],\"dimension\":%zu,\"types\":[%u,%u],\"microbatches\":2,\"max_scaled_errors\":[%.9g,%.9g,%.9g],\"tolerance\":1e-5,\"peak_bytes\":%zu,\"live_bytes\":%zu,\"uploads\":%llu,\"downloads\":%llu,\"timing_accepted\":false,\"ms\":[",n,width,heads[0],heads[1],dim,weights[0]->type,weights[1]->type,maximum[0],maximum[1],maximum[2],stats.peak_bytes,stats.used_bytes,(unsigned long long)stats.uploads,(unsigned long long)stats.downloads);
    for (size_t k=0;k<5;++k) printf("%s%.9f",k?",":"",times[k]);
    printf("]}\n");
    for (size_t b=0;b<2;++b) { free(seed[b]); free(expected[b]); }
    free(x); free(freq); free(actual); nya_train_parameter_free(px); nya_train_device_free(d); return 0;
}
/* Complete frozen first-layer attention branch: Q/K/V projections, rotary
   positions, causal GQA, output projection, and shared input gradients. */
static int attention_branch(const nya_llm_context *model,size_t n)
{
    nya_train_device *d=NULL;
    const nya_llm_layer *layer=&model->layers[0];
    const nya_llm_tensor *weights[4]={layer->query,layer->key,layer->value,layer->attention_output};
    size_t heads=model->head_count,kv=model->key_value_head_count,width=model->embedding_length;
    REQUIRE(heads && kv && width && width%heads==0 && heads%kv==0);
    size_t dim=width/heads,cols[3]={width,kv*dim,kv*dim},count=n*width;
    REQUIRE(dim && !(dim%2) && n<=SIZE_MAX/4/width && isfinite(model->rope_frequency_base) && model->rope_frequency_base>=1);
    for (size_t b=0;b<4;++b) REQUIRE(weights[b] && weights[b]->dimensions[0]==width && weights[b]->dimensions[1]==(b<3?cols[b]:width));
    float *x=malloc(count*4),*seed=malloc(count*4),*expected=malloc(count*4),*actual=malloc(count*4),*freq=malloc(dim/2*4);
    REQUIRE(x && seed && expected && actual && freq);
    for (size_t j=0;j<count;++j) { x[j]=(float)((int)(j%71)-35)/64; seed[j]=(float)((int)(j%67)-33)/64; }
    for (size_t j=0;j<dim/2;++j) freq[j]=powf(model->rope_frequency_base,-2.0f*(float)j/(float)dim);
    float scale=1.0f/sqrtf((float)dim);
    nya_train_parameter *px=nya_train_parameter_create(n,width,x);
    nya_train_executor *executor=nya_train_executor_create(6); REQUIRE(px && executor);
    for (unsigned pass=0;pass<2;++pass) {
        nya_train_graph *g=nya_train_graph_create_with_executor(256*1024*1024,executor); REQUIRE(g);
        nya_train_tensor *input=nya_train_leaf(g,px),*projected[3]; REQUIRE(input);
        for (size_t b=0;b<3;++b) {
            projected[b]=nya_train_linear_mapped(input,weights[b]);
            if (b<2) projected[b]=nya_train_rope(projected[b],b?kv:heads,dim,freq,0);
            REQUIRE(projected[b]);
        }
        nya_train_tensor *attn=nya_train_attention(projected[0],projected[1],projected[2],heads,kv,dim,scale,0,NULL);
        nya_train_tensor *out=nya_train_linear_mapped(attn,weights[3]); REQUIRE(out);
        memcpy(expected,nya_train_data(out),count*4);
        REQUIRE(!nya_train_backward(nya_train_linear(nya_train_reshape(out,1,count),nya_train_input(g,1,count,seed))));
        nya_train_graph_free(g);
    }
    nya_train_executor_free(executor);
    d=nya_train_device_create("cuda",64*1024*1024); REQUIRE(d);
    nya_train_buffer bw[4];
    for (size_t b=0;b<4;++b) {
        bw[b]=nya_train_device_alloc(d,weights[b]->data_size); REQUIRE(bw[b] && !nya_train_device_write(d,bw[b],0,weights[b]->data,weights[b]->data_size));
    }
    nya_train_buffer bx=nya_train_device_alloc(d,count*4),bs=nya_train_device_alloc(d,count*4),bf=nya_train_device_alloc(d,dim/2*4);
    nya_train_buffer by=nya_train_device_alloc(d,count*4),dx=nya_train_device_alloc(d,count*4),status=nya_train_device_alloc(d,4);
    REQUIRE(bx && bs && bf && by && dx && status && !nya_train_device_write(d,bx,0,x,count*4) &&
        !nya_train_device_write(d,bs,0,seed,count*4) && !nya_train_device_write(d,bf,0,freq,dim/2*4));
    size_t work_bytes=nya_train_attention_workspace_bytes(n,heads); REQUIRE(work_bytes);
    double times[5],maximum[2]={0};
    for (unsigned repetition=0;repetition<7;++repetition) {
        REQUIRE(!nya_train_device_zero(d,dx) && !nya_train_device_zero(d,status) && !nya_train_device_finish(d));
        nya_train_device_stats before,after; nya_train_device_get_stats(d,&before);
        double start=tr_seconds();
        for (unsigned pass=0;pass<2;++pass) {
            nya_train_scope scope=nya_train_device_scratch_begin(d); REQUIRE(scope);
            nya_train_buffer projected[3],rotated[3],grads[3],linear_grads[3];
            for (size_t b=0;b<3;++b) {
                projected[b]=nya_train_device_alloc(d,n*cols[b]*4); grads[b]=nya_train_device_alloc(d,n*cols[b]*4);
                REQUIRE(projected[b] && grads[b]);
                rotated[b]=projected[b]; linear_grads[b]=grads[b];
                if (b<2) {
                    rotated[b]=nya_train_device_alloc(d,n*cols[b]*4); linear_grads[b]=nya_train_device_alloc(d,n*cols[b]*4);
                    REQUIRE(rotated[b] && linear_grads[b]);
                }
                REQUIRE(!nya_train_device_linear(d,projected[b],bw[b],weights[b]->type,cols[b],width,bx,n));
                if (b<2) REQUIRE(!nya_train_device_rope(d,rotated[b],(nya_train_view){projected[b],n,cols[b]},b?kv:heads,dim,bf,0));
            }
            nya_train_buffer attn=nya_train_device_alloc(d,count*4),da=nya_train_device_alloc(d,count*4);
            nya_train_buffer state=nya_train_device_alloc(d,n*heads*16),work=nya_train_device_alloc(d,work_bytes);
            REQUIRE(attn && da && state && work);
            nya_train_attention_desc desc={{rotated[0],n,cols[0]},{rotated[1],n,cols[1]},{rotated[2],n,cols[2]},heads,kv,dim,0,scale,0};
            REQUIRE(!nya_train_device_attention(d,attn,state,desc));
            REQUIRE(!nya_train_device_linear(d,by,bw[3],weights[3]->type,width,width,attn,n));
            REQUIRE(!nya_train_device_check_finite(d,status,by,count,1));
            REQUIRE(!nya_train_device_linear_dx(d,da,bw[3],weights[3]->type,width,width,bs,n));
            REQUIRE(!nya_train_device_attention_backward(d,grads[0],grads[1],grads[2],state,da,work,desc));
            for (size_t reverse=0;reverse<3;++reverse) {
                size_t b=2-reverse;
                if (b<2) REQUIRE(!nya_train_device_rope_backward(d,linear_grads[b],(nya_train_view){grads[b],n,cols[b]},b?kv:heads,dim,bf,0));
                REQUIRE(!nya_train_device_linear_dx(d,dx,bw[b],weights[b]->type,cols[b],width,linear_grads[b],n));
            }
            REQUIRE(!nya_train_device_check_finite(d,status,dx,count,2));
            REQUIRE(!nya_train_device_scratch_end(d,scope));
        }
        REQUIRE(!nya_train_device_finish(d));
        double ms=(tr_seconds()-start)*1000;
        nya_train_device_get_stats(d,&after);
        REQUIRE(after.kernel_launches==before.kernel_launches+58+6*(n/16+(n%16!=0)) && after.uploads==before.uploads &&
            after.downloads==before.downloads && after.synchronizations==before.synchronizations+1 && after.used_bytes==before.used_bytes);
        if (repetition>=2) times[repetition-2]=ms;
        uint32_t tag=99; REQUIRE(!nya_train_device_read(d,status,0,&tag,4) && !tag);
        for (unsigned gradient=0;gradient<2;++gradient) {
            const float *reference=gradient?nya_train_parameter_gradient(px):expected;
            REQUIRE(!nya_train_device_read(d,gradient?dx:by,0,actual,count*4));
            for (size_t j=0;j<count;++j) {
                double error=fabs((double)actual[j]-reference[j])/(1+fabs(reference[j]));
                if (!isfinite(actual[j]) || error>1e-5) fprintf(stderr,"attention gradient=%u index=%zu CPU=%.9g GPU=%.9g error=%.9g\n",gradient,j,(double)reference[j],(double)actual[j],error);
                REQUIRE(isfinite(actual[j]) && error<=1e-5);
                if (error>maximum[gradient]) maximum[gradient]=error;
            }
        }
    }
    nya_train_device_stats stats; nya_train_device_get_stats(d,&stats);
    printf("{\"component\":\"frozen-attention-branch\",\"tokens\":%zu,\"inputs\":%zu,\"heads\":[%zu,%zu],\"dimension\":%zu,\"microbatches\":2,\"max_scaled_output_error\":%.9g,\"max_scaled_input_gradient_error\":%.9g,\"tolerance\":1e-5,\"attention_state_bytes\":%zu,\"attention_workspace_bytes\":%zu,\"peak_bytes\":%zu,\"live_bytes\":%zu,\"uploads\":%llu,\"downloads\":%llu,\"timing_accepted\":false,\"ms\":[",n,width,heads,kv,dim,maximum[0],maximum[1],n*heads*16,work_bytes,stats.peak_bytes,stats.used_bytes,(unsigned long long)stats.uploads,(unsigned long long)stats.downloads);
    for (size_t j=0;j<5;++j) printf("%s%.9f",j?",":"",times[j]);
    printf("]}\n");
    free(x);free(seed);free(expected);free(actual);free(freq);nya_train_parameter_free(px);nya_train_device_free(d);return 0;
}
/* Real quantized vocabulary projection plus masked CE or paired DPO. This
   validates composition/residency, not full-model throughput or convergence. */
static int loss_head(const nya_llm_context *model,size_t n,int paired)
{
    nya_train_device *d=NULL;
    const nya_llm_tensor *w=model->output?model->output:model->token_embedding;
    REQUIRE(w && w->dimensions[0] && w->dimensions[1]);
    size_t width=(size_t)w->dimensions[0],vocab=(size_t)w->dimensions[1],branches=paired?2:1;
    REQUIRE(n && width<=SIZE_MAX/4/n && vocab<=UINT32_MAX && vocab<=SIZE_MAX/4/n);
    size_t count=n*width,logits_count=n*vocab;
    float *x[2]={0},*expected[2]={0},*actual=malloc((count>logits_count?count:logits_count)*4);
    uint32_t *labels=malloc(n*4); unsigned char *mask=malloc(n);
    nya_train_parameter *parameters[2]={0};
    REQUIRE(actual && labels && mask);
    for (size_t j=0;j<n;++j) { mask[j]=(unsigned char)(j%3?255:0); labels[j]=mask[j]?(uint32_t)((j*937)%vocab):UINT32_MAX; }
    for (size_t b=0;b<branches;++b) {
        x[b]=malloc(count*4); expected[b]=malloc(logits_count*4); REQUIRE(x[b] && expected[b]);
        for (size_t j=0;j<count;++j) x[b][j]=(float)((int)((j+b*17)%71)-35)/64;
        parameters[b]=nya_train_parameter_create(n,width,x[b]); REQUIRE(parameters[b]);
    }
    nya_train_executor *executor=nya_train_executor_create(6); REQUIRE(executor);
    float expected_loss=0; double refs[2]={0};
    for (size_t micro=0;micro<2;++micro) {
        nya_train_graph *g=nya_train_graph_create_with_executor(256*1024*1024,executor); REQUIRE(g);
        nya_train_tensor *losses[2]={0};
        for (size_t b=0;b<branches;++b) {
            nya_train_tensor *logits=nya_train_linear_mapped(nya_train_leaf(g,parameters[b]),w); REQUIRE(logits);
            memcpy(expected[b],nya_train_data(logits),logits_count*4);
            losses[b]=paired?nya_train_logprob(logits,labels,mask,n):nya_train_cross_entropy(logits,labels,mask,n); REQUIRE(losses[b]);
            refs[b]=(double)*nya_train_data(losses[b])+(b?0.5:0);
        }
        nya_train_tensor *loss=paired?nya_train_dpo(losses[0],losses[1],refs[0],refs[1],0.2f):losses[0]; REQUIRE(loss);
        expected_loss=*nya_train_data(loss); REQUIRE(!nya_train_backward(loss));
        nya_train_graph_free(g);
    }
    nya_train_executor_free(executor);
    d=nya_train_device_create("cuda",128*1024*1024); REQUIRE(d);
    nya_train_buffer bw=nya_train_device_alloc(d,w->data_size),bl=nya_train_device_alloc(d,n*4),bm=nya_train_device_alloc(d,n);
    nya_train_buffer seed=nya_train_device_alloc(d,4),status=nya_train_device_alloc(d,4),y=nya_train_device_alloc(d,4);
    nya_train_buffer bx[2]={0},dx[2]={0},logits[2]={0};
    float one=1;
    REQUIRE(bw && bl && bm && seed && status && y && !nya_train_device_write(d,bw,0,w->data,w->data_size) &&
        !nya_train_device_write(d,bl,0,labels,n*4) && !nya_train_device_write(d,bm,0,mask,n) && !nya_train_device_write(d,seed,0,&one,4));
    for (size_t b=0;b<branches;++b) {
        bx[b]=nya_train_device_alloc(d,count*4); dx[b]=nya_train_device_alloc(d,count*4); logits[b]=nya_train_device_alloc(d,logits_count*4);
        REQUIRE(bx[b] && dx[b] && logits[b] && !nya_train_device_write(d,bx[b],0,x[b],count*4));
    }
    double maximum[3]={0},times[5];
    for (size_t rep=0;rep<7;++rep) {
        for (size_t b=0;b<branches;++b) REQUIRE(!nya_train_device_zero(d,dx[b]));
        REQUIRE(!nya_train_device_zero(d,status) && !nya_train_device_finish(d));
        nya_train_device_stats before,after; nya_train_device_get_stats(d,&before);
        double start=tr_seconds();
        for (size_t micro=0;micro<2;++micro) {
            nya_train_scope scope=nya_train_device_scratch_begin(d); REQUIRE(scope);
            nya_train_buffer ls[2]={0},ly[2]={0},lg[2]={0},dl[2]={0};
            for (size_t b=0;b<branches;++b) {
                ls[b]=nya_train_device_alloc(d,nya_train_loss_state_bytes(n)); ly[b]=paired?nya_train_device_alloc(d,4):y;
                lg[b]=paired?nya_train_device_alloc(d,4):seed; dl[b]=nya_train_device_alloc(d,logits_count*4);
                REQUIRE(ls[b] && ly[b] && lg[b] && dl[b]);
                REQUIRE(!nya_train_device_linear(d,logits[b],bw,w->type,vocab,width,bx[b],n));
                REQUIRE(!nya_train_device_loss(d,ly[b],ls[b],(nya_train_view){logits[b],n,vocab},bl,bm,paired?NYA_TRAIN_LOSS_LOGPROB:NYA_TRAIN_LOSS_CE));
                REQUIRE(!nya_train_device_check_finite(d,status,logits[b],logits_count,1+(uint32_t)b));
                REQUIRE(!nya_train_device_check_finite(d,status,ly[b],1,3+(uint32_t)b));
            }
            if (paired) {
                nya_train_buffer ds=nya_train_device_alloc(d,8); REQUIRE(ds);
                REQUIRE(!nya_train_device_dpo(d,y,ds,ly[0],ly[1],refs[0],refs[1],0.2f));
                REQUIRE(!nya_train_device_dpo_backward(d,lg[0],lg[1],ds,seed));
                REQUIRE(!nya_train_device_check_finite(d,status,y,1,5));
            }
            for (size_t b=0;b<branches;++b) {
                REQUIRE(!nya_train_device_loss_backward(d,dl[b],ls[b],(nya_train_view){logits[b],n,vocab},bl,bm,lg[b]));
                REQUIRE(!nya_train_device_linear_dx(d,dx[b],bw,w->type,vocab,width,dl[b],n));
                REQUIRE(!nya_train_device_check_finite(d,status,dx[b],count,6+(uint32_t)b));
            }
            REQUIRE(!nya_train_device_scratch_end(d,scope));
        }
        REQUIRE(!nya_train_device_finish(d));
        double ms=(tr_seconds()-start)*1000;
        nya_train_device_get_stats(d,&after);
        REQUIRE(after.kernel_launches==before.kernel_launches+(paired?56:20) && after.uploads==before.uploads &&
            after.downloads==before.downloads && after.synchronizations==before.synchronizations+1 && after.used_bytes==before.used_bytes);
        if (rep>=2) times[rep-2]=ms;
        uint32_t tag=99; REQUIRE(!nya_train_device_read(d,status,0,&tag,4) && !tag);
        float loss; REQUIRE(!nya_train_device_read(d,y,0,&loss,4));
        double err=fabs((double)loss-expected_loss)/(1+fabs(expected_loss)); REQUIRE(isfinite(loss) && err<=1e-5);
        if (err>maximum[0]) maximum[0]=err;
        for (size_t b=0;b<branches;++b) for (size_t gradient=0;gradient<2;++gradient) {
            size_t size=gradient?count:logits_count;
            const float *reference=gradient?nya_train_parameter_gradient(parameters[b]):expected[b];
            REQUIRE(!nya_train_device_read(d,gradient?dx[b]:logits[b],0,actual,size*4));
            for (size_t j=0;j<size;++j) {
                double error=fabs((double)actual[j]-reference[j])/(1+fabs(reference[j]));
                REQUIRE(isfinite(actual[j]) && error<=1e-5);
                if (error>maximum[1+gradient]) maximum[1+gradient]=error;
            }
        }
    }
    nya_train_device_stats stats; nya_train_device_get_stats(d,&stats);
    printf("{\"component\":\"frozen-vocabulary-%s\",\"weight_type\":%u,\"tokens\":%zu,\"width\":%zu,\"vocabulary\":%zu,\"microbatches\":2,\"cpu_loss\":%.9g,\"max_scaled_loss_error\":%.9g,\"max_scaled_logits_error\":%.9g,\"max_scaled_input_gradient_error\":%.9g,\"tolerance\":1e-5,\"loss_state_bytes_per_branch\":%zu,\"peak_bytes\":%zu,\"live_bytes\":%zu,\"uploads\":%llu,\"downloads\":%llu,\"timing_accepted\":false,\"ms\":[",paired?"dpo":"ce",w->type,n,width,vocab,(double)expected_loss,maximum[0],maximum[1],maximum[2],nya_train_loss_state_bytes(n),stats.peak_bytes,stats.used_bytes,(unsigned long long)stats.uploads,(unsigned long long)stats.downloads);
    for (size_t j=0;j<5;++j) printf("%s%.9f",j?",":"",times[j]);
    printf("]}\n");
    for (size_t b=0;b<branches;++b) { free(x[b]); free(expected[b]); nya_train_parameter_free(parameters[b]); }
    free(actual); free(labels); free(mask); nya_train_device_free(d); return 0;
}
/* Real frozen vocabulary weights with trainable rank-four LoRA matrices.
   CPU and GPU run the same 32-step trajectory; no full-model claim is made. */
static int optimizer_head(const nya_llm_context *model,size_t n)
{
    nya_train_device *d=NULL;
    const nya_llm_tensor *w=model->output?model->output:model->token_embedding;
    REQUIRE(w && w->dimensions[0] && w->dimensions[1]);
    size_t width=(size_t)w->dimensions[0],vocab=(size_t)w->dimensions[1],rank=4;
    REQUIRE(n && width<=SIZE_MAX/4/n && vocab<=UINT32_MAX && vocab<=SIZE_MAX/4/n && width<=SIZE_MAX/16 && vocab<=SIZE_MAX/16);
    size_t sizes[]={rank*width,vocab*rank},logits_count=n*vocab;
    float *input=malloc(n*width*4),*initial=malloc(sizes[0]*4),*base=malloc(logits_count*4),*actual=malloc((sizes[0]>sizes[1]?sizes[0]:sizes[1])*4);
    uint32_t *labels=malloc(n*4); unsigned char *mask=malloc(n); REQUIRE(input && initial && base && actual && labels && mask);
    for (size_t i=0;i<n*width;++i) input[i]=(float)((int)(i%71)-35)/64;
    for (size_t i=0;i<sizes[0];++i) initial[i]=(float)((int)(i%31)-15)/256;
    for (size_t i=0;i<n;++i) { mask[i]=(unsigned char)(i%3?1:0); labels[i]=mask[i]?(uint32_t)(i*937%vocab):UINT32_MAX; }
    nya_train_parameter *parameters[]={nya_train_parameter_create(rank,width,initial),nya_train_parameter_create(vocab,rank,NULL)};
    nya_train_executor *executor=nya_train_executor_create(6); REQUIRE(parameters[0] && parameters[1] && executor);
    nya_train_graph *g=nya_train_graph_create_for_evaluation(128*1024*1024,executor); REQUIRE(g);
    nya_train_tensor *frozen=nya_train_linear_mapped(nya_train_input(g,n,width,input),w); REQUIRE(frozen);
    memcpy(base,nya_train_data(frozen),logits_count*4); nya_train_graph_free(g);
    d=nya_train_device_create("cuda",128*1024*1024); REQUIRE(d);
    nya_train_adamw_tensor tensors[2];
    for (size_t p=0;p<2;++p) {
        tensors[p]=(nya_train_adamw_tensor){nya_train_device_alloc(d,sizes[p]*4),nya_train_device_alloc(d,sizes[p]*4),
            nya_train_device_alloc(d,sizes[p]*4),nya_train_device_alloc(d,sizes[p]*4),sizes[p]};
        REQUIRE(tensors[p].values && tensors[p].gradient && tensors[p].moment && tensors[p].variance);
    }
    REQUIRE(!nya_train_device_write(d,tensors[0].values,0,initial,sizes[0]*4));
    nya_train_optimizer_plan plan=nya_train_device_adamw_plan(d,tensors,2); REQUIRE(plan);
    nya_train_buffer step=nya_train_device_alloc(d,8),status=nya_train_device_alloc(d,4),norm=nya_train_device_alloc(d,8);
    nya_train_buffer bw=nya_train_device_alloc(d,w->data_size),x=nya_train_device_alloc(d,n*width*4),bl=nya_train_device_alloc(d,n*4),bm=nya_train_device_alloc(d,n);
    nya_train_buffer bbase=nya_train_device_alloc(d,logits_count*4),low=nya_train_device_alloc(d,n*rank*4),branch=nya_train_device_alloc(d,logits_count*4);
    nya_train_buffer scaled=nya_train_device_alloc(d,logits_count*4),logits=nya_train_device_alloc(d,logits_count*4),dl=nya_train_device_alloc(d,logits_count*4);
    nya_train_buffer db=nya_train_device_alloc(d,logits_count*4),da=nya_train_device_alloc(d,n*rank*4);
    nya_train_buffer loss=nya_train_device_alloc(d,4),loss_state=nya_train_device_alloc(d,nya_train_loss_state_bytes(n)),seed=nya_train_device_alloc(d,4);
    REQUIRE(step && status && norm && bw && x && bl && bm && bbase && low && branch && scaled && logits && dl && db && da && loss && loss_state && seed);
    float one=1;
    REQUIRE(!nya_train_device_write(d,bw,0,w->data,w->data_size) && !nya_train_device_write(d,x,0,input,n*width*4) &&
        !nya_train_device_write(d,bl,0,labels,n*4) && !nya_train_device_write(d,bm,0,mask,n) && !nya_train_device_write(d,seed,0,&one,4));
    REQUIRE(!nya_train_device_linear(d,bbase,bw,w->type,vocab,width,x,n));
    nya_train_adamw optimizer; nya_train_adamw_defaults(&optimizer); optimizer.learning_rate=0.04f; optimizer.weight_decay=0.01f;
    double maximum[3]={0},times[32]; float losses[32],cpu_losses[32];
    for (size_t iteration=0;iteration<32;++iteration) {
        for (size_t p=0;p<2;++p) nya_train_zero_grad(parameters[p]);
        g=nya_train_graph_create_with_executor(256*1024*1024,executor); REQUIRE(g);
        nya_train_tensor *a=nya_train_linear(nya_train_input(g,n,width,input),nya_train_leaf(g,parameters[0]));
        nya_train_tensor *b=nya_train_linear(a,nya_train_leaf(g,parameters[1]));
        nya_train_tensor *output=nya_train_add(nya_train_input(g,n,vocab,base),nya_train_scale(b,0.5f));
        nya_train_tensor *objective=nya_train_cross_entropy(output,labels,mask,n); REQUIRE(objective);
        cpu_losses[iteration]=*nya_train_data(objective); REQUIRE(!nya_train_backward(objective)); nya_train_graph_free(g);
        optimizer.learning_rate*=0.98f;
        char error[256]; REQUIRE(!nya_train_adamw_step(&optimizer,parameters,2,error,sizeof(error)));
        REQUIRE(!nya_train_device_finish(d));
        nya_train_device_stats before,after; nya_train_device_get_stats(d,&before);
        double start=tr_seconds();
        for (size_t p=0;p<2;++p) REQUIRE(!nya_train_device_zero(d,tensors[p].gradient));
        REQUIRE(!nya_train_device_zero(d,dl) && !nya_train_device_zero(d,da));
        REQUIRE(!nya_train_device_linear(d,low,tensors[0].values,0,rank,width,x,n));
        REQUIRE(!nya_train_device_linear(d,branch,tensors[1].values,0,vocab,rank,low,n));
        REQUIRE(!nya_train_device_unary(d,scaled,branch,logits_count,NYA_TRAIN_UNARY_SCALE,0.5));
        REQUIRE(!nya_train_device_binary(d,logits,(nya_train_view){bbase,n,vocab},(nya_train_view){scaled,n,vocab},NYA_TRAIN_BINARY_ADD));
        REQUIRE(!nya_train_device_loss(d,loss,loss_state,(nya_train_view){logits,n,vocab},bl,bm,NYA_TRAIN_LOSS_CE));
        REQUIRE(!nya_train_device_check_finite(d,status,loss,1,11));
        REQUIRE(!nya_train_device_loss_backward(d,dl,loss_state,(nya_train_view){logits,n,vocab},bl,bm,seed));
        REQUIRE(!nya_train_device_unary(d,db,dl,logits_count,NYA_TRAIN_UNARY_SCALE,0.5));
        REQUIRE(!nya_train_device_linear_dw(d,tensors[1].gradient,low,db,vocab,rank,n));
        REQUIRE(!nya_train_device_linear_dx(d,da,tensors[1].values,0,vocab,rank,db,n));
        REQUIRE(!nya_train_device_linear_dw(d,tensors[0].gradient,x,da,rank,width,n));
        nya_train_adamw_config config={optimizer.learning_rate,optimizer.beta1,optimizer.beta2,optimizer.epsilon,optimizer.weight_decay,optimizer.max_grad_norm};
        REQUIRE(!nya_train_device_adamw(d,plan,config,step,status,norm,17) && !nya_train_device_finish(d));
        times[iteration]=(tr_seconds()-start)*1000;
        nya_train_device_get_stats(d,&after);
        REQUIRE(after.kernel_launches==before.kernel_launches+21 && after.used_bytes==before.used_bytes && after.uploads==before.uploads &&
            after.downloads==before.downloads && after.synchronizations==before.synchronizations+1);
        uint64_t actual_step=0; uint32_t tag=99;
        REQUIRE(!nya_train_device_read(d,step,0,&actual_step,8) && actual_step==iteration+1 && !nya_train_device_read(d,status,0,&tag,4) && !tag);
        REQUIRE(!nya_train_device_read(d,loss,0,&losses[iteration],4));
        double loss_error=fabs((double)losses[iteration]-cpu_losses[iteration])/(1+fabs(cpu_losses[iteration]));
        REQUIRE(isfinite(losses[iteration]) && loss_error<=1e-5); if (loss_error>maximum[0]) maximum[0]=loss_error;
        for (size_t p=0;p<2;++p) for (size_t grad=0;grad<2;++grad) {
            REQUIRE(!nya_train_device_read(d,grad?tensors[p].gradient:tensors[p].values,0,actual,sizes[p]*4));
            const float *reference=grad?nya_train_parameter_gradient(parameters[p]):nya_train_parameter_data(parameters[p]);
            for (size_t i=0;i<sizes[p];++i) {
                double delta=fabs((double)actual[i]-reference[i])/(1+fabs(reference[i]));
                if (!isfinite(actual[i]) || delta>1e-5) fprintf(stderr,"iteration=%zu parameter=%zu grad=%zu index=%zu CPU=%.9g GPU=%.9g error=%.9g\n",iteration,p,grad,i,(double)reference[i],(double)actual[i],delta);
                REQUIRE(isfinite(actual[i]) && delta<=1e-5); if (delta>maximum[1+grad]) maximum[1+grad]=delta;
            }
        }
    }
    REQUIRE(losses[31]<losses[0]*0.9f);
    nya_train_device_stats stats; nya_train_device_get_stats(d,&stats);
    printf("{\"component\":\"real-vocabulary-lora-adamw\",\"tokens\":%zu,\"width\":%zu,\"vocabulary\":%zu,\"rank\":%zu,\"steps\":32,\"max_scaled_loss_error\":%.9g,\"max_scaled_parameter_error\":%.9g,\"max_scaled_gradient_error\":%.9g,\"tolerance\":1e-5,\"optimizer_plan_bytes\":%zu,\"peak_bytes\":%zu,\"live_bytes\":%zu,\"uploads\":%llu,\"downloads\":%llu,\"timing_accepted\":false,\"losses\":[",n,width,vocab,rank,maximum[0],maximum[1],maximum[2],nya_train_adamw_plan_bytes(tensors,2),stats.peak_bytes,stats.used_bytes,(unsigned long long)stats.uploads,(unsigned long long)stats.downloads);
    for (size_t i=0;i<32;++i) printf("%s%.9g",i?",":"",(double)losses[i]);
    printf("],\"cpu_losses\":[");
    for (size_t i=0;i<32;++i) printf("%s%.9g",i?",":"",(double)cpu_losses[i]);
    printf("],\"step_ms\":[");
    for (size_t i=0;i<32;++i) printf("%s%.9f",i?",":"",times[i]);
    printf("]}\n");
    free(input); free(initial); free(base); free(actual); free(labels); free(mask);
    nya_train_parameter_free(parameters[0]); nya_train_parameter_free(parameters[1]); nya_train_executor_free(executor); nya_train_device_free(d); return 0;
}
int main(int argc,char **argv)
{
    nya_train_device *d=NULL;
    REQUIRE(argc==3);
    if (!strcmp(argv[1],"--startup")) return startup(argv[2]);
    FILE *f=fopen(argv[1],"rb"); REQUIRE(f);
    REQUIRE(!fseek(f,0,SEEK_END)); long size=ftell(f); fclose(f); REQUIRE(size>0);
    nya_llm_context *m=NULL; char error[256];
    if (nya_llm_load(argv[1],(uint64_t)size,&m,error,sizeof(error))) { fprintf(stderr,"%s\n",error); return 1; }
    if (!strcmp(argv[2],"adamw8") || !strcmp(argv[2],"adamw64")) {
        int result=optimizer_head(m,!strcmp(argv[2],"adamw8")?8u:64u); nya_llm_free(m); return result;
    }
    if (!strcmp(argv[2],"loss8") || !strcmp(argv[2],"loss64") || !strcmp(argv[2],"dpo8") || !strcmp(argv[2],"dpo64")) {
        int result=loss_head(m,(!strcmp(argv[2],"loss8") || !strcmp(argv[2],"dpo8"))?8u:64u,!strncmp(argv[2],"dpo",3)); nya_llm_free(m); return result;
    }
    if (!strcmp(argv[2],"attention8") || !strcmp(argv[2],"attention64")) {
        int result=attention_branch(m,!strcmp(argv[2],"attention8")?8u:64u); nya_llm_free(m); return result;
    }
    if (!strcmp(argv[2],"qkrope8") || !strcmp(argv[2],"qkrope64")) {
        int result=qk_rope(m,!strcmp(argv[2],"qkrope8")?8u:64u); nya_llm_free(m); return result;
    }
    if (!strcmp(argv[2],"ffn8") || !strcmp(argv[2],"ffn64") || !strcmp(argv[2],"rmsffn8") || !strcmp(argv[2],"rmsffn64") ||
        !strcmp(argv[2],"embedffn8") || !strcmp(argv[2],"embedffn64")) {
        int embedded=!strncmp(argv[2],"embed",5), normalized=embedded || !strncmp(argv[2],"rms",3);
        int result=ffn(m,(!strcmp(argv[2],"ffn8") || !strcmp(argv[2],"rmsffn8") || !strcmp(argv[2],"embedffn8"))?8u:64u,normalized,embedded);
        nya_llm_free(m); return result;
    }
    const nya_llm_tensor *weights[]={m->layers[0].query,m->layers[0].feed_forward_down,m->layers[0].feed_forward_up};
    /* Reuse mode retires each complete matrix job before admitting the next. */
    int checked=!strcmp(argv[2],"checked-forward") || !strcmp(argv[2],"checked-reverse");
    int recycle=checked || !strcmp(argv[2],"reuse-forward") || !strcmp(argv[2],"reuse-reverse");
    d=nya_train_device_create("cuda",(recycle?64u:256u)*1024*1024); REQUIRE(d);
    nya_train_buffer status=checked?nya_train_device_alloc(d,4):0; REQUIRE(!checked || status);
    int reverse=!strcmp(argv[2],"reverse") || !strcmp(argv[2],"reuse-reverse") || !strcmp(argv[2],"checked-reverse");
    for (size_t j=0;j<6;++j) {
        size_t job=reverse?5-j:j;
        REQUIRE(!run(d,weights[job/2],job%2?64:8,recycle,status));
    }
    if (checked) {
        uint32_t failure=UINT32_MAX;
        REQUIRE(!nya_train_device_read(d,status,0,&failure,4) && !failure);
    }
    nya_train_device_stats stats; nya_train_device_get_stats(d,&stats);
    fprintf(stderr,"arena_capacity=%zu arena_used=%zu buffers=%zu launches=%llu uploads=%llu upload_bytes=%llu downloads=%llu download_bytes=%llu peak_bytes=%zu scratch_resets=%llu\n",
        stats.capacity_bytes,stats.used_bytes,stats.buffers,(unsigned long long)stats.kernel_launches,(unsigned long long)stats.uploads,
        (unsigned long long)stats.upload_bytes,(unsigned long long)stats.downloads,(unsigned long long)stats.download_bytes,
        stats.peak_bytes,(unsigned long long)stats.scratch_resets);
    nya_train_device_free(d); nya_llm_free(m); return 0;
}
