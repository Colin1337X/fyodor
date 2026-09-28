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
static int ffn(const nya_llm_context *model,size_t n)
{
    nya_train_device *d=NULL;
    const nya_llm_tensor *w[]={model->layers[0].feed_forward_gate,model->layers[0].feed_forward_up,model->layers[0].feed_forward_down};
    REQUIRE(w[0] && w[1] && w[2]);
    size_t i=(size_t)w[0]->dimensions[0], h=(size_t)w[0]->dimensions[1];
    REQUIRE(i && h && w[1]->dimensions[0]==i && w[1]->dimensions[1]==h && w[2]->dimensions[0]==h && w[2]->dimensions[1]==i);
    REQUIRE(n<=SIZE_MAX/4/i && n<=SIZE_MAX/4/h);
    size_t count=n*i, hidden=n*h;
    float *x=malloc(count*4), *dy=malloc(count*4), *expected=malloc(count*4), *actual=malloc(count*4);
    REQUIRE(x && dy && expected && actual);
    for (size_t k=0;k<count;++k) { x[k]=(float)((int)(k%71)-35)/64; dy[k]=(float)((int)(k%67)-33)/64; }
    nya_train_parameter *px=nya_train_parameter_create(n,i,x);
    nya_train_executor *executor=nya_train_executor_create(6); REQUIRE(px && executor);
    for (unsigned pass=0;pass<2;++pass) {
        nya_train_graph *g=nya_train_graph_create_with_executor(256*1024*1024,executor); REQUIRE(g);
        nya_train_tensor *input=nya_train_leaf(g,px);
        nya_train_tensor *a=nya_train_linear_mapped(input,w[0]), *b=nya_train_linear_mapped(input,w[1]);
        nya_train_tensor *out=nya_train_linear_mapped(nya_train_mul(nya_train_silu(a),b),w[2]); REQUIRE(out);
        memcpy(expected,nya_train_data(out),count*4);
        REQUIRE(!nya_train_backward(nya_train_linear(nya_train_reshape(out,1,count),nya_train_input(g,1,count,dy))));
        nya_train_graph_free(g);
    }
    nya_train_executor_free(executor);
    d=nya_train_device_create("cuda",64*1024*1024); REQUIRE(d);
    nya_train_buffer weights[3];
    for (size_t k=0;k<3;++k) {
        weights[k]=nya_train_device_alloc(d,w[k]->data_size); REQUIRE(weights[k]);
        REQUIRE(!nya_train_device_write(d,weights[k],0,w[k]->data,w[k]->data_size));
    }
    nya_train_buffer input=nya_train_device_alloc(d,count*4), seed=nya_train_device_alloc(d,count*4);
    nya_train_buffer output=nya_train_device_alloc(d,count*4), dx=nya_train_device_alloc(d,count*4), status=nya_train_device_alloc(d,4);
    REQUIRE(input && seed && output && dx && status);
    REQUIRE(!nya_train_device_write(d,input,0,x,count*4) && !nya_train_device_write(d,seed,0,dy,count*4));
    double times[5];
    for (unsigned repetition=0;repetition<7;++repetition) {
        REQUIRE(!nya_train_device_zero(d,dx) && !nya_train_device_zero(d,status) && !nya_train_device_finish(d));
        nya_train_device_stats before,after; nya_train_device_get_stats(d,&before);
        double start=tr_seconds();
        for (unsigned pass=0;pass<2;++pass) {
            nya_train_scope scope=nya_train_device_scratch_begin(d); REQUIRE(scope);
            nya_train_buffer a=nya_train_device_alloc(d,hidden*4), b=nya_train_device_alloc(d,hidden*4), s=nya_train_device_alloc(d,hidden*4), m=nya_train_device_alloc(d,hidden*4);
            nya_train_buffer dm=nya_train_device_alloc(d,hidden*4), ds=nya_train_device_alloc(d,hidden*4), db=nya_train_device_alloc(d,hidden*4), da=nya_train_device_alloc(d,hidden*4);
            REQUIRE(a && b && s && m && dm && ds && db && da);
            REQUIRE(!nya_train_device_linear(d,a,weights[0],w[0]->type,h,i,input,n));
            REQUIRE(!nya_train_device_linear(d,b,weights[1],w[1]->type,h,i,input,n));
            REQUIRE(!nya_train_device_unary(d,s,a,hidden,NYA_TRAIN_UNARY_SILU,0));
            REQUIRE(!nya_train_device_binary(d,m,(nya_train_view){s,n,h},(nya_train_view){b,n,h},NYA_TRAIN_BINARY_MUL));
            REQUIRE(!nya_train_device_linear(d,output,weights[2],w[2]->type,i,h,m,n));
            REQUIRE(!nya_train_device_linear_dx(d,dm,weights[2],w[2]->type,i,h,seed,n));
            REQUIRE(!nya_train_device_binary_backward(d,ds,db,(nya_train_view){s,n,h},(nya_train_view){b,n,h},dm,NYA_TRAIN_BINARY_MUL));
            REQUIRE(!nya_train_device_unary_backward(d,da,a,ds,hidden,NYA_TRAIN_UNARY_SILU,0));
            REQUIRE(!nya_train_device_linear_dx(d,dx,weights[1],w[1]->type,h,i,db,n));
            REQUIRE(!nya_train_device_linear_dx(d,dx,weights[0],w[0]->type,h,i,da,n));
            REQUIRE(!nya_train_device_check_finite(d,status,output,count,1) && !nya_train_device_check_finite(d,status,dx,count,2));
            REQUIRE(!nya_train_device_scratch_end(d,scope));
        }
        REQUIRE(!nya_train_device_finish(d));
        double ms=(tr_seconds()-start)*1000;
        nya_train_device_get_stats(d,&after);
        REQUIRE(after.kernel_launches==before.kernel_launches+40 && after.uploads==before.uploads &&
            after.downloads==before.downloads && after.synchronizations==before.synchronizations+1 && after.used_bytes==before.used_bytes);
        if (repetition>=2) times[repetition-2]=ms;
    }
    uint32_t failure=1; REQUIRE(!nya_train_device_read(d,status,0,&failure,4) && !failure);
    double maximum[2]={0,0};
    for (unsigned gradient=0;gradient<2;++gradient) {
        const float *reference=gradient?nya_train_parameter_gradient(px):expected;
        REQUIRE(!nya_train_device_read(d,gradient?dx:output,0,actual,count*4));
        for (size_t k=0;k<count;++k) {
            double error=fabs((double)actual[k]-reference[k])/(1+fabs(reference[k]));
            REQUIRE(isfinite(actual[k]) && error<=1e-5);
            if (error>maximum[gradient]) maximum[gradient]=error;
        }
    }
    nya_train_device_stats stats; nya_train_device_get_stats(d,&stats);
    printf("{\"component\":\"frozen-silu-ffn\",\"tokens\":%zu,\"inputs\":%zu,\"hidden\":%zu,\"microbatches\":2,\"max_scaled_output_error\":%.9g,\"max_scaled_input_gradient_error\":%.9g,\"tolerance\":1e-5,\"peak_bytes\":%zu,\"live_bytes\":%zu,\"uploads\":%llu,\"downloads\":%llu,\"timing_accepted\":false,\"ms\":[",
        n,i,h,maximum[0],maximum[1],stats.peak_bytes,stats.used_bytes,(unsigned long long)stats.uploads,(unsigned long long)stats.downloads);
    for (size_t k=0;k<5;++k) printf("%s%.9f",k?",":"",times[k]);
    printf("]}\n");
    nya_train_device_free(d); nya_train_parameter_free(px);
    free(x); free(dy); free(expected); free(actual);
    return 0;
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
    if (!strcmp(argv[2],"ffn8") || !strcmp(argv[2],"ffn64")) {
        int result=ffn(m,!strcmp(argv[2],"ffn8")?8u:64u); nya_llm_free(m); return result;
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
