/* Resident matrix microbenchmark, not a complete training throughput test. */
#include "training_device.h"
#include "llm_internal.h"
#include "train_clock.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define REQUIRE(x) do { if (!(x)) { fprintf(stderr,"line %d: %s; %s\n",__LINE__,#x,nya_train_device_error(d)); return 1; } } while (0)
static int dispatch(nya_train_device *d, unsigned op, nya_train_buffer y, nya_train_buffer w, nya_train_buffer x,
    nya_train_buffer dy, nya_train_buffer dx, nya_train_buffer dw, unsigned type, size_t o, size_t i, size_t n)
{
    return op==0 ? nya_train_device_linear(d,y,w,type,o,i,x,n) : op==1 ?
        nya_train_device_linear_dx(d,dx,w,type,o,i,dy,n) : nya_train_device_linear_dw(d,dw,x,dy,o,i,n);
}
static int run(nya_train_device *d, const nya_llm_tensor *w, size_t n)
{
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
        REQUIRE(!dispatch(d,op,by,bw,bx,bd,dx,dw,w->type,o,i,n));
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
            for (unsigned repeat=0;repeat<20;++repeat) REQUIRE(!dispatch(d,op,by,bw,bx,bd,dx,dw,w->type,o,i,n));
            REQUIRE(!nya_train_device_finish(d));
            double ms=(tr_seconds()-start)*1000/20;
            nya_train_device_get_stats(d,&after);
            REQUIRE(after.uploads==before.uploads && after.downloads==before.downloads && after.synchronizations==before.synchronizations+1 && after.kernel_launches==before.kernel_launches+20);
            if (pass>=2) times[pass-2]=ms;
        }
        printf("{\"tensor\":\"%s\",\"type\":%u,\"outputs\":%zu,\"inputs\":%zu,\"tokens\":%zu,\"operation\":%u,\"ms\":[",w->name,w->type,o,i,n,op);
        for (size_t k=0;k<5;++k) printf("%s%.9f",k?",":"",times[k]);
        printf("]}\n"); fflush(stdout);
    }
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
int main(int argc,char **argv)
{
    nya_train_device *d=NULL;
    REQUIRE(argc==3);
    if (!strcmp(argv[1],"--startup")) return startup(argv[2]);
    FILE *f=fopen(argv[1],"rb"); REQUIRE(f);
    REQUIRE(!fseek(f,0,SEEK_END)); long size=ftell(f); fclose(f); REQUIRE(size>0);
    nya_llm_context *m=NULL; char error[256];
    if (nya_llm_load(argv[1],(uint64_t)size,&m,error,sizeof(error))) { fprintf(stderr,"%s\n",error); return 1; }
    const nya_llm_tensor *weights[]={m->layers[0].query,m->layers[0].feed_forward_down,m->layers[0].feed_forward_up};
    /* Six jobs each own their buffers; bounded arena includes all full F32 dW. */
    d=nya_train_device_create("cuda",256*1024*1024); REQUIRE(d);
    int reverse=!strcmp(argv[2],"reverse");
    for (size_t j=0;j<6;++j) {
        size_t job=reverse?5-j:j;
        REQUIRE(!run(d,weights[job/2],job%2?64:8));
    }
    nya_train_device_stats stats; nya_train_device_get_stats(d,&stats);
    fprintf(stderr,"arena_capacity=%zu arena_used=%zu buffers=%zu launches=%llu uploads=%llu upload_bytes=%llu downloads=%llu download_bytes=%llu\n",
        stats.capacity_bytes,stats.used_bytes,stats.buffers,(unsigned long long)stats.kernel_launches,(unsigned long long)stats.uploads,
        (unsigned long long)stats.upload_bytes,(unsigned long long)stats.downloads,(unsigned long long)stats.download_bytes);
    nya_train_device_free(d); nya_llm_free(m); return 0;
}
