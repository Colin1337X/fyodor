#include "compute.h"
#include "vendor.h"
#include "thread.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define CHECK(x) do { if (!(x)) { fprintf(stderr,"vendor check line %d: %s\n",__LINE__,#x); return 1; } } while (0)
typedef struct worker_job { nya_compute_context *context; int result; } worker_job;
static int worker(void *argument)
{
    worker_job *job=argument;
    /* Static immutable weights outlive both workers and the cache. */
    static const float w[]={1,2,3,4},x[]={2,3}; float y[2];
    job->result=nya_compute_matvec(job->context,w,2,2,x,y) || y[0]!=8 || y[1]!=18;
    return job->result;
}

int main(int argc,char **argv)
{
    if (argc==2 && !strcmp(argv[1],"--unavailable")) {
        nya_compute_context *c=nya_compute_create();
        CHECK(!c && !strcmp(nya_compute_name(c),"cpu")); return 0;
    }
    CHECK(argc==3);
    const char *provider=argv[1];
    void *library=nya_vendor_open(argv[2],NULL); CHECK(library);
    void (*inject)(int); int (*live)(void);
    CHECK(!nya_vendor_symbol(library,"nya_mock_fail_after",&inject,sizeof(inject)));
    CHECK(!nya_vendor_symbol(library,"nya_mock_live",&live,sizeof(live)));
    CHECK(nya_compute_backend_known(provider) && nya_compute_backend_compiled(provider));
    CHECK(!nya_compute_backend_known("unknown") && !nya_compute_backend_compiled("unknown"));
    float w[]={1,2,3,4,5,6},x[]={1,2,3,4,5,6},y[6];
    /* Fail every initialization step, then every execution step. Always retain
       a separate loader reference so live ownership remains observable after
       the provider closes its own library. No mocks enter performance runs. */
    for (int n=1;n<=40;++n) {
        inject(n);
        nya_compute_context *c=nya_compute_create_for(provider);
        inject(0); nya_compute_free(c); CHECK(live()==0);
    }
    for (int n=0;n<=22;++n) {
        inject(0); nya_compute_context *c=nya_compute_create_for(provider); CHECK(c);
        CHECK(nya_compute_matvec(c,w,SIZE_MAX,3,x,y)==-1);
        CHECK(nya_compute_matmul_typed(c,w,2,3,99,x,y,2)==-1);
        CHECK(!strcmp(nya_compute_name(c),provider));
        y[0]=y[5]=12345; inject(n);
        int status=nya_compute_matmul_typed(c,w,2,3,0,x,y+1,2);
        inject(0);
        if (status) CHECK(!strcmp(nya_compute_name(c),"cpu"));
        else { CHECK(y[1]==14 && y[2]==32 && y[3]==32 && y[4]==77); }
        CHECK(y[0]==12345 && y[5]==12345);
        nya_compute_free(c); CHECK(live()==0);
    }
    /* Two independent contexts exercise MLX module reference counting. Freeing
       either must not invalidate the other's cached arrays or function table. */
    nya_compute_context *a=nya_compute_create_for(provider),*b=nya_compute_create_for(provider); CHECK(a && b);
    CHECK(!nya_compute_matvec(a,w,2,3,x,y));
    nya_compute_stats before,after; nya_compute_context_stats(a,&before);
    CHECK(!nya_compute_matvec(a,w,2,3,x,y)); nya_compute_context_stats(a,&after);
    CHECK(after.uploads==before.uploads+1 && after.weights_bytes==before.weights_bytes);
    nya_compute_free(a); CHECK(!nya_compute_matvec(b,w,2,3,x,y));
    for (int i=0;i<2;++i) {
        nya_thread thread={0}; worker_job job={b,-1};
        CHECK(!nya_thread_create(&thread,worker,&job)); CHECK(!nya_thread_join(&thread)); CHECK(!job.result);
    }
    nya_compute_free(b); CHECK(live()==0);
    /* The test environment sets a 1 MiB budget: two individually fitting
       weights force LRU eviction, then a single oversized matrix fails safely. */
    size_t rows=128,cols=1024,n=rows*cols;
    float *large=calloc(n*3,sizeof(float)),*input=calloc(cols,sizeof(float)),*output=calloc(rows*3,sizeof(float));
    CHECK(large && input && output); a=nya_compute_create_for(provider); CHECK(a);
    CHECK(!nya_compute_matvec(a,large,rows,cols,input,output));
    CHECK(!nya_compute_matvec(a,large+n,rows,cols,input,output));
    CHECK(!nya_compute_matvec(a,large,rows,cols,input,output));
    nya_compute_context_stats(a,&after); CHECK(after.weights_bytes<=1024*1024);
    CHECK(nya_compute_matvec(a,large,rows*3,cols,input,output)==-1);
    CHECK(!strcmp(nya_compute_name(a),"cpu")); nya_compute_free(a); CHECK(live()==0);
    /* A synchronization/device-selection failure during destruction must not
       bypass release of independent handles and cache ownership records. */
    a=nya_compute_create_for(provider); CHECK(a); inject(1); nya_compute_free(a); inject(0); CHECK(live()==0);
    free(large); free(input); free(output); nya_vendor_close(library);
    puts("vendor ABI, numerical layout, cache, fallback and ownership checks passed (mock runtime)"); return 0;
}
