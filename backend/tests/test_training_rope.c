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
static int rope_case(nya_train_device *d,size_t rows,size_t heads,size_t dim,int split,int extreme)
{
    size_t cols=heads*dim,n=rows*cols;
    float *x=malloc(n*4),*seed=malloc(n*4),*freq=malloc(dim/2*4),*actual=malloc((n+1)*4);
    CHECK(x && seed && freq && actual);
    for (size_t i=0;i<n;++i) { x[i]=(float)((int)(i%43)-21)/16; seed[i]=(float)((int)(i%17)-8)/16; }
    for (size_t j=0;j<dim/2;++j) freq[j]=j%3 ? (float)pow(10000,-2.0*(double)j/(double)dim) : 0;
    if (extreme) for (size_t j=0;j<dim/2;++j) {
        const float values[]={FLT_MAX,-FLT_MAX,FLT_TRUE_MIN,-FLT_TRUE_MIN,0,-0.0f,-1,1};
        freq[j]=values[j%8];
    }
    nya_train_parameter *p=nya_train_parameter_create(rows,cols,x);
    nya_train_graph *g=nya_train_graph_create(64*1024*1024); CHECK(p && g);
    nya_train_tensor *y=nya_train_rope(nya_train_leaf(g,p),heads,dim,freq,split); CHECK(y);
    CHECK(!nya_train_backward(nya_train_linear(nya_train_reshape(y,1,n),nya_train_input(g,1,n,seed))));
    nya_train_scope scope=nya_train_device_scratch_begin(d); CHECK(scope);
    nya_train_buffer bx=nya_train_device_alloc(d,n*4),bf=nya_train_device_alloc(d,dim/2*4),bs=nya_train_device_alloc(d,n*4);
    nya_train_buffer by=nya_train_device_alloc(d,(n+1)*4),dx=nya_train_device_alloc(d,(n+1)*4); CHECK(bx && bf && bs && by && dx);
    CHECK(!nya_train_device_write(d,bx,0,x,n*4) && !nya_train_device_write(d,bf,0,freq,dim/2*4) && !nya_train_device_write(d,bs,0,seed,n*4));
    for (size_t i=0;i<n;++i) actual[i]=0.125f;
    actual[n]=12345;
    CHECK(!nya_train_device_write(d,dx,0,actual,(n+1)*4) && !nya_train_device_write(d,by,n*4,actual+n,4));
    nya_train_device_stats before,after; nya_train_device_get_stats(d,&before);
    CHECK(!nya_train_device_rope(d,by,(nya_train_view){bx,rows,cols},heads,dim,bf,split));
    for (int repeat=0;repeat<2;++repeat) CHECK(!nya_train_device_rope_backward(d,dx,(nya_train_view){bs,rows,cols},heads,dim,bf,split));
    nya_train_device_get_stats(d,&after);
    CHECK(after.kernel_launches==before.kernel_launches+3 && after.used_bytes==before.used_bytes &&
        after.uploads==before.uploads && after.downloads==before.downloads && after.synchronizations==before.synchronizations);
    CHECK(!nya_train_device_read(d,by,0,actual,(n+1)*4) && actual[n]==12345);
    for (size_t i=0;i<n;++i) {
        if (!close_value(nya_train_data(y)[i],actual[i])) {
            fprintf(stderr,"rows=%zu heads=%zu dim=%zu split=%d extreme=%d index=%zu\n",rows,heads,dim,split,extreme,i);
            return 1;
        }
    }
    CHECK(!nya_train_device_read(d,dx,0,actual,(n+1)*4) && actual[n]==12345);
    for (size_t i=0;i<n;++i) {
        float expected=0.125f; expected+=nya_train_parameter_gradient(p)[i]; expected+=nya_train_parameter_gradient(p)[i];
        CHECK(close_value(expected,actual[i]));
    }
    nya_train_graph_free(g); nya_train_parameter_free(p); free(x); free(seed); free(freq); free(actual);
    CHECK(!nya_train_device_scratch_end(d,scope)); return 0;
}
static double objective(const double *x,const float *seed,const float *freq,int split)
{
    double loss=0;
    for (size_t row=0;row<3;++row) for (size_t head=0;head<2;++head) for (size_t j=0;j<3;++j) {
        size_t i=row*12+head*6+(split?j:2*j),k=i+(split?3:1);
        double c=cos((double)row*freq[j]),s=sin((double)row*freq[j]);
        loss+=seed[i]*(x[i]*c-x[k]*s)+seed[k]*(x[k]*c+x[i]*s);
    }
    return loss;
}
static int differences(nya_train_device *d,int split)
{
    double x[36]; float seed[36],actual[36],freq[3]={1,0,-0.37f};
    for (size_t i=0;i<36;++i) { x[i]=((double)i-17)/16; seed[i]=(float)((int)(i%11)-5)/8; }
    nya_train_scope scope=nya_train_device_scratch_begin(d); CHECK(scope);
    nya_train_buffer bs=nya_train_device_alloc(d,sizeof(seed)),bf=nya_train_device_alloc(d,sizeof(freq)),dx=nya_train_device_alloc(d,sizeof(actual));
    CHECK(bs && bf && dx && !nya_train_device_write(d,bs,0,seed,sizeof(seed)) && !nya_train_device_write(d,bf,0,freq,sizeof(freq)));
    CHECK(!nya_train_device_rope_backward(d,dx,(nya_train_view){bs,3,12},2,6,bf,split) && !nya_train_device_read(d,dx,0,actual,sizeof(actual)));
    for (size_t i=0;i<36;++i) {
        double old=x[i],h=1e-5; x[i]=old+h; double plus=objective(x,seed,freq,split);
        x[i]=old-h; double minus=objective(x,seed,freq,split); x[i]=old;
        CHECK(close_value((plus-minus)/(2*h),actual[i]));
    }
    CHECK(!nya_train_device_scratch_end(d,scope)); return 0;
}
static int descriptors(nya_train_device *d)
{
    nya_train_scope scope=nya_train_device_scratch_begin(d); CHECK(scope);
    nya_train_buffer stale=nya_train_device_alloc(d,32); CHECK(stale && !nya_train_device_scratch_end(d,scope));
    scope=nya_train_device_scratch_begin(d); CHECK(scope);
    nya_train_buffer x=nya_train_device_alloc(d,32),f=nya_train_device_alloc(d,8),y=nya_train_device_alloc(d,32),tiny=nya_train_device_alloc(d,1);
    uint32_t id=0; nya_train_indices map=nya_train_device_indices(d,&id,1,1);
    nya_train_device *other=nya_train_device_create("cuda",1024); CHECK(other);
    nya_train_buffer foreign=nya_train_device_alloc(other,32); CHECK(x && f && y && tiny && map && foreign);
    float sentinel[8]; for (size_t i=0;i<8;++i) sentinel[i]=42;
    CHECK(!nya_train_device_write(d,y,0,sentinel,32));
    nya_train_device_stats before,after; nya_train_device_get_stats(d,&before);
    const nya_train_buffer invalid[]={0,stale,foreign,tiny,map};
    for (size_t k=0;k<sizeof(invalid)/sizeof(invalid[0]);++k) {
        CHECK(nya_train_device_rope(d,invalid[k],(nya_train_view){x,2,4},1,4,f,0));
        CHECK(nya_train_device_rope(d,y,(nya_train_view){invalid[k],2,4},1,4,f,0));
        CHECK(nya_train_device_rope(d,y,(nya_train_view){x,2,4},1,4,invalid[k],0));
        CHECK(nya_train_device_rope_backward(d,invalid[k],(nya_train_view){x,2,4},1,4,f,1));
        CHECK(nya_train_device_rope_backward(d,y,(nya_train_view){invalid[k],2,4},1,4,f,1));
        CHECK(nya_train_device_rope_backward(d,y,(nya_train_view){x,2,4},1,4,invalid[k],1));
    }
    const size_t shapes[][4]={{0,4,1,4},{2,0,1,4},{SIZE_MAX,4,1,4},{2,SIZE_MAX,1,4},{3,4,1,4},
        {2,4,0,4},{2,4,1,0},{2,4,1,3},{2,4,2,4},{2,4,SIZE_MAX,4},{2,4,2,SIZE_MAX-1}};
    for (size_t j=0;j<sizeof(shapes)/sizeof(shapes[0]);++j) {
        CHECK(nya_train_device_rope(d,y,(nya_train_view){x,shapes[j][0],shapes[j][1]},shapes[j][2],shapes[j][3],f,0));
        CHECK(nya_train_device_rope_backward(d,y,(nya_train_view){x,shapes[j][0],shapes[j][1]},shapes[j][2],shapes[j][3],f,0));
    }
    CHECK(nya_train_device_rope(d,x,(nya_train_view){x,2,4},1,4,f,0));
    CHECK(nya_train_device_rope_backward(d,x,(nya_train_view){x,2,4},1,4,f,0));
    CHECK(nya_train_device_rope(d,y,(nya_train_view){x,2,4},1,4,y,0));
    CHECK(nya_train_device_rope_backward(d,y,(nya_train_view){x,2,4},1,4,y,0));
    CHECK(nya_train_device_rope(d,y,(nya_train_view){x,2,4},1,4,f,-1));
    CHECK(nya_train_device_rope_backward(d,y,(nya_train_view){x,2,4},1,4,f,2));
    nya_train_device_get_stats(d,&after);
    CHECK(!after.failed && after.kernel_launches==before.kernel_launches && after.synchronizations==before.synchronizations &&
        after.uploads==before.uploads && after.downloads==before.downloads && after.used_bytes==before.used_bytes);
    CHECK(!nya_train_device_read(d,y,0,sentinel,32));
    for (size_t i=0;i<8;++i) CHECK(sentinel[i]==42);
    /* Read-only frequency/source alias is deliberately valid. */
    CHECK(!nya_train_device_rope(d,y,(nya_train_view){x,2,4},1,4,x,0));
    CHECK(!nya_train_device_rope_backward(d,y,(nya_train_view){x,2,4},1,4,x,0));
    nya_train_device_free(other); CHECK(!nya_train_device_scratch_end(d,scope)); return 0;
}
static int residency_and_status(nya_train_device *d)
{
    float seed[4]={1,2,3,4},frequency=0.25f,actual[4],expected[4]={0};
    nya_train_buffer f=nya_train_device_alloc(d,4),x=nya_train_device_alloc(d,16),dx=nya_train_device_alloc(d,16),status=nya_train_device_alloc(d,4);
    CHECK(f && x && dx && status && !nya_train_device_write(d,f,0,&frequency,4) && !nya_train_device_write(d,x,0,seed,16));
    nya_train_device_stats before,after; nya_train_device_get_stats(d,&before);
    for (int repeat=0;repeat<100;++repeat) {
        nya_train_scope scope=nya_train_device_scratch_begin(d); nya_train_buffer y=nya_train_device_alloc(d,16); CHECK(scope && y);
        CHECK(!nya_train_device_rope(d,y,(nya_train_view){x,2,2},1,2,f,0));
        CHECK(!nya_train_device_rope_backward(d,dx,(nya_train_view){x,2,2},1,2,f,0));
        CHECK(!nya_train_device_scratch_end(d,scope));
        expected[0]+=1; expected[1]+=2;
        expected[2]+=(float)(3*cos(0.25)+4*sin(0.25)); expected[3]+=(float)(4*cos(0.25)-3*sin(0.25));
    }
    nya_train_device_get_stats(d,&after);
    CHECK(after.kernel_launches==before.kernel_launches+300 && after.scratch_resets==before.scratch_resets+100 &&
        after.used_bytes==before.used_bytes && after.buffers==before.buffers && after.uploads==before.uploads &&
        after.downloads==before.downloads && after.synchronizations==before.synchronizations);
    CHECK(!nya_train_device_read(d,dx,0,actual,16));
    for (size_t i=0;i<4;++i) CHECK(close_value(expected[i],actual[i]));
    const float bad[]={NAN,INFINITY,-INFINITY};
    for (size_t j=0;j<3;++j) {
        CHECK(!nya_train_device_write(d,f,0,bad+j,4) && !nya_train_device_zero(d,status));
        CHECK(!nya_train_device_check_finite(d,status,f,1,17));
        CHECK(!nya_train_device_rope(d,dx,(nya_train_view){x,2,2},1,2,f,0));
        CHECK(!nya_train_device_check_finite(d,status,dx,4,19));
        uint32_t tag=0; CHECK(!nya_train_device_read(d,status,0,&tag,4) && tag==17);
        nya_train_device_get_stats(d,&after); CHECK(!after.failed);
    }
    CHECK(!nya_train_device_write(d,f,0,&frequency,4) && !nya_train_device_zero(d,status));
    CHECK(!nya_train_device_rope(d,dx,(nya_train_view){x,2,2},1,2,f,0));
    CHECK(!nya_train_device_check_finite(d,status,dx,4,19));
    uint32_t tag=99; CHECK(!nya_train_device_read(d,status,0,&tag,4) && !tag);
    return 0;
}
/* cos/sin(FLT_MAX), independently evaluated with mpmath at 100 digits.
   This runs in CPU-only configurations too; do not use host libm as its oracle. */
static int cpu_large_angle(nya_train_device *d)
{
    const float x[8]={1,0,0,1,1,0,0,1},freq[2]={FLT_MAX,FLT_MAX};
    const double c=0.8530210398303041580517914676921611,s=-0.5218765233336585405515053570198067;
    for (int split=0;split<2;++split) {
        nya_train_parameter *p=nya_train_parameter_create(2,4,x);
        nya_train_graph *g=nya_train_graph_create(65536); CHECK(p && g);
        nya_train_tensor *y=nya_train_rope(nya_train_leaf(g,p),1,4,freq,split); CHECK(y);
        const float *out=nya_train_data(y);
        if (split) CHECK(close_value(c,out[4]) && close_value(-s,out[5]) && close_value(s,out[6]) && close_value(c,out[7]));
        else CHECK(close_value(c,out[4]) && close_value(s,out[5]) && close_value(-s,out[6]) && close_value(c,out[7]));
        CHECK(!nya_train_backward(nya_train_linear(nya_train_reshape(y,1,8),nya_train_input(g,1,8,x))));
        const float *dx=nya_train_parameter_gradient(p);
        if (split) CHECK(close_value(c,dx[4]) && close_value(s,dx[5]) && close_value(-s,dx[6]) && close_value(c,dx[7]));
        else CHECK(close_value(c,dx[4]) && close_value(-s,dx[5]) && close_value(s,dx[6]) && close_value(c,dx[7]));
        nya_train_graph_free(g); nya_train_parameter_free(p);
    }
    return 0;
}
int main(int argc,char **argv)
{
    nya_train_device *d=NULL;
    CHECK(nya_train_device_rope(NULL,0,(nya_train_view){0},0,0,0,0));
    CHECK(nya_train_device_rope_backward(NULL,0,(nya_train_view){0},0,0,0,0));
    CHECK(!cpu_large_angle(d));
    if (argc==1) return 0;
    d=nya_train_device_create("cuda",16*1024*1024); if (!d) return 77;
    if (!strcmp(argv[1],"--failure")) {
        nya_train_buffer x=nya_train_device_alloc(d,8),f=nya_train_device_alloc(d,4),y=nya_train_device_alloc(d,8); CHECK(x && f && y);
        int result=nya_train_device_rope(d,y,(nya_train_view){x,1,2},1,2,f,0);
        if (!result) result=nya_train_device_rope_backward(d,y,(nya_train_view){x,1,2},1,2,f,0);
        CHECK(result); nya_train_device_stats stats; nya_train_device_get_stats(d,&stats); CHECK(stats.failed);
        CHECK(stats.kernel_launches==strtoull(getenv("NYA_TRAIN_DEVICE_FAIL_LAUNCH_AFTER"),NULL,10));
        float value=123; CHECK(nya_train_device_read(d,y,0,&value,4) && value==123 && nya_train_device_finish(d));
        CHECK(nya_train_device_rope(d,y,(nya_train_view){x,1,2},1,2,f,0));
        nya_train_device_free(d); return 0;
    }
    const size_t widths[]={2,6,64,128,256,514,1024},rows[]={1,3,65},heads[]={1,3};
    for (int split=0;split<2;++split) {
        for (size_t r=0;r<3;++r) for (size_t h=0;h<2;++h) for (size_t j=0;j<7;++j)
            CHECK(!rope_case(d,rows[r],heads[h],widths[j],split,0));
        CHECK(!rope_case(d,8193,2,8,split,0) && !rope_case(d,65,3,16,split,1));
        CHECK(!differences(d,split));
    }
    CHECK(!descriptors(d) && !residency_and_status(d));
    nya_train_device_free(d);
    puts("resident RoPE: CPU autograd, finite differences, layouts, long positions, extreme frequencies, descriptors, residency and finite status passed");
    return 0;
}
