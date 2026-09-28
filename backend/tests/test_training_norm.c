#include "training_device.h"
#include "training.h"
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(x) do { if (!(x)) { fprintf(stderr,"line %d: %s (%s)\n",__LINE__,#x,nya_train_device_error(d)); return 1; } } while (0)
static int close_value(double expected, float actual)
{
    if (isfinite(actual) && fabs(expected-actual)<=1e-6*(1+fabs(expected))) return 1;
    fprintf(stderr,"expected %.12g actual %.9g\n",expected,(double)actual); return 0;
}
static int norm_case(nya_train_device *d,size_t rows,size_t cols,int weighted,int extremes)
{
    size_t n=rows*cols;
    float *x=malloc(n*4), *w=malloc(cols*4), *seed=malloc(n*4), *actual=malloc((n+1)*4);
    CHECK(x && w && seed && actual);
    for (size_t i=0;i<n;++i) { x[i]=(float)((int)(i%43)-21)/16; seed[i]=(float)((int)(i%17)-8)/16; }
    for (size_t j=0;j<cols;++j) w[j]=(float)((int)(j%11)-5)/8;
    float eps=1e-5f;
    if (extremes) {
        eps=FLT_TRUE_MIN;
        for (size_t i=0;i<n;++i) x[i]=i/cols==0 ? ((i%2)?FLT_MAX:-FLT_MAX) : i/cols==1 ? 0 : FLT_TRUE_MIN;
    }
    nya_train_parameter *px=nya_train_parameter_create(rows,cols,x), *pw=nya_train_parameter_create(1,cols,w);
    nya_train_graph *g=nya_train_graph_create(32*1024*1024); CHECK(px && pw && g);
    nya_train_tensor *y=nya_train_rms_norm(nya_train_leaf(g,px),weighted?nya_train_leaf(g,pw):NULL,eps); CHECK(y);
    CHECK(!nya_train_backward(nya_train_linear(nya_train_reshape(y,1,n),nya_train_input(g,1,n,seed))));
    nya_train_scope scope=nya_train_device_scratch_begin(d); CHECK(scope);
    nya_train_buffer bx=nya_train_device_alloc(d,n*4), bw=nya_train_device_alloc(d,cols*4), bs=nya_train_device_alloc(d,n*4);
    nya_train_buffer by=nya_train_device_alloc(d,(n+1)*4), dx=nya_train_device_alloc(d,(n+1)*4), dw=nya_train_device_alloc(d,(cols+1)*4);
    nya_train_buffer inv=nya_train_device_alloc(d,(rows+1)*8); CHECK(bx && bw && bs && by && dx && dw && inv);
    CHECK(!nya_train_device_write(d,bx,0,x,n*4) && !nya_train_device_write(d,bw,0,w,cols*4) && !nya_train_device_write(d,bs,0,seed,n*4));
    for (size_t i=0;i<n;++i) actual[i]=0.125f;
    actual[n]=12345;
    CHECK(!nya_train_device_write(d,dx,0,actual,(n+1)*4));
    actual[cols]=12345;
    CHECK(!nya_train_device_write(d,dw,0,actual,(cols+1)*4));
    float guard=12345; double guard64=98765;
    CHECK(!nya_train_device_write(d,by,n*4,&guard,4) && !nya_train_device_write(d,inv,rows*8,&guard64,8));
    nya_train_view view={bx,rows,cols};
    nya_train_device_stats before,after; nya_train_device_get_stats(d,&before);
    CHECK(!nya_train_device_rms_norm(d,by,inv,view,weighted?bw:0,eps));
    CHECK(!nya_train_device_rms_norm_backward(d,dx,weighted?dw:0,inv,view,weighted?bw:0,bs));
    CHECK(!nya_train_device_rms_norm_backward(d,dx,0,inv,view,weighted?bw:0,bs));
    if (weighted) CHECK(!nya_train_device_rms_norm_backward(d,0,dw,inv,view,bw,bs));
    nya_train_device_get_stats(d,&after);
    CHECK(after.kernel_launches==before.kernel_launches+(weighted?5u:3u) && after.used_bytes==before.used_bytes &&
        after.uploads==before.uploads && after.downloads==before.downloads && after.synchronizations==before.synchronizations);
    CHECK(!nya_train_device_read(d,by,0,actual,(n+1)*4) && actual[n]==guard);
    for (size_t i=0;i<n;++i) CHECK(close_value(nya_train_data(y)[i],actual[i]));
    CHECK(!nya_train_device_read(d,dx,0,actual,(n+1)*4) && actual[n]==guard);
    for (size_t i=0;i<n;++i) {
        float expected=0.125f;
        expected+=nya_train_parameter_gradient(px)[i]; expected+=nya_train_parameter_gradient(px)[i];
        CHECK(close_value(expected,actual[i]));
    }
    CHECK(!nya_train_device_read(d,dw,0,actual,(cols+1)*4) && actual[cols]==guard);
    /* CPU weight gradients sum rounded contributions across rows. Replay two
       full backwards into the CPU parameter to test the same accumulation. */
    nya_train_graph_free(g);
    g=nya_train_graph_create(32*1024*1024); CHECK(g);
    y=nya_train_rms_norm(nya_train_leaf(g,px),weighted?nya_train_leaf(g,pw):NULL,eps); CHECK(y);
    CHECK(!nya_train_backward(nya_train_linear(nya_train_reshape(y,1,n),nya_train_input(g,1,n,seed))));
    for (size_t j=0;j<cols;++j) CHECK(close_value(0.125+nya_train_parameter_gradient(pw)[j],actual[j]));
    CHECK(!nya_train_device_read(d,inv,rows*8,&guard64,8) && guard64==98765);
    nya_train_graph_free(g); nya_train_parameter_free(px); nya_train_parameter_free(pw);
    free(x); free(w); free(seed); free(actual);
    CHECK(!nya_train_device_scratch_end(d,scope)); return 0;
}
static double objective(const double *x,const double *w,const float *seed)
{
    double loss=0;
    for (size_t row=0;row<3;++row) {
        double square=0;
        for (size_t j=0;j<5;++j) square+=x[row*5+j]*x[row*5+j];
        double inverse=1/sqrt(square/5+0.03125);
        for (size_t j=0;j<5;++j) loss+=seed[row*5+j]*x[row*5+j]*inverse*w[j];
    }
    return loss;
}
static int finite_differences(nya_train_device *d)
{
    double x[15],w[5]; float xf[15],wf[5],seed[15],actual[15];
    for (size_t i=0;i<15;++i) xf[i]=(float)(x[i]=((double)i-7)/8),seed[i]=(float)((int)(i%7)-3)/8;
    for (size_t j=0;j<5;++j) wf[j]=(float)(w[j]=((double)j-2)/4);
    nya_train_scope scope=nya_train_device_scratch_begin(d); CHECK(scope);
    nya_train_buffer bx=nya_train_device_alloc(d,sizeof(xf)), bw=nya_train_device_alloc(d,sizeof(wf));
    nya_train_buffer dy=nya_train_device_alloc(d,sizeof(seed)), by=nya_train_device_alloc(d,sizeof(xf));
    nya_train_buffer dx=nya_train_device_alloc(d,sizeof(xf)), dw=nya_train_device_alloc(d,sizeof(wf)), inv=nya_train_device_alloc(d,24);
    CHECK(bx && bw && dy && by && dx && dw && inv);
    CHECK(!nya_train_device_write(d,bx,0,xf,sizeof(xf)) && !nya_train_device_write(d,bw,0,wf,sizeof(wf)) && !nya_train_device_write(d,dy,0,seed,sizeof(seed)));
    CHECK(!nya_train_device_rms_norm(d,by,inv,(nya_train_view){bx,3,5},bw,0.03125f));
    CHECK(!nya_train_device_rms_norm_backward(d,dx,dw,inv,(nya_train_view){bx,3,5},bw,dy));
    for (unsigned side=0;side<2;++side) {
        size_t count=side?5u:15u; double *p=side?w:x;
        CHECK(!nya_train_device_read(d,side?dw:dx,0,actual,count*4));
        for (size_t i=0;i<count;++i) {
            double old=p[i], h=1e-5; p[i]=old+h; double plus=objective(x,w,seed);
            p[i]=old-h; double minus=objective(x,w,seed); p[i]=old;
            CHECK(close_value((plus-minus)/(2*h),actual[i]));
        }
    }
    CHECK(!nya_train_device_scratch_end(d,scope)); return 0;
}
static int shared_gradient(nya_train_device *d)
{
    float x[5]={0.25f,-0.5f,0.75f,1,-1.25f}, seed[5]={1,-0.25f,0.125f,-1,0.5f}, actual[5];
    nya_train_parameter *p=nya_train_parameter_create(1,5,x); nya_train_graph *g=nya_train_graph_create(65536); CHECK(p && g);
    nya_train_tensor *leaf=nya_train_leaf(g,p), *y=nya_train_rms_norm(leaf,leaf,0.0625f); CHECK(y);
    CHECK(!nya_train_backward(nya_train_linear(y,nya_train_input(g,1,5,seed))));
    nya_train_scope scope=nya_train_device_scratch_begin(d); CHECK(scope);
    nya_train_buffer bx=nya_train_device_alloc(d,20), dy=nya_train_device_alloc(d,20), by=nya_train_device_alloc(d,20);
    nya_train_buffer dx=nya_train_device_alloc(d,20), inv=nya_train_device_alloc(d,8); CHECK(bx && dy && by && dx && inv);
    CHECK(!nya_train_device_write(d,bx,0,x,20) && !nya_train_device_write(d,dy,0,seed,20));
    CHECK(!nya_train_device_rms_norm(d,by,inv,(nya_train_view){bx,1,5},bx,0.0625f));
    CHECK(!nya_train_device_rms_norm_backward(d,dx,dx,inv,(nya_train_view){bx,1,5},bx,dy));
    CHECK(!nya_train_device_read(d,dx,0,actual,20));
    for (size_t i=0;i<5;++i) CHECK(close_value(nya_train_parameter_gradient(p)[i],actual[i]));
    nya_train_graph_free(g); nya_train_parameter_free(p);
    CHECK(!nya_train_device_scratch_end(d,scope)); return 0;
}
static int descriptors(nya_train_device *d)
{
    nya_train_scope scope=nya_train_device_scratch_begin(d); CHECK(scope);
    nya_train_buffer stale=nya_train_device_alloc(d,32); CHECK(stale && !nya_train_device_scratch_end(d,scope));
    scope=nya_train_device_scratch_begin(d); CHECK(scope);
    nya_train_buffer x=nya_train_device_alloc(d,32), w=nya_train_device_alloc(d,16), dy=nya_train_device_alloc(d,32);
    nya_train_buffer out=nya_train_device_alloc(d,32), dw=nya_train_device_alloc(d,16), inv=nya_train_device_alloc(d,16), tiny=nya_train_device_alloc(d,1);
    CHECK(x && w && dy && out && dw && inv && tiny);
    nya_train_device *other=nya_train_device_create("cuda",1024); CHECK(other);
    nya_train_buffer foreign=nya_train_device_alloc(other,32); CHECK(foreign);
    float sentinels[8]; for (size_t i=0;i<8;++i) sentinels[i]=42;
    CHECK(!nya_train_device_write(d,out,0,sentinels,32));
    nya_train_device_stats before,after; nya_train_device_get_stats(d,&before);
    nya_train_view view={x,2,4};
    CHECK(nya_train_device_rms_norm(d,out,inv,view,w,0));
    CHECK(nya_train_device_rms_norm(d,out,inv,view,w,-1));
    CHECK(nya_train_device_rms_norm(d,out,inv,view,w,NAN));
    CHECK(nya_train_device_rms_norm(d,out,inv,view,w,INFINITY));
    const nya_train_buffer invalid[]={0,stale,foreign,tiny};
    for (size_t k=0;k<sizeof(invalid)/sizeof(invalid[0]);++k) {
        CHECK(nya_train_device_rms_norm(d,invalid[k],inv,view,w,1e-5f));
        CHECK(nya_train_device_rms_norm(d,out,invalid[k],view,w,1e-5f));
        CHECK(nya_train_device_rms_norm(d,out,inv,(nya_train_view){invalid[k],2,4},w,1e-5f));
        if (invalid[k]) CHECK(nya_train_device_rms_norm(d,out,inv,view,invalid[k],1e-5f));
        CHECK(nya_train_device_rms_norm_backward(d,out,dw,inv,view,w,invalid[k]));
    }
    CHECK(nya_train_device_rms_norm(d,out,inv,(nya_train_view){x,0,4},w,1e-5f));
    CHECK(nya_train_device_rms_norm(d,out,inv,(nya_train_view){x,SIZE_MAX,4},w,1e-5f));
    CHECK(nya_train_device_rms_norm(d,out,inv,(nya_train_view){x,2,SIZE_MAX},w,1e-5f));
    CHECK(nya_train_device_rms_norm(d,out,inv,(nya_train_view){x,3,4},w,1e-5f));
    const nya_train_buffer aliases[]={x,w,inv};
    for (size_t k=0;k<3;++k) {
        CHECK(nya_train_device_rms_norm(d,aliases[k],inv,view,w,1e-5f));
        CHECK(nya_train_device_rms_norm_backward(d,aliases[k],dw,inv,view,w,dy));
        CHECK(nya_train_device_rms_norm_backward(d,out,aliases[k],inv,view,w,dy));
    }
    CHECK(nya_train_device_rms_norm(d,out,x,view,w,1e-5f));
    CHECK(nya_train_device_rms_norm(d,out,w,view,w,1e-5f));
    CHECK(nya_train_device_rms_norm_backward(d,dy,dw,inv,view,w,dy));
    CHECK(nya_train_device_rms_norm_backward(d,out,dy,inv,view,w,dy));
    CHECK(nya_train_device_rms_norm_backward(d,out,dw,dy,view,w,dy));
    CHECK(nya_train_device_rms_norm_backward(d,out,out,inv,view,w,dy));
    CHECK(nya_train_device_rms_norm_backward(d,0,0,inv,view,w,dy));
    CHECK(nya_train_device_rms_norm_backward(d,out,dw,inv,view,0,dy));
    CHECK(nya_train_device_rms_norm_backward(d,out,tiny,inv,view,w,dy));
    nya_train_device_get_stats(d,&after);
    CHECK(!after.failed && after.kernel_launches==before.kernel_launches && after.uploads==before.uploads &&
        after.downloads==before.downloads && after.synchronizations==before.synchronizations);
    CHECK(!nya_train_device_read(d,out,0,sentinels,32));
    for (size_t i=0;i<8;++i) CHECK(sentinels[i]==42);
    nya_train_device_free(other); CHECK(!nya_train_device_scratch_end(d,scope)); return 0;
}
int main(int argc,char **argv)
{
    nya_train_device *d=NULL;
    CHECK(nya_train_device_rms_norm(NULL,0,0,(nya_train_view){0},0,1e-5f));
    CHECK(nya_train_device_rms_norm_backward(NULL,0,0,0,(nya_train_view){0},0,0));
    if (argc==1) return 0;
    d=nya_train_device_create("cuda",16*1024*1024); if (!d) return 77;
    if (!strcmp(argv[1],"--failure")) {
        nya_train_buffer x=nya_train_device_alloc(d,4), w=nya_train_device_alloc(d,4), dy=nya_train_device_alloc(d,4);
        nya_train_buffer y=nya_train_device_alloc(d,4), inv=nya_train_device_alloc(d,8), dx=nya_train_device_alloc(d,4), dw=nya_train_device_alloc(d,4);
        CHECK(x && w && dy && y && inv && dx && dw);
        int result=nya_train_device_rms_norm(d,y,inv,(nya_train_view){x,1,1},w,1e-5f);
        if (!result) result=nya_train_device_rms_norm_backward(d,dx,dw,inv,(nya_train_view){x,1,1},w,dy);
        CHECK(result);
        nya_train_device_stats stats; nya_train_device_get_stats(d,&stats); CHECK(stats.failed);
        CHECK(stats.kernel_launches==strtoull(getenv("NYA_TRAIN_DEVICE_FAIL_LAUNCH_AFTER"),NULL,10));
        float value=123; CHECK(nya_train_device_read(d,dx,0,&value,4) && value==123 && nya_train_device_finish(d));
        CHECK(nya_train_device_rms_norm(d,y,inv,(nya_train_view){x,1,1},w,1e-5f));
        nya_train_device_free(d); return 0;
    }
    const size_t widths[]={1,5,31,32,33,255,256,257,1025,2048};
    const size_t rows[]={1,3,33};
    for (size_t r=0;r<3;++r) for (size_t j=0;j<sizeof(widths)/sizeof(widths[0]);++j)
        for (int weighted=0;weighted<2;++weighted) CHECK(!norm_case(d,rows[r],widths[j],weighted,0));
    CHECK(!norm_case(d,3,257,0,1) && !norm_case(d,3,257,1,1));
    CHECK(!finite_differences(d) && !shared_gradient(d) && !descriptors(d));
    nya_train_device_free(d);
    printf("resident RMSNorm: CPU autograd, finite differences, optional/shared gradients, extremes and descriptors passed\n");
    return 0;
}
