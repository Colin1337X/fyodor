#include "training_device.h"
#include "llm_internal.h"
#include "training.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(x) do { if (!(x)) { fprintf(stderr,"line %d: %s (%s)\n",__LINE__,#x,nya_train_device_error(d)); return 1; } } while (0)
static int close_sum(double sum, double magnitude, float actual)
{
    if (isfinite(actual) && fabs(sum-actual) <= 1e-6*(1+magnitude)) return 1;
    fprintf(stderr,"oracle %.12g actual %.9g magnitude %.9g\n",sum,(double)actual,magnitude);
    return 0;
}
static int matrix(nya_train_device *d, unsigned type, size_t o, size_t i, size_t n)
{
    size_t block = type == 2 || type == 8 ? 32 : type == 12 || type == 14 ? 256 : 1;
    size_t unit = type == 0 ? 4 : type == 1 || type == 30 ? 2 : type == 2 ? 18 : type == 8 ? 34 : type == 12 ? 144 : 210;
    size_t bytes = o*(i/block)*unit, nx = n*i, ny = n*o, nw = o*i;
    unsigned char *w = malloc(bytes);
    float *x = malloc(nx*4), *dy = malloc(ny*4), *actual = malloc(((nw>nx?nw:nx)>ny?(nw>nx?nw:nx):ny)*4+4);
    CHECK(w && x && dy && actual);
    for (size_t k = 0; k < bytes; ++k) w[k] = (unsigned char)((k*37+13)%251);
    if (block == 1) for (size_t k = 0; k < nw; ++k) {
        float f = (float)((int)(k%71)-35)/64;
        if (!type) memcpy(w+k*4,&f,4);
        else {
            uint32_t bits; memcpy(&bits,&f,4);
            unsigned v = type == 30 ? bits>>16 : 0x3000u+(unsigned)(k%1024)+(k%2?0x8000u:0);
            w[k*2] = (unsigned char)v; w[k*2+1] = (unsigned char)(v>>8);
        }
    } else for (size_t k = 0; k < bytes; k += unit) {
        size_t s = type == 14 ? 208 : 0;
        w[k+s] = 0; w[k+s+1] = 0x28;
        if (type == 12) { w[k+2] = 0; w[k+3] = 0x24; }
    }
    for (size_t k = 0; k < nx; ++k) x[k] = (float)((int)(k%61)-30)/31;
    for (size_t k = 0; k < ny; ++k) dy[k] = (float)((int)(k%47)-23)/29;
    nya_train_buffer bw = nya_train_device_alloc(d,bytes), bx = nya_train_device_alloc(d,nx*4);
    nya_train_buffer by = nya_train_device_alloc(d,ny*4+4), bd = nya_train_device_alloc(d,ny*4);
    nya_train_buffer bdx = nya_train_device_alloc(d,nx*4+4), bdw = nya_train_device_alloc(d,nw*4+4);
    CHECK(bw && bx && by && bd && bdx && bdw);
    CHECK(!nya_train_device_write(d,bw,0,w,bytes) && !nya_train_device_write(d,bx,0,x,nx*4) && !nya_train_device_write(d,bd,0,dy,ny*4));
    float guard = 12345;
    CHECK(!nya_train_device_write(d,by,ny*4,&guard,4) && !nya_train_device_write(d,bdx,nx*4,&guard,4) && !nya_train_device_write(d,bdw,nw*4,&guard,4));
    nya_train_device_stats before, after;
    nya_train_device_get_stats(d,&before);
    CHECK(!nya_train_device_linear(d,by,bw,type,o,i,bx,n));
    for (unsigned pass = 0; pass < 2; ++pass) {
        CHECK(!nya_train_device_linear_dx(d,bdx,bw,type,o,i,bd,n));
        CHECK(!nya_train_device_linear_dw(d,bdw,bx,bd,o,i,n));
    }
    nya_train_device_get_stats(d,&after);
    CHECK(after.kernel_launches == before.kernel_launches+5 && after.uploads == before.uploads &&
        after.downloads == before.downloads && after.synchronizations == before.synchronizations && after.used_bytes == before.used_bytes);
    nya_llm_tensor tensor = {0}; tensor.data = w; tensor.type = type;
    CHECK(!nya_train_device_read(d,by,0,actual,ny*4+4) && actual[ny] == guard);
    for (size_t t = 0; t < n; ++t) for (size_t r = 0; r < o; ++r) {
        double sum=0, mag=0;
        for (size_t k=0; k<i; ++k) { double p=(double)nya_llm_tensor_value(&tensor,r*i+k)*x[t*i+k]; sum+=p; mag+=fabs(p); }
        CHECK(close_sum(sum,mag,actual[t*o+r]));
    }
    CHECK(!nya_train_device_read(d,bdx,0,actual,nx*4+4) && actual[nx] == guard);
    for (size_t t = 0; t < n; ++t) for (size_t k = 0; k < i; ++k) {
        double sum=0, mag=0;
        for (size_t r=0; r<o; ++r) { double p=2*(double)nya_llm_tensor_value(&tensor,r*i+k)*dy[t*o+r]; sum+=p; mag+=fabs(p); }
        CHECK(close_sum(sum,mag,actual[t*i+k]));
    }
    CHECK(!nya_train_device_read(d,bdw,0,actual,nw*4+4) && actual[nw] == guard);
    for (size_t r = 0; r < o; ++r) for (size_t k = 0; k < i; ++k) {
        double sum=0, mag=0;
        for (size_t t=0; t<n; ++t) { double p=2*(double)x[t*i+k]*dy[t*o+r]; sum+=p; mag+=fabs(p); }
        CHECK(close_sum(sum,mag,actual[r*i+k]));
    }
    CHECK(!nya_train_device_zero(d,bdw) && !nya_train_device_read(d,bdw,0,actual,nw*4+4));
    for (size_t k=0; k<=nw; ++k) CHECK(actual[k] == 0);
    free(w); free(x); free(dy); free(actual);
    printf("matrix type=%u outputs=%zu inputs=%zu tokens=%zu passed\n",type,o,i,n);
    return 0;
}
static int autograd(nya_train_device *d)
{
    float x[15], w[20], dy[12], actual[20];
    uint32_t labels[] = {1,2,0};
    for (size_t k=0; k<15; ++k) x[k]=(float)((int)k-7)/16;
    for (size_t k=0; k<20; ++k) w[k]=(float)((int)k-10)/32;
    nya_train_parameter *px=nya_train_parameter_create(3,5,x), *pw=nya_train_parameter_create(4,5,w);
    nya_train_graph *g=nya_train_graph_create(1024*1024);
    CHECK(px && pw && g);
    nya_train_tensor *y=nya_train_linear(nya_train_leaf(g,px),nya_train_leaf(g,pw));
    CHECK(y);
    const float *logits=nya_train_data(y);
    for (size_t t=0; t<3; ++t) {
        double sum=0;
        for (size_t r=0; r<4; ++r) sum+=exp(logits[t*4+r]);
        for (size_t r=0; r<4; ++r) dy[t*4+r]=(float)((exp(logits[t*4+r])/sum-(r==labels[t]))/3);
    }
    CHECK(!nya_train_backward(nya_train_cross_entropy(y,labels,NULL,3)));
    nya_train_buffer bx=nya_train_device_alloc(d,sizeof(x)), bw=nya_train_device_alloc(d,sizeof(w)), bd=nya_train_device_alloc(d,sizeof(dy));
    nya_train_buffer dx=nya_train_device_alloc(d,sizeof(x)), dw=nya_train_device_alloc(d,sizeof(w));
    CHECK(bx && bw && bd && dx && dw);
    CHECK(!nya_train_device_write(d,bx,0,x,sizeof(x)) && !nya_train_device_write(d,bw,0,w,sizeof(w)) && !nya_train_device_write(d,bd,0,dy,sizeof(dy)));
    CHECK(!nya_train_device_linear_dx(d,dx,bw,0,4,5,bd,3) && !nya_train_device_linear_dw(d,dw,bx,bd,4,5,3));
    CHECK(!nya_train_device_read(d,dx,0,actual,sizeof(x)));
    for (size_t k=0; k<15; ++k) CHECK(fabsf(actual[k]-nya_train_parameter_gradient(px)[k])<1e-6f);
    CHECK(!nya_train_device_read(d,dw,0,actual,sizeof(w)));
    for (size_t k=0; k<20; ++k) CHECK(fabsf(actual[k]-nya_train_parameter_gradient(pw)[k])<1e-6f);
    nya_train_graph_free(g); nya_train_parameter_free(px); nya_train_parameter_free(pw);
    return 0;
}
int main(int argc, char **argv)
{
    nya_train_device *d = NULL;
    CHECK(!nya_train_device_create(NULL,1) && !nya_train_device_create("absent",1) && !nya_train_device_create("cuda",0));
    CHECK(nya_train_device_finish(NULL) && !nya_train_device_alloc(NULL,4));
    if (argc == 1) return 0;
    d=nya_train_device_create("cuda",64*1024*1024);
    if (!d) { fprintf(stderr,"CUDA training storage unavailable\n"); return 77; }
    if (!strcmp(argv[1],"--failure")) {
        nya_train_buffer b=nya_train_device_alloc(d,4);
        CHECK(b && nya_train_device_zero(d,b));
        nya_train_device_stats s; nya_train_device_get_stats(d,&s);
        CHECK(s.failed && s.kernel_launches==1 && nya_train_device_finish(d) && !nya_train_device_alloc(d,4));
        nya_train_device_free(d); return 0;
    }
    if (!strcmp(argv[1],"--matrix-failure")) {
        nya_train_buffer b[6];
        for (size_t k=0;k<6;++k) { b[k]=nya_train_device_alloc(d,4); CHECK(b[k]); }
        CHECK(!nya_train_device_linear(d,b[0],b[1],0,1,1,b[2],1));
        CHECK(nya_train_device_linear_dx(d,b[3],b[1],0,1,1,b[4],1));
        CHECK(nya_train_device_linear_dw(d,b[5],b[2],b[4],1,1,1));
        float value;
        CHECK(nya_train_device_read(d,b[0],0,&value,4));
        nya_train_device_stats s; nya_train_device_get_stats(d,&s);
        CHECK(s.failed && s.kernel_launches==7 && s.downloads==0);
        nya_train_device_free(d); return 0;
    }
    nya_train_buffer small=nya_train_device_alloc(d,17);
    unsigned char bytes[17]; memset(bytes,123,sizeof(bytes));
    CHECK(small && !nya_train_device_read(d,small,0,bytes,sizeof(bytes)));
    for (size_t k=0; k<sizeof(bytes); ++k) CHECK(!bytes[k]);
    nya_train_device_stats before,after; nya_train_device_get_stats(d,&before);
    CHECK(!nya_train_device_alloc(d,SIZE_MAX) && !nya_train_device_alloc(d,0));
    CHECK(nya_train_device_read(d,small,SIZE_MAX,bytes,1) && nya_train_device_write(d,small,16,bytes,2));
    CHECK(nya_train_device_linear(d,small,small,0,1,1,small,1));
    CHECK(nya_train_device_linear_dx(d,small,UINT64_MAX,0,1,1,small,1));
    CHECK(nya_train_device_linear_dw(d,small,small,small,SIZE_MAX,1,1));
    nya_train_device_get_stats(d,&after);
    CHECK(!after.failed && before.used_bytes==after.used_bytes && before.kernel_launches==after.kernel_launches);
    nya_train_device *other=nya_train_device_create("cuda",1024);
    CHECK(other && nya_train_device_zero(other,small));
    nya_train_buffer stale=nya_train_device_alloc(other,4); CHECK(stale);
    nya_train_device_free(other);
    other=nya_train_device_create("cuda",1024); CHECK(other);
    CHECK(nya_train_device_zero(other,stale));
    CHECK(!nya_train_device_alloc(other,1025));
    CHECK(nya_train_device_alloc(other,1024) && !nya_train_device_alloc(other,1));
    nya_train_device_free(other);
    nya_train_buffer a=nya_train_device_alloc(d,4), b=nya_train_device_alloc(d,4);
    CHECK(a && b);
    nya_train_device_get_stats(d,&before);
    CHECK(nya_train_device_linear(d,small,a,99,1,1,b,1));
    CHECK(nya_train_device_linear(d,small,a,2,1,1,b,1));
    CHECK(nya_train_device_linear(d,small,a,0,2,2,b,1));
    CHECK(nya_train_device_linear_dx(d,small,a,0,1,SIZE_MAX,b,1));
    CHECK(nya_train_device_linear_dw(d,small,a,b,1,1,0));
    nya_train_device_get_stats(d,&after);
    CHECK(!after.failed && before.kernel_launches==after.kernel_launches);
    const unsigned types[]={0,1,30,2,8,12,14};
    for (size_t k=0; k<7; ++k) {
        CHECK(!matrix(d,types[k],65,256,33));
        CHECK(!matrix(d,types[k],3,256,1));
    }
    CHECK(!matrix(d,0,33,65,17) && !matrix(d,30,31,33,65));
    CHECK(!matrix(d,0,8192,33,3) && !matrix(d,0,5,33,8192) && !matrix(d,1,33,8192,3));
    CHECK(!autograd(d) && !nya_train_device_finish(d));
    nya_train_device_get_stats(d,&after);
    printf("resident matrix suite: buffers=%zu bytes=%zu launches=%llu uploads=%llu downloads=%llu\n",after.buffers,after.used_bytes,
        (unsigned long long)after.kernel_launches,(unsigned long long)after.uploads,(unsigned long long)after.downloads);
    nya_train_device_free(d);
    return 0;
}
