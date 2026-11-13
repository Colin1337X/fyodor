#ifndef _WIN32
#define _POSIX_C_SOURCE 200809L
#endif
#include "llm_internal.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define CHECK(x) do {if(!(x)) {fprintf(stderr,"line %d: %s\n",__LINE__,#x);goto done;}} while(0)
static int setting(const char *value)
{
#ifdef _WIN32
    return _putenv_s("NYA_CUDA_ASSISTED_FFN",value);
#else
    return setenv("NYA_CUDA_ASSISTED_FFN",value,1);
#endif
}
static unsigned char *matrix(unsigned type,size_t rows,size_t cols,unsigned tag,size_t *bytes)
{
    size_t block=1,size=4;
    if(type==1 || type==30) size=2;
    if(type==2) {block=32;size=18;}
    if(type==8) {block=32;size=34;}
    if(type==12) {block=256;size=144;}
    if(type==14) {block=256;size=210;}
    *bytes=rows*(cols/block)*size;
    unsigned char *data=calloc(1,*bytes);if(!data)return NULL;
    for(size_t row=0;row<rows;++row) {
        unsigned v=tag+(unsigned)(row%3);
        for(size_t k=0;k<cols/block;++k) {
            unsigned char *p=data+(row*(cols/block)+k)*size;
            if(type==0 || type==1 || type==30) {
                float f=(float)v/128;uint32_t bits;memcpy(&bits,&f,4);
                if(type==0)memcpy(p,&f,4);
                else {unsigned code=type==30?bits>>16:(((bits>>23)-112)<<10)|((bits>>13)&1023U);p[0]=(unsigned char)code;p[1]=(unsigned char)(code>>8);}
            } else if(type==2 || type==8) {
                p[1]=0x20;
                memset(p+2,type==2?(int)((v+8)*17):(int)v,size-2);
            } else if(type==12) {
                p[1]=0x20;
                for(size_t j=0;j<4;++j){p[4+j]=1;p[12+j]=1;}
                memset(p+16,(int)(v*17),128);
            } else {
                memset(p,(int)(v*17),128);memset(p+128,0xaa,64);memset(p+192,1,16);p[209]=0x20;
            }
        }
    }
    return data;
}
static int run(unsigned type,size_t d,size_t h)
{
    int result=1;nya_compute_context *c=NULL;unsigned char *g=NULL,*u=NULL,*w=NULL;
    float *input=calloc(d,sizeof(float)),*output=calloc(d+2,sizeof(float));size_t gb=0,ub=0,wb=0;
    CHECK(input && output);
    g=matrix(type,h,d,1,&gb);u=matrix(type,h,d,2,&ub);w=matrix(type,d,h,3,&wb);CHECK(g && u && w);
    c=nya_compute_create_for("cuda");if(!c){result=77;goto done;}
    nya_compute_ffn f={g,u,w,gb,ub,wb,d,h,type,type,type,0,input,output+1};
    output[0]=output[d+1]=12345;
    CHECK(!setting("0"));CHECK(nya_compute_ffn_f32(c,&f)<0);CHECK(!setting("1"));
    for(unsigned gelu=0;gelu<2;++gelu) for(unsigned repeat=0;repeat<2;++repeat) {
        f.gelu=gelu;double sum=0;
        for(size_t i=0;i<d;++i) {input[i]=(float)((int)(i%7)-2)*(repeat?-0.03125f:0.03125f);sum+=input[i];}
        CHECK(!nya_compute_ffn_f32(c,&f));
        double inner=0;
        for(size_t j=0;j<h;++j) {double a=sum*(double)(1+j%3)/128,b=sum*(double)(2+j%3)/128;
            inner+=(gelu?0.5*a*(1+tanh(0.7978845608028654*(a+0.044715*a*a*a))):a/(1+exp(-a)))*b;}
        for(size_t i=0;i<d;++i) {double expected=inner*(double)(3+i%3)/128;
            CHECK(isfinite(output[i+1]) && fabs(output[i+1]-expected)<=0.0005*(1+fabs(expected)));}
        CHECK(output[0]==12345 && output[d+1]==12345);
    }
    f.gate_bytes--;CHECK(nya_compute_ffn_f32(c,&f)<0);f.gate_bytes++;
    f.gate_type=99;CHECK(nya_compute_ffn_f32(c,&f)<0);f.gate_type=type;
    f.gelu=2;CHECK(nya_compute_ffn_f32(c,&f)<0);f.gelu=1;
    f.width=SIZE_MAX;CHECK(nya_compute_ffn_f32(c,&f)<0);f.width=d;
    CHECK(nya_compute_ffn_f32(c,NULL)<0);CHECK(nya_compute_ffn_f32(NULL,&f)<0);
    CHECK(!nya_compute_ffn_f32(c,&f)); /* rejection must not poison valid reuse */
    printf("type=%u width=%zu hidden=%zu passed\n",type,d,h);result=0;
done:
    nya_compute_free(c);free(g);free(u);free(w);free(input);free(output);return result;
}
/* A rejected large request must neither write output nor poison the context.
   The subsequent small chain uses that same context and immutable weights. */
static int budget(void)
{
    int result=1;nya_compute_context *c=NULL;unsigned char *g=NULL,*u=NULL,*w=NULL;
    float input[256]={0},output[258];size_t gb=0,ub=0,wb=0;
    g=matrix(0,512,256,1,&gb);u=matrix(0,512,256,2,&ub);w=matrix(0,256,512,3,&wb);
    CHECK(g && u && w);CHECK(!setting("1"));
    c=nya_compute_create_for("cuda");if(!c){result=77;goto done;}
    nya_compute_ffn f={g,u,w,gb,ub,wb,256,512,0,0,0,1,input,output+1};
    for(size_t i=0;i<258;++i)output[i]=12345;
    CHECK(nya_compute_ffn_f32(c,&f)<0);
    for(size_t i=0;i<258;++i)CHECK(output[i]==12345);
    /* Reinterpret a small prefix of each F32 allocation; zero input has an
       exact zero oracle for both activation functions and all finite weights. */
    f.width=33;f.hidden=65;f.gate_bytes=f.up_bytes=f.down_bytes=33*65*sizeof(float);
    for(unsigned gelu=0;gelu<2;++gelu) {
        f.gelu=gelu;CHECK(!nya_compute_ffn_f32(c,&f));
        for(size_t i=1;i<=33;++i)CHECK(output[i]==0);
        CHECK(output[0]==12345);
        for(size_t i=34;i<258;++i)CHECK(output[i]==12345);
    }
    puts("1 MiB budget: rejection, unchanged output and same-context recovery passed");result=0;
done:
    nya_compute_free(c);free(g);free(u);free(w);return result;
}
int main(int argc,char **argv)
{
    if(argc==2 && !strcmp(argv[1],"--budget"))return budget();
    if(argc!=1)return 1;
    const unsigned types[]={0,1,30,2,8,12,14};
    for(size_t i=0;i<sizeof(types)/sizeof(types[0]);++i) {int rc=run(types[i],256,512);if(rc)return rc;}
    return run(0,33,65);
}
