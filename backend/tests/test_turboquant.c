#include "turboquant.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define CHECK(x) do { if (!(x)) { fprintf(stderr,"TurboQuant line %d: %s\n",__LINE__,#x); return 1; } } while(0)

static int shape(size_t d,unsigned bits)
{
    const size_t tokens=7,kvheads=2,heads=4;
    float k[7*2*512],v[7*2*512],dk[7*2*512],dv[7*2*512],q[4*512],out[4*512+2];
    nya_turboquant_cache *c=nya_turboquant_create(d,kvheads,tokens,bits,617);
    CHECK(c);
    CHECK(nya_turboquant_token_bytes(c)<2*kvheads*d*sizeof(float));
    CHECK(nya_turboquant_bytes(c)>tokens*nya_turboquant_token_bytes(c));
    for (size_t i=0;i<tokens*kvheads*d;++i) {k[i]=sinf((float)i*0.123f);v[i]=cosf((float)i*0.231f);}
    for (size_t i=0;i<heads*d;++i) q[i]=sinf((float)i*0.37f);
    CHECK(nya_turboquant_store(c,1,k,v)==-1);
    CHECK(nya_turboquant_read(c,0,dk,dv)==-1);
    for (size_t t=0;t<tokens;++t) {
        CHECK(!nya_turboquant_store(c,t,k+t*kvheads*d,v+t*kvheads*d));
        CHECK(!nya_turboquant_read(c,t,dk+t*kvheads*d,dv+t*kvheads*d));
    }
    double err=0,norm=0;
    for (size_t i=0;i<tokens*kvheads*d;++i) {double delta=(double)v[i]-dv[i];err+=delta*delta;norm+=(double)v[i]*v[i];}
    double bound=bits==2?0.19:bits==3?0.06:0.02;
    CHECK(err/norm<bound);
    printf("width=%zu bits=%u value_relative_mse=%.9g bytes/token=%zu\n",d,bits,err/norm,nya_turboquant_token_bytes(c));
    for (size_t window=0;window<8;window+=3) {
        const size_t pos=6,first=window?pos-window+1:0;
        out[0]=out[heads*d+1]=12345;
        CHECK(!nya_turboquant_attention(c,q,heads,pos,window,0.25f,out+1));
        CHECK(out[0]==12345 && out[heads*d+1]==12345);
        for (size_t h=0;h<heads;++h) {
            double scores[7],maximum=-INFINITY,den=0;
            for (size_t t=first;t<=pos;++t) {
                double dot=0;for (size_t j=0;j<d;++j) dot+=(double)q[h*d+j]*dk[(t*kvheads+h/2)*d+j];
                scores[t]=dot*0.25;maximum=fmax(maximum,scores[t]);
            }
            for (size_t t=first;t<=pos;++t) {scores[t]=exp(scores[t]-maximum);den+=scores[t];}
            for (size_t j=0;j<d;++j) {
                double expected=0;
                for (size_t t=first;t<=pos;++t) expected+=scores[t]/den*dv[(t*kvheads+h/2)*d+j];
                CHECK(fabs((double)out[1+h*d+j]-expected)<2e-6*(1+fabs(expected)));
            }
        }
    }
    /* Failed overwrite is atomic, including a bad final head. */
    float save=k[2*d-1];k[2*d-1]=NAN;
    CHECK(nya_turboquant_store(c,0,k,v)==-1);k[2*d-1]=save;
    float rk[1024],rv[1024];CHECK(!nya_turboquant_read(c,0,rk,rv));
    CHECK(!memcmp(rk,dk,2*d*sizeof(float)) && !memcmp(rv,dv,2*d*sizeof(float)));
    CHECK(nya_turboquant_store(c,tokens,k,v)==-1);
    CHECK(nya_turboquant_attention(c,q,3,0,0,1,out)==-1);
    CHECK(nya_turboquant_attention(c,q,4,tokens,0,1,out)==-1);
    CHECK(nya_turboquant_attention(c,q,4,0,0,NAN,out)==-1);
    q[0]=INFINITY;CHECK(nya_turboquant_attention(c,q,4,0,0,1,out)==-1);
    /* Zero and signed zero must remain exactly zero despite nonzero centroids. */
    memset(k,0,2*d*sizeof(float));memset(v,0,2*d*sizeof(float));
    CHECK(!nya_turboquant_store(c,0,k,v));CHECK(!nya_turboquant_read(c,0,rk,rv));
    for (size_t j=0;j<2*d;++j) CHECK(rk[j]==0 && rv[j]==0);
    nya_turboquant_free(c);return 0;
}
int main(void)
{
    CHECK(!nya_turboquant_create(0,1,1,3,1));CHECK(!nya_turboquant_create(513,1,1,3,1));
    CHECK(!nya_turboquant_create(64,SIZE_MAX,1,3,1));CHECK(!nya_turboquant_create(64,1,SIZE_MAX,3,1));
    CHECK(!nya_turboquant_create(64,1,1,1,1));CHECK(!nya_turboquant_create(64,0,1,3,1));
    nya_turboquant_free(NULL);
    for (unsigned bits=2;bits<=4;++bits) {
        CHECK(!shape(32,bits));CHECK(!shape(63,bits));CHECK(!shape(128,bits));CHECK(!shape(512,bits));
    }
    /* QJL expectation is across independently seeded sketches, not across the
     * coordinates of a single fixed cache. This detects omitted residual scale,
     * incorrect transpose and biased MSE-only key reconstruction. */
    double sum=0,squares=0;float x[32],v[32],k[32],rv[32];
    for (unsigned j=0;j<32;++j) x[j]=v[j]=(j==0?1.0f:0.0f);
    for (uint64_t seed=0;seed<256;++seed) {
        nya_turboquant_cache *c=nya_turboquant_create(32,1,1,2,seed);CHECK(c);
        CHECK(!nya_turboquant_store(c,0,x,v));CHECK(!nya_turboquant_read(c,0,k,rv));
        sum+=k[0];squares+=(double)k[0]*k[0];nya_turboquant_free(c);
    }
    double mean=sum/256,variance=squares/256-mean*mean;
    printf("QJL mean %.9g variance %.9g\n",mean,variance);
    CHECK(fabs(mean-1)<0.025 && variance<0.04);
    return 0;
}
