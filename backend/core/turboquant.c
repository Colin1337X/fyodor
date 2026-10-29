/* Original implementation of Zandieh et al., TurboQuant (2025), algorithms
 * 1/2 and QJL, https://arxiv.org/abs/2504.19874. No upstream source copied.
 * Dense Gaussian QR reference; not a fast-Hadamard approximation. */
#include "turboquant.h"
#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

struct nya_turboquant_cache {
    size_t width, heads, capacity, length, key_bytes, value_bytes, token_bytes, bytes;
    unsigned bits;
    double *rotation, *sketch;
    double key_book[8], value_book[16];
    unsigned char *data, *pending;
};
static uint64_t random64(uint64_t *s)
{
    uint64_t z = (*s += UINT64_C(0x9e3779b97f4a7c15));
    z = (z ^ (z >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)) * UINT64_C(0x94d049bb133111eb);
    return z ^ (z >> 31);
}
static double uniform(uint64_t *s)
{ return ((double)(random64(s) >> 11) + 0.5) * 0x1p-53; }
static double gaussian(uint64_t *s)
{ return sqrt(-2 * log(uniform(s))) * cos(6.2831853071795864769 * uniform(s)); }

/* Midpoint quadrature of the exact sphere-coordinate density. Normalization
 * cancels in centroid ratios. Setup only; independent of observed KV data. */
static void codebook(double *book, unsigned bits, size_t width)
{
    unsigned n = 1U << bits;
    double sums[16], weights[16];
    for (unsigned i=0;i<n;++i) book[i]=(4.0*((double)i+0.5)/(double)n-2)/sqrt((double)width);
    for (unsigned iteration=0;iteration<256;++iteration) {
        memset(sums,0,sizeof(sums)); memset(weights,0,sizeof(weights));
        unsigned cell=0;
        for (unsigned j=0;j<8192;++j) {
            double x=-1+2*((double)j+0.5)/8192;
            double weight=exp(((double)width-3)*0.5*log1p(-x*x));
            while (cell+1<n && x>(book[cell]+book[cell+1])*0.5) ++cell;
            sums[cell]+=x*weight; weights[cell]+=weight;
        }
        double change=0;
        for (unsigned i=0;i<n;++i) if (weights[i]>0) {
            double next=sums[i]/weights[i];
            change=fmax(change,fabs(next-book[i]));book[i]=next;
        }
        if (change<1e-12) break;
    }
}
void nya_turboquant_free(nya_turboquant_cache *c)
{
    if (!c) return;
    free(c->rotation);free(c->sketch);free(c->data);free(c->pending);free(c);
}
nya_turboquant_cache *nya_turboquant_create(size_t width,size_t heads,size_t capacity,unsigned bits,uint64_t seed)
{
    if (width<32 || width>512 || !heads || !capacity || bits<2 || bits>4) return NULL;
    size_t kb=8+(width*(bits-1)+7)/8+(width+7)/8, vb=4+(width*bits+7)/8;
    if (heads>SIZE_MAX/(kb+vb)) return NULL;
    size_t token=heads*(kb+vb), overhead=sizeof(nya_turboquant_cache)+2*width*width*sizeof(double);
    if (token>SIZE_MAX-overhead || capacity>(SIZE_MAX-overhead-token)/token) return NULL;
    nya_turboquant_cache *c=calloc(1,sizeof(*c));
    if (!c) return NULL;
    c->width=width;c->heads=heads;c->capacity=capacity;c->bits=bits;
    c->key_bytes=kb;c->value_bytes=vb;c->token_bytes=token;
    c->bytes=overhead+token+capacity*token;
    c->rotation=malloc(width*width*sizeof(double));
    c->sketch=malloc(width*width*sizeof(double));
    c->data=malloc(capacity*token);c->pending=malloc(token);
    if (!c->rotation || !c->sketch || !c->data || !c->pending) goto fail;
    /* Twice-reorthogonalized modified Gram-Schmidt of Gaussian rows. Positive
     * norms fix QR signs and yield a Haar orthogonal frame. */
    for (size_t i=0;i<width;++i) {
        double *r=c->rotation+i*width;
        for (size_t j=0;j<width;++j) r[j]=gaussian(&seed);
        for (unsigned pass=0;pass<2;++pass) for (size_t k=0;k<i;++k) {
            const double *prev=c->rotation+k*width;double dot=0;
            for (size_t j=0;j<width;++j) dot+=r[j]*prev[j];
            for (size_t j=0;j<width;++j) r[j]-=dot*prev[j];
        }
        double norm=0;for (size_t j=0;j<width;++j) norm+=r[j]*r[j];
        norm=sqrt(norm);if (!(norm>1e-12) || !isfinite(norm)) goto fail;
        for (size_t j=0;j<width;++j) r[j]/=norm;
    }
    for (size_t i=0;i<width*width;++i) c->sketch[i]=gaussian(&seed);
    codebook(c->key_book,bits-1,width);codebook(c->value_book,bits,width);
    return c;
fail: nya_turboquant_free(c);return NULL;
}
size_t nya_turboquant_bytes(const nya_turboquant_cache *c) { return c?c->bytes:0; }
size_t nya_turboquant_token_bytes(const nya_turboquant_cache *c) { return c?c->token_bytes:0; }
static void put(unsigned char *p,size_t i,unsigned bits,unsigned code)
{
    size_t bit=i*bits;
    for (unsigned j=0;j<bits;++j) if (code&(1U<<j)) p[(bit+j)/8]|=(unsigned char)(1U<<((bit+j)%8));
}
static unsigned get(const unsigned char *p,size_t i,unsigned bits)
{
    unsigned code=0;size_t bit=i*bits;
    for (unsigned j=0;j<bits;++j) code|=((unsigned)(p[(bit+j)/8]>>((bit+j)%8))&1U)<<j;
    return code;
}
static float norm_at(const unsigned char *p) { float f;memcpy(&f,p,4);return f; }
static int encode(const nya_turboquant_cache *c,const float *x,unsigned char *p,int key)
{
    size_t d=c->width;unsigned bits=c->bits-(unsigned)key, n=1U<<bits;
    const double *book=key?c->key_book:c->value_book;
    double norm=0, rotated[512], residual[512];
    for (size_t j=0;j<d;++j) { if (!isfinite(x[j])) return -1; norm+=(double)x[j]*x[j]; }
    norm=sqrt(norm);if (norm>FLT_MAX) return -1;
    float stored=(float)norm;memcpy(p,&stored,4);
    unsigned char *codes=p+(key?8:4);
    for (size_t i=0;i<d;++i) {
        double dot=0;
        for (size_t j=0;j<d;++j) dot+=c->rotation[i*d+j]*(double)x[j];
        rotated[i]=norm>0?dot/norm:0;
        unsigned cell=0;
        while (cell+1<n && rotated[i]>(book[cell]+book[cell+1])*0.5) ++cell;
        put(codes,i,bits,cell);rotated[i]=book[cell];
    }
    if (key) {
        double rn=0;
        for (size_t j=0;j<d;++j) {
            double estimate=0;for (size_t i=0;i<d;++i) estimate+=c->rotation[i*d+j]*rotated[i];
            /* Correct the actual stored norm, including its rounding. */
            residual[j]=(double)x[j]-(double)stored*estimate;rn+=residual[j]*residual[j];
        }
        rn=sqrt(rn);if (rn>FLT_MAX) return -1;
        float r=(float)rn;memcpy(p+4,&r,4);
        unsigned char *signs=codes+(d*bits+7)/8;
        for (size_t i=0;i<d;++i) {
            double dot=0;for (size_t j=0;j<d;++j) dot+=c->sketch[i*d+j]*residual[j];
            put(signs,i,1,dot>=0?1U:0U);
        }
    }
    return 0;
}
int nya_turboquant_store(nya_turboquant_cache *c,size_t pos,const float *keys,const float *values)
{
    if (!c || !keys || !values || pos>=c->capacity || pos>c->length) return -1;
    memset(c->pending,0,c->token_bytes);
    for (size_t h=0;h<c->heads;++h) {
        unsigned char *p=c->pending+h*(c->key_bytes+c->value_bytes);
        if (encode(c,keys+h*c->width,p,1) || encode(c,values+h*c->width,p+c->key_bytes,0)) return -1;
    }
    memcpy(c->data+pos*c->token_bytes,c->pending,c->token_bytes);
    if (pos==c->length) ++c->length;
    return 0;
}
static void decode(const nya_turboquant_cache *c,const unsigned char *p,double *x,int key)
{
    size_t d=c->width;unsigned bits=c->bits-(unsigned)key;
    const double *book=key?c->key_book:c->value_book;
    const unsigned char *codes=p+(key?8:4),*signs=codes+(d*bits+7)/8;
    double scale=norm_at(p), residual=key?(double)norm_at(p+4)*1.2533141373155002512/(double)d:0;
    for (size_t j=0;j<d;++j) {
        double sum=0,correction=0;
        for (size_t i=0;i<d;++i) {
            sum+=c->rotation[i*d+j]*book[get(codes,i,bits)];
            if (key) correction+=c->sketch[i*d+j]*(get(signs,i,1)?1:-1);
        }
        x[j]=scale*sum+residual*correction;
    }
}
int nya_turboquant_read(nya_turboquant_cache *c,size_t pos,float *keys,float *values)
{
    if (!c || !keys || !values || pos>=c->length) return -1;
    double x[512];
    for (size_t h=0;h<c->heads;++h) {
        const unsigned char *p=c->data+pos*c->token_bytes+h*(c->key_bytes+c->value_bytes);
        decode(c,p,x,1);
        for (size_t j=0;j<c->width;++j) { if (fabs(x[j])>FLT_MAX) return -1;keys[h*c->width+j]=(float)x[j]; }
        decode(c,p+c->key_bytes,x,0);
        for (size_t j=0;j<c->width;++j) { if (fabs(x[j])>FLT_MAX) return -1;values[h*c->width+j]=(float)x[j]; }
    }
    return 0;
}
int nya_turboquant_attention(nya_turboquant_cache *c,const float *query,size_t heads,size_t pos,size_t window,float scale,float *out)
{
    if (!c || !query || !out || !heads || heads%c->heads || heads>SIZE_MAX/c->width/sizeof(float) || pos>=c->length || !isfinite(scale)) return -1;
    size_t d=c->width,first=window && pos>=window?pos-window+1:0;
    for (size_t i=0;i<heads*d;++i) if (!isfinite(query[i])) return -1;
    for (size_t h=0;h<heads;++h) {
        double rq[512],sq[512],acc[512]={0},maximum=-INFINITY,den=0;
        for (size_t i=0;i<d;++i) {
            rq[i]=sq[i]=0;
            for (size_t j=0;j<d;++j) { rq[i]+=c->rotation[i*d+j]*query[h*d+j];sq[i]+=c->sketch[i*d+j]*query[h*d+j]; }
        }
        for (size_t t=first;t<=pos;++t) {
            const unsigned char *p=c->data+t*c->token_bytes+(h/(heads/c->heads))*(c->key_bytes+c->value_bytes);
            const unsigned char *signs=p+8+(d*(c->bits-1)+7)/8,*v=p+c->key_bytes;
            double dot=0,correction=0;
            for (size_t j=0;j<d;++j) { dot+=rq[j]*c->key_book[get(p+8,j,c->bits-1)];correction+=sq[j]*(get(signs,j,1)?1:-1); }
            double score=(dot*norm_at(p)+correction*norm_at(p+4)*1.2533141373155002512/(double)d)*scale;
            if (!isfinite(score)) return -1;
            double next=fmax(maximum,score),old=exp(maximum-next),weight=exp(score-next);
            den=den*old+weight;maximum=next;
            for (size_t j=0;j<d;++j) acc[j]=acc[j]*old+weight*norm_at(v)*c->value_book[get(v+4,j,c->bits)];
        }
        for (size_t j=0;j<d;++j) {
            double sum=0;for (size_t i=0;i<d;++i) sum+=c->rotation[i*d+j]*acc[i];
            sum/=den;if (!isfinite(sum) || fabs(sum)>FLT_MAX) return -1;out[h*d+j]=(float)sum;
        }
    }
    return 0;
}
