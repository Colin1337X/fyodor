#include "compute.h"
#include "cpu_kernels.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

/* Independent double-precision causal/GQA oracle. Odd head widths exercise
   SIMD tails, while multiple queries exercise worker partitioning and windows. */
static int check(const nya_compute_attention *a)
{
    double scores[65];
    size_t stride = a->kv_heads*a->width;
    for (size_t token = 0; token < a->batch; ++token) {
        size_t end = a->position+token;
        size_t first = a->window && end >= a->window ? end-a->window+1 : 0;
        for (size_t head = 0; head < a->heads; ++head) {
            size_t kh = head/(a->heads/a->kv_heads), query = (token*a->heads+head)*a->width;
            double maximum = -INFINITY, sum = 0;
            for (size_t t = first; t <= end; ++t) {
                double dot = 0;
                for (size_t j = 0; j < a->width; ++j) dot += (double)a->query[query+j]*a->keys[t*stride+kh*a->width+j];
                scores[t] = dot*a->scale;
                if (scores[t] > maximum) maximum = scores[t];
            }
            for (size_t t = first; t <= end; ++t) { scores[t] = exp(scores[t]-maximum); sum += scores[t]; }
            for (size_t j = 0; j < a->width; ++j) {
                double expected = 0;
                for (size_t t = first; t <= end; ++t) expected += scores[t]/sum*a->values[t*stride+kh*a->width+j];
                float actual = a->output[query+j];
                if (!isfinite(actual) || fabs(actual-expected) > 2e-6*(1+fabs(expected))) {
                    fprintf(stderr,"attention token %zu head %zu lane %zu: %.9g vs %.12g\n",token,head,j,(double)actual,expected);
                    return 1;
                }
            }
        }
    }
    return 0;
}

int main(void)
{
    float q[33*4*37], k[65*2*37], v[65*2*37], out[33*4*37+2];
    for (size_t i = 0; i < sizeof(q)/sizeof(q[0]); ++i) q[i] = sinf((float)i*0.13f);
    for (size_t i = 0; i < sizeof(k)/sizeof(k[0]); ++i) { k[i] = cosf((float)i*0.17f); v[i] = sinf((float)i*0.07f); }
    nya_compute_context *c = nya_compute_create_for("cpu");
    if (!c) return 1;
    const size_t positions[] = {0,7,17,32,0}, batches[] = {1,17,33,33,1}, windows[] = {1,7,0,17,0};
    nya_compute_attention a = {q,k,v,out+1,1,4,2,37,65,0,0,0.16439899f};
    int supported = nya_cpu_select_attention() != NULL;
    for (size_t p = 0; p < 5; ++p) {
        a.position = positions[p]; a.batch = batches[p]; a.window = windows[p];
        out[0] = out[a.batch*a.heads*a.width+1] = 12345;
        int result = nya_compute_attention_f32(c,&a);
        if ((supported ? result != 0 : result != -1) || (supported && check(&a)) ||
            out[0] != 12345 || out[a.batch*a.heads*a.width+1] != 12345) return 1;
    }
    if (nya_compute_attention_f32(c,NULL) != -1 || nya_compute_attention_f32(NULL,&a) != -1) return 1;
    /* Rejection must precede size multiplication or any output write. */
    for (unsigned p = 0; p < 7; ++p) {
        nya_compute_attention bad = a;
        if (p == 0) bad.heads = SIZE_MAX;
        if (p == 1) bad.batch = SIZE_MAX;
        if (p == 2) bad.capacity = SIZE_MAX;
        if (p == 3) bad.kv_heads = 3;
        if (p == 4) bad.width = 0;
        if (p == 5) bad.query = NULL;
        if (p == 6) bad.scale = NAN;
        if (nya_compute_attention_f32(c,&bad) != -1) return 1;
    }
    if (supported) {
        float saved = q[0]; q[0] = NAN;
        if (nya_compute_attention_f32(c,&a) != -1) return 1;
        q[0] = saved;
        if (nya_compute_attention_f32(c,&a) || check(&a)) return 1;
    }
    nya_compute_free(c);
    return 0;
}
