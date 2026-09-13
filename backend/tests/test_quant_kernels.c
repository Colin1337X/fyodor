#include "llm_internal.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* F32 reductions may disagree near zero through cancellation, so extended
   fixtures also use a double-precision dot oracle and its sum of magnitudes.
   A 1e-6 forward-error bound is far tighter than the worst-case 1024-term F32
   bound. The original 17-row / three-token relative checks remain below. */
static int oracle(const nya_llm_tensor *t, const float *x, size_t row, size_t columns, float actual)
{
    double sum = 0, magnitude = 0;
    for (size_t i = 0; i < columns; ++i) {
        double product = (double)nya_llm_tensor_value(t,row*columns+i)*x[i];
        sum += product; magnitude += fabs(product);
    }
    if (!isfinite(actual) || fabs((double)actual-sum) > 1e-6*(1+magnitude)) {
        fprintf(stderr,"double oracle: type %u row %zu actual %.9g expected %.12g magnitude %.9g\n",t->type,row,(double)actual,sum,magnitude);
        return 0;
    }
    return 1;
}

static int storage_edges(void)
{
    unsigned char *data = malloc(65536*2);
    float *out = malloc(65536*sizeof(float)), x = 1;
    nya_compute_context *c = nya_compute_create();
    if (!data || !out || !c) { free(data); free(out); nya_compute_free(c); return 1; }
    const unsigned types[] = {1,30};
    for (size_t t = 0; t < 2; ++t) {
        /* Exhaust every finite F16/BF16 bit pattern, especially subnormal
           scales. Nonfinite encodings are replaced with zero for this test. */
        unsigned mask = t == 0 ? 0x7c00U : 0x7f80U;
        for (unsigned i = 0; i < 65536; ++i) {
            unsigned bits = (i & mask) == mask ? 0 : i;
            data[2*i] = (unsigned char)bits; data[2*i+1] = (unsigned char)(bits >> 8);
        }
        nya_llm_tensor tensor = {0}; tensor.data = data; tensor.type = types[t];
        if (nya_compute_matvec_typed(c,data,65536,1,types[t],&x,out)) return 1;
        for (size_t i = 0; i < 65536; ++i) if (out[i] != nya_llm_tensor_value(&tensor,i)) {
            fprintf(stderr,"storage conversion type %u encoding %zu failed\n",types[t],i); return 1;
        }
    }
    free(data); free(out); nya_compute_free(c); return 0;
}

/* A 1 MiB optional scratch budget must split this matrix into 32/32/1 rows.
   Output leading dimension remains 65 across chunks and all 33 prompt columns.
   Exact binary fractions give an independent analytic dot product and expose
   transposition/stride mistakes without a tolerance dominated by cancellation. */
static int chunked_matrix(void)
{
    const size_t rows=getenv("NYA_TEST_CUTLASS") ? 68U : 65U, columns=8192, batch=33;
    float *w=malloc(rows*columns*sizeof(float)), *x=malloc(batch*columns*sizeof(float));
    float *y=malloc((rows*batch+2)*sizeof(float));
    nya_compute_context *c=nya_compute_create();
    int failed=1;
    if (!w || !x || !y || !c) goto done;
    for (size_t r=0; r<rows; ++r) for (size_t k=0; k<columns; ++k) w[r*columns+k]=(float)(r+1)/128;
    for (size_t b=0; b<batch; ++b) for (size_t k=0; k<columns; ++k) x[b*columns+k]=(float)(b+1)/64;
    y[0]=y[rows*batch+1]=12345;
    if (nya_compute_matmul_typed(c,w,rows,columns,0,x,y+1,batch)) goto done;
    for (size_t b=0; b<batch; ++b) for (size_t r=0; r<rows; ++r)
        if (y[1+b*rows+r] != (float)((r+1)*(b+1))) goto done;
    if (y[0]!=12345 || y[rows*batch+1]!=12345) goto done;
    failed=0;
done:
    free(w);free(x);free(y);nya_compute_free(c);
    if (failed) fprintf(stderr,"chunked matrix/leading-dimension validation failed\n");
    return failed;
}

int main(void)
{
    if (getenv("NYA_TEST_BLAS_CHUNKS") && chunked_matrix()) return 1;
    const unsigned types[] = {0, 1, 30, 2, 8, 12, 14};
    const size_t strides[] = {4096, 2048, 2048, 576, 1088, 576, 840};
    const size_t rows = getenv("NYA_TEST_CUTLASS") ? 64U : 65U, columns = 1024;
    float x[1024], actual[65], expected[65];
    for (size_t k = 0; k < 7; ++k) {
        unsigned char *data = malloc(strides[k] * rows);
        nya_compute_context *compute = nya_compute_create();
        const char *requested = getenv("NYA_COMPUTE");
        if (requested && !strcmp(requested, "cuda") && strcmp(nya_compute_name(compute), "cuda")) {
            free(data); nya_compute_free(compute); return 77;
        }
        if (!data || !compute) { free(data); nya_compute_free(compute); return 1; }
        for (size_t i = 0; i < strides[k]*rows; ++i) data[i] = (unsigned char)((i*73 + i/17 + 31) & 255);
        if (types[k] == 0 || types[k] == 1 || types[k] == 30) {
            for (size_t i = 0; i < rows*columns; ++i) {
                float value = (float)((int)(i%127)-63) / 64;
                if (types[k] == 0) memcpy(data+i*4, &value, 4);
                else {
                    uint32_t bits; memcpy(&bits, &value, 4);
                    uint16_t v = types[k] == 30 ? (uint16_t)(bits >> 16) : (uint16_t)(0x3000U + (unsigned)(i%1024) + (i%2 ? 0x8000U : 0));
                    data[i*2] = (unsigned char)v; data[i*2+1] = (unsigned char)(v >> 8);
                }
            }
        } else {
            size_t block = types[k] == 2 ? 18 : types[k] == 8 ? 34 : types[k] == 12 ? 144 : 210;
            for (size_t i = 0; i < strides[k]*rows; i += block) {
                size_t scale = types[k] == 14 ? 208 : 0;
                data[i+scale] = 0; data[i+scale+1] = 0x28;
                if (types[k] == 12) { data[i+2] = 0; data[i+3] = 0x24; }
            }
        }
        nya_llm_tensor tensor = {0}; tensor.data = data; tensor.type = types[k];
        for (size_t pass = 0; pass < 3; ++pass) {
            for (size_t i = 0; i < columns; ++i) x[i] = (float)((int)((i+pass*3)%97)-48) / 49;
            nya_llm_matvec(NULL, expected, &tensor, x, columns, rows);
            if (nya_compute_matvec_typed(compute, data, rows, columns, types[k], x, actual)) return 1;
            for (size_t row = 0; row < rows; ++row) {
                if (!oracle(&tensor,x,row,columns,actual[row]) || (row < 17 && fabsf(actual[row]-expected[row]) > 0.00015f*(1+fabsf(expected[row])))) {
                    fprintf(stderr, "type %u row %zu: %.9g vs %.9g\n", types[k], row, (double)actual[row], (double)expected[row]); return 1;
                }
            }
        }
        /* Cross both 32/64 CUDA tile boundaries and CPU SIMD widths, including
           partial final tiles. Reuse one context across differently sized jobs
           to cover persistent scratch growth, shrink/reuse and worker tails. */
        if (!strcmp(nya_compute_name(compute), "cpu") || !strcmp(nya_compute_name(compute), "cuda")) {
            const size_t batches[] = {3, 17, 65, 32, 8};
            float *inputs = malloc(65*columns*sizeof(float)), *outputs = malloc((65*rows+2)*sizeof(float));
            if (!inputs || !outputs) { free(inputs); free(outputs); free(data); nya_compute_free(compute); return 1; }
            for (size_t i = 0; i < 65*columns; ++i) inputs[i] = (float)((int)(i%61)-30)/31;
            for (size_t p = 0; p < sizeof(batches)/sizeof(batches[0]); ++p) {
            size_t batch = batches[p];
            outputs[0] = outputs[batch*rows+1] = 12345;
            if (nya_compute_matmul_typed(compute, data, rows, columns, types[k], inputs, outputs+1, batch)) return 1;
            if (outputs[0] != 12345 || outputs[batch*rows+1] != 12345) return 1;
            for (size_t token = 0; token < batch; ++token) {
                nya_llm_matvec(NULL, expected, &tensor, inputs+token*columns, columns, rows);
                for (size_t row = 0; row < rows; ++row) {
                    float value = outputs[1+token*rows+row];
                    if (!oracle(&tensor,inputs+token*columns,row,columns,value) ||
                        (row < 17 && token < 3 && fabsf(value-expected[row]) > 0.00015f*(1+fabsf(expected[row])))) {
                        fprintf(stderr,"GEMM type %u batch %zu token %zu row %zu: %.9g vs %.9g\n",types[k],batch,token,row,(double)value,(double)expected[row]); return 1;
                    }
                }
            }
            }
            free(inputs); free(outputs);
        }
        nya_compute_free(compute); free(data);
    }
    return storage_edges();
}
