#include "training.h"
#include "training_internal.h"
#include "thread.h"
#include "train_executor.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define REQUIRE(x) do { if (!(x)) { fprintf(stderr, "training check failed at line %d\n", __LINE__); return -1; } } while (0)

static nya_train_tensor *objective(nya_train_graph *g, nya_train_parameter **p, int dpo)
{
    const uint32_t ids[] = {1, 3}, other[] = {2, 0}, labels[] = {2, 1};
    const unsigned char mask[] = {0, 1};
    nya_train_tensor *table = nya_train_leaf(g, p[0]), *w = nya_train_leaf(g, p[1]);
    nya_train_tensor *norm = nya_train_leaf(g, p[2]), *bias = nya_train_leaf(g, p[3]);
    nya_train_tensor *input = nya_train_embedding(table, ids, 2);
    nya_train_tensor *x = nya_train_rms_norm(input, norm, 1e-4f);
    x = nya_train_mul(nya_train_gelu(x), nya_train_silu(nya_train_scale(input, 0.75f)));
    nya_train_tensor *logits = nya_train_add(nya_train_linear(x, w), bias);
    if (!dpo) return nya_train_cross_entropy(logits, labels, mask, 2);
    nya_train_tensor *rejected = nya_train_add(nya_train_linear(nya_train_embedding(table, other, 2), w), bias);
    return nya_train_dpo(nya_train_logprob(logits, labels, NULL, 2),
        nya_train_logprob(rejected, labels, NULL, 2), -1.2, -2.1, 0.4f);
}

/* Central finite differences independently check all trainable coordinates in
   a branched graph. This covers broadcasting, repeated leaves, embedding gather,
   nonlinearities, RMSNorm, label masking, log-softmax and DPO chain rules. */
static int gradients(void)
{
    nya_train_parameter *p[4];
    size_t rows[] = {4, 4, 1, 1}, cols[] = {3, 3, 3, 4};
    float initial[16], expected[4][16];
    for (size_t k = 0; k < 4; ++k) {
        for (size_t i = 0; i < rows[k] * cols[k]; ++i) initial[i] = (float)((int)((i * 7 + k * 11) % 19) - 9) / 16.0f;
        p[k] = nya_train_parameter_create(rows[k], cols[k], initial); REQUIRE(p[k] != NULL);
    }
    for (int mode = 0; mode < 2; ++mode) {
        nya_train_graph *g = nya_train_graph_create(1024 * 1024);
        for (size_t k = 0; k < 4; ++k) nya_train_zero_grad(p[k]);
        nya_train_tensor *loss = objective(g, p, mode);
        REQUIRE(loss != NULL && nya_train_backward(loss) == 0);
        for (size_t k = 0; k < 4; ++k) memcpy(expected[k], nya_train_parameter_gradient(p[k]), rows[k] * cols[k] * sizeof(float));
        nya_train_graph *eval = nya_train_graph_create_for_evaluation(1024*1024,NULL);
        nya_train_tensor *evaluated = objective(eval,p,mode);
        REQUIRE(evaluated && memcmp(nya_train_data(loss),nya_train_data(evaluated),sizeof(float)) == 0);
        REQUIRE(nya_train_memory_used(eval) < nya_train_memory_used(g));
        REQUIRE(nya_train_backward(evaluated) != 0);
        for (size_t k = 0; k < 4; ++k)
            REQUIRE(memcmp(expected[k],nya_train_parameter_gradient(p[k]),rows[k]*cols[k]*sizeof(float)) == 0);
        nya_train_graph_free(eval);
        REQUIRE(nya_train_backward(loss) != 0);
        nya_train_graph_free(g);
        for (size_t k = 0; k < 4; ++k) for (size_t i = 0; i < rows[k] * cols[k]; ++i) {
            float *v = nya_train_parameter_data(p[k]), original = v[i], plus, minus;
            g = nya_train_graph_create(1024 * 1024); v[i] = original + 0.001f;
            loss = objective(g, p, mode); REQUIRE(loss != NULL); plus = nya_train_data(loss)[0]; nya_train_graph_free(g);
            g = nya_train_graph_create(1024 * 1024); v[i] = original - 0.001f;
            loss = objective(g, p, mode); REQUIRE(loss != NULL); minus = nya_train_data(loss)[0]; nya_train_graph_free(g);
            v[i] = original;
            double numeric = ((double)plus - minus) / 0.002;
            if (fabs(numeric - expected[k][i]) > 0.0003) {
                fprintf(stderr, "gradient mode=%d parameter=%zu index=%zu analytic=%g numeric=%g\n", mode,k,i,(double)expected[k][i],numeric); return -1;
            }
        }
    }
    for (size_t k = 0; k < 4; ++k) nya_train_parameter_free(p[k]);
    return 0;
}

static int lora(void)
{
    const float frozen[] = {0.2f,0.1f,-0.1f, -0.2f,0.2f,0.3f, 0.4f,-0.1f,0.1f};
    const float initial[] = {0.1f,-0.1f,0.2f, -0.2f,0.15f,0.05f};
    const float inputs[] = {1,0,0, 0,1,0, 0,0,1};
    const uint32_t labels[] = {2,0,1};
    nya_train_parameter *p[] = {nya_train_parameter_create(2,3,initial), nya_train_parameter_create(3,2,NULL)};
    nya_train_adamw optimizer; char error[160] = {0}; float first = 0, last = 0;
    REQUIRE(p[0] != NULL && p[1] != NULL);
    nya_train_adamw_defaults(&optimizer); optimizer.learning_rate = 0.05f; optimizer.weight_decay = 0;
    for (int step = 0; step < 200; ++step) {
        nya_train_graph *g = nya_train_graph_create(1024 * 1024);
        nya_train_zero_grad(p[0]); nya_train_zero_grad(p[1]);
        nya_train_tensor *x = nya_train_input(g,3,3,inputs);
        /* W remains an input constant; only A/B are parameter leaves. The base
           tensor receives no gradient or optimizer state. LoRA uses alpha/r=1. */
        nya_train_tensor *base = nya_train_linear(x, nya_train_input(g,3,3,frozen));
        nya_train_tensor *adapter = nya_train_linear(nya_train_linear(x,nya_train_leaf(g,p[0])),nya_train_leaf(g,p[1]));
        nya_train_tensor *loss = nya_train_cross_entropy(nya_train_add(base,adapter),labels,NULL,3);
        REQUIRE(loss != NULL); last = nya_train_data(loss)[0]; if (step == 0) first = last;
        REQUIRE(nya_train_backward(loss) == 0);
        nya_train_graph_free(g);
        REQUIRE(nya_train_adamw_step(&optimizer,p,2,error,sizeof(error)) == 0);
    }
    REQUIRE(last < 0.01f && last < first / 50);
    /* Invalid gradients must not advance moments, weights or the step count. */
    float saved = nya_train_parameter_data(p[0])[0]; uint64_t saved_step = optimizer.step;
    nya_train_graph *g = nya_train_graph_create(1024);
    nya_train_tensor *loss = nya_train_scale(nya_train_leaf(g,p[0]), INFINITY);
    REQUIRE(loss == NULL && nya_train_error(g)[0] != '\0'); nya_train_graph_free(g);
    nya_train_parameter *duplicates[] = {p[0],p[0]};
    REQUIRE(nya_train_adamw_step(&optimizer,duplicates,2,error,sizeof(error)) != 0);
    REQUIRE(optimizer.step == saved_step && nya_train_parameter_data(p[0])[0] == saved);
    nya_train_parameter_free(p[0]); nya_train_parameter_free(p[1]);
    printf("LoRA loss %.6f -> %.6f\n",(double)first,(double)last);
    return 0;
}

static nya_train_tensor *attention_loss(nya_train_graph *g, nya_train_parameter **p, int mode)
{
    float frequency = 0.3f;
    uint32_t labels[] = {1,2,3}, groups[] = {0,1,1};
    nya_train_tensor *q = nya_train_rope(nya_train_leaf(g,p[0]),2,2,&frequency,mode);
    nya_train_tensor *k = nya_train_rope(nya_train_leaf(g,p[1]),1,2,&frequency,mode);
    nya_train_tensor *v = nya_train_reshape(nya_train_leaf(g,p[2]),3,2);
    return nya_train_cross_entropy(nya_train_attention(q,k,v,2,1,2,0.7f,mode ? 2 : 0,groups),labels,NULL,3);
}

static int attention_gradients(void)
{
    float initial[12], expected[3][12];
    nya_train_parameter *p[3]; size_t counts[] = {12,6,6};
    for (size_t k = 0; k < 3; ++k) {
        for (size_t i = 0; i < counts[k]; ++i) initial[i] = (float)((int)((i * 7 + k * 5) % 17) - 8) / 8.0f;
        p[k] = nya_train_parameter_create(3,counts[k]/3,initial); REQUIRE(p[k] != NULL);
    }
    for (int mode = 0; mode < 2; ++mode) {
        nya_train_graph *g = nya_train_graph_create(1024*1024);
        for (size_t k = 0; k < 3; ++k) nya_train_zero_grad(p[k]);
        nya_train_tensor *loss = attention_loss(g,p,mode);
        REQUIRE(loss != NULL && nya_train_backward(loss) == 0);
        for (size_t k = 0; k < 3; ++k) memcpy(expected[k],nya_train_parameter_gradient(p[k]),counts[k]*sizeof(float));
        nya_train_graph *eval = nya_train_graph_create_for_evaluation(1024*1024,NULL);
        nya_train_tensor *value = attention_loss(eval,p,mode);
        REQUIRE(value && memcmp(nya_train_data(value),nya_train_data(loss),sizeof(float)) == 0);
        REQUIRE(nya_train_memory_used(eval) < nya_train_memory_used(g));
        REQUIRE(nya_train_backward(value) != 0);
        for (size_t k = 0; k < 3; ++k)
            REQUIRE(memcmp(expected[k],nya_train_parameter_gradient(p[k]),counts[k]*sizeof(float)) == 0);
        nya_train_graph_free(eval);
        nya_train_graph_free(g);
        for (size_t k = 0; k < 3; ++k) for (size_t i = 0; i < counts[k]; ++i) {
            float *v = nya_train_parameter_data(p[k]), original = v[i], plus, minus;
            g = nya_train_graph_create(1024*1024); v[i] = original+0.001f;
            loss = attention_loss(g,p,mode); REQUIRE(loss != NULL); plus = nya_train_data(loss)[0]; nya_train_graph_free(g);
            g = nya_train_graph_create(1024*1024); v[i] = original-0.001f;
            loss = attention_loss(g,p,mode); REQUIRE(loss != NULL); minus = nya_train_data(loss)[0]; nya_train_graph_free(g);
            v[i] = original;
            REQUIRE(fabs(((double)plus-minus)/0.002-expected[k][i]) < 0.0003);
        }
    }
    for (size_t k = 0; k < 3; ++k) nya_train_parameter_free(p[k]);
    return 0;
}

static int invalid_inputs(void)
{
    nya_train_graph *g = nya_train_graph_create(256);
    float input = 0;
    REQUIRE(nya_train_input(g,SIZE_MAX,2,&input) == NULL);
    nya_train_graph_free(g);
    g = nya_train_graph_create(1024);
    float logits[] = {1,2}; uint32_t label = 1; unsigned char mask = 0;
    REQUIRE(nya_train_cross_entropy(nya_train_input(g,1,2,logits),&label,&mask,1) == NULL);
    nya_train_graph_free(g);
    return 0;
}

static int checkpoint(void)
{
    float initial[] = {-0.5f,0.8f}; uint32_t label = 0;
    nya_train_parameter *a = nya_train_parameter_create(1,2,initial), *b = nya_train_parameter_create(1,2,NULL);
    nya_train_adamw first, resumed; char error[160];
    FILE *file = tmpfile();
    REQUIRE(a != NULL && b != NULL && file != NULL);
    nya_train_adamw_defaults(&first); nya_train_adamw_defaults(&resumed);
    nya_train_graph *g = nya_train_graph_create(4096);
    REQUIRE(nya_train_backward(nya_train_cross_entropy(nya_train_leaf(g,a),&label,NULL,1)) == 0);
    nya_train_graph_free(g);
    REQUIRE(nya_train_adamw_step(&first,&a,1,error,sizeof(error)) == 0);
    REQUIRE(nya_train_checkpoint_write(file,&first,&a,1) == 0);
    rewind(file);
    REQUIRE(nya_train_checkpoint_read(file,&resumed,&b,1) == 0);
    REQUIRE(first.step == resumed.step && memcmp(nya_train_parameter_data(a),nya_train_parameter_data(b),sizeof(initial)) == 0);
    /* The next update checks that moments and accumulated gradients, not only
       weights, survive a save/resume boundary exactly. */
    REQUIRE(nya_train_adamw_step(&first,&a,1,error,sizeof(error)) == 0);
    REQUIRE(nya_train_adamw_step(&resumed,&b,1,error,sizeof(error)) == 0);
    REQUIRE(first.step == resumed.step && memcmp(nya_train_parameter_data(a),nya_train_parameter_data(b),sizeof(initial)) == 0);
    REQUIRE(fseek(file,-1,SEEK_END) == 0);
    int byte = fgetc(file); REQUIRE(byte != EOF && fseek(file,-1,SEEK_END) == 0 && fputc(byte^1,file) != EOF && fflush(file) == 0);
    nya_train_parameter_data(b)[0] = 99.0f; resumed.step = 99;
    rewind(file);
    REQUIRE(nya_train_checkpoint_read(file,&resumed,&b,1) != 0);
    REQUIRE(nya_train_parameter_data(b)[0] == 99.0f && resumed.step == 99);
    fclose(file); nya_train_parameter_free(a); nya_train_parameter_free(b);
    return 0;
}

static int frozen_matrix(void)
{
    /* BF16 weights exercise the exact frozen-quantized backward path that LoRA
       needs to propagate gradients into earlier adapters, without base grads. */
    const unsigned char bytes[] = {0x00,0x3f, 0x80,0x3f, 0x00,0xbf, 0x80,0xbf, 0x00,0x3e, 0x40,0x3f};
    float initial[] = {0.2f,-0.4f,0.7f}; uint32_t label = 1;
    nya_llm_tensor weight = {0};
    weight.type = NYA_LLM_TENSOR_BF16; weight.dimension_count = 2;
    weight.dimensions[0] = 3; weight.dimensions[1] = 2; weight.data = bytes; weight.data_size = sizeof(bytes);
    nya_train_parameter *p = nya_train_parameter_create(1,3,initial); REQUIRE(p != NULL);
    nya_train_graph *g = nya_train_graph_create(4096);
    nya_train_tensor *loss = nya_train_cross_entropy(nya_train_linear_mapped(nya_train_leaf(g,p),&weight),&label,NULL,1);
    REQUIRE(loss != NULL && nya_train_backward(loss) == 0);
    float gradients[3]; memcpy(gradients,nya_train_parameter_gradient(p),sizeof(gradients));
    nya_train_graph_free(g);
    for (size_t i = 0; i < 3; ++i) {
        float *data = nya_train_parameter_data(p), value = data[i], plus, minus;
        data[i] = value+0.001f; g = nya_train_graph_create(4096);
        loss = nya_train_cross_entropy(nya_train_linear_mapped(nya_train_leaf(g,p),&weight),&label,NULL,1);
        REQUIRE(loss != NULL); plus = nya_train_data(loss)[0]; nya_train_graph_free(g);
        data[i] = value-0.001f; g = nya_train_graph_create(4096);
        loss = nya_train_cross_entropy(nya_train_linear_mapped(nya_train_leaf(g,p),&weight),&label,NULL,1);
        REQUIRE(loss != NULL); minus = nya_train_data(loss)[0]; nya_train_graph_free(g); data[i] = value;
        REQUIRE(fabs(((double)plus-minus)/0.002-gradients[i]) < 0.0003);
    }
    nya_train_parameter_free(p);
    return 0;
}

/* Independent scalar linear oracle, including token tails, only one trainable
   operand, shared parameter leaves, and accumulation across graphs. The legacy
   operation order is the contract: require bitwise equality, not a new tolerance. */
static int linear_oracle(void)
{
    const size_t shapes[][3] = {{1,3,5},{3,17,9},{4,65,5},{5,17,9},{7,65,5},{8,3,8}};
    for (size_t s = 0; s < sizeof(shapes)/sizeof(shapes[0]); ++s) for (int mode = 0; mode < 4; ++mode) {
        size_t n = shapes[s][0], k = shapes[s][1], m = mode == 3 ? n : shapes[s][2];
        float x[8*65], w[9*65], expected_x[8*65] = {0}, expected_w[9*65] = {0}, y[8*9];
        uint32_t labels[8];
        for (size_t i = 0; i < n*k; ++i) x[i] = (float)((int)(i*17%31)-15)/32;
        for (size_t i = 0; i < m*k; ++i) w[i] = mode == 3 ? x[i] : (float)((int)(i*13%29)-14)/32;
        for (size_t i = 0; i < n; ++i) labels[i] = (uint32_t)(i%m);
        nya_train_parameter *px = nya_train_parameter_create(n,k,x);
        nya_train_parameter *pw = mode == 3 ? px : nya_train_parameter_create(m,k,w);
        REQUIRE(px != NULL && pw != NULL);
        for (int repeat = 0; repeat < 2; ++repeat) {
            nya_train_graph *g = nya_train_graph_create(1024*1024);
            nya_train_tensor *a = mode == 1 ? nya_train_input(g,n,k,x) : nya_train_leaf(g,px);
            nya_train_tensor *b = mode == 2 ? nya_train_input(g,m,k,w) : nya_train_leaf(g,pw);
            nya_train_tensor *out = nya_train_linear(a,b);
            REQUIRE(out != NULL);
            for (size_t row = 0; row < n; ++row) for (size_t o = 0; o < m; ++o) {
                double sum = 0;
                for (size_t i = 0; i < k; ++i) sum += (double)x[row*k+i]*w[o*k+i];
                y[row*m+o] = (float)sum;
            }
            REQUIRE(memcmp(y,nya_train_data(out),n*m*sizeof(float)) == 0);
            REQUIRE(nya_train_backward(nya_train_cross_entropy(out,labels,NULL,n)) == 0);
            for (size_t row = 0; row < n; ++row) {
                double maximum = y[row*m], sum = 0;
                for (size_t o = 1; o < m; ++o) if (y[row*m+o] > maximum) maximum = y[row*m+o];
                for (size_t o = 0; o < m; ++o) sum += exp((double)y[row*m+o]-maximum);
                for (size_t o = 0; o < m; ++o) {
                    double probability = exp((double)y[row*m+o]-maximum)/sum;
                    float dy = (float)((-1.0/(double)n)*((o == labels[row] ? 1.0 : 0.0)-probability));
                    for (size_t i = 0; i < k; ++i) {
                        if (mode != 1) expected_x[row*k+i] += dy*w[o*k+i];
                        if (mode == 3) expected_x[o*k+i] += dy*x[row*k+i];
                        else if (mode != 2) expected_w[o*k+i] += dy*x[row*k+i];
                    }
                }
            }
            REQUIRE(memcmp(expected_x,nya_train_parameter_gradient(px),n*k*sizeof(float)) == 0);
            if (mode != 3) REQUIRE(memcmp(expected_w,nya_train_parameter_gradient(pw),m*k*sizeof(float)) == 0);
            nya_train_graph_free(g);
        }
        if (mode != 3) nya_train_parameter_free(pw);
        nya_train_parameter_free(px);
    }
    return 0;
}

static nya_train_tensor *attention_tile_loss(nya_train_graph *g, nya_train_parameter **p, size_t dimension, int window)
{
    const uint32_t labels[] = {1,2,0,3}, groups[] = {0,1,1,0};
    return nya_train_cross_entropy(nya_train_attention(nya_train_leaf(g,p[0]),
        nya_train_leaf(g,p[1]),nya_train_leaf(g,p[2]),2,1,dimension,0.5f,
        window ? 2 : 0,groups),labels,NULL,4);
}

static int attention_tile_gradients(void)
{
    const size_t dimensions[] = {3,4,5,7,8,9};
    for (size_t d = 0; d < sizeof(dimensions)/sizeof(dimensions[0]); ++d) for (int window = 0; window < 2; ++window) {
        size_t dimension = dimensions[d];
        nya_train_parameter *p[3]; float initial[72], gradient[3][72];
        for (size_t k = 0; k < 3; ++k) {
            for (size_t i = 0; i < 72; ++i) initial[i] = (float)((int)((i*7+k*13)%23)-11)/16;
            p[k] = nya_train_parameter_create(4,dimension*(k == 0 ? 2 : 1),initial);
            REQUIRE(p[k] != NULL);
        }
        nya_train_graph *g = nya_train_graph_create(1024*1024);
        REQUIRE(nya_train_backward(attention_tile_loss(g,p,dimension,window)) == 0);
        for (size_t k = 0; k < 3; ++k) memcpy(gradient[k],nya_train_parameter_gradient(p[k]),
            4*dimension*(k == 0 ? 2 : 1)*sizeof(float));
        nya_train_graph_free(g);
        for (size_t k = 0; k < 3; ++k) for (size_t i = 0; i < 4*dimension*(k == 0 ? 2 : 1); ++i) {
            float *values = nya_train_parameter_data(p[k]), value = values[i], plus, minus;
            values[i] = value+0.001f; g = nya_train_graph_create(1024*1024);
            nya_train_tensor *loss = attention_tile_loss(g,p,dimension,window); REQUIRE(loss != NULL);
            plus = nya_train_data(loss)[0]; nya_train_graph_free(g);
            values[i] = value-0.001f; g = nya_train_graph_create(1024*1024);
            loss = attention_tile_loss(g,p,dimension,window); REQUIRE(loss != NULL);
            minus = nya_train_data(loss)[0]; nya_train_graph_free(g); values[i] = value;
            REQUIRE(fabs(((double)plus-minus)/0.002-gradient[k][i]) < 0.0003);
        }
        for (size_t k = 0; k < 3; ++k) nya_train_parameter_free(p[k]);
    }
    return 0;
}

static int mapped_gradient_oracle(void)
{
    const unsigned types[] = {0,1,30,2,8,12,14};
    const size_t row_bytes[] = {1024,512,512,144,272,144,210};
    unsigned char bytes[5*2048]; float input[7*512], expected[7*512], output[7*5];
    uint32_t labels[7] = {0,1,2,3,4,0,1};
    for (size_t i = 0; i < 7*512; ++i) input[i] = (float)((int)(i*7%17)-8)/128;
    for (size_t format = 0; format < 7; ++format) for (size_t shape = 0; shape < 2; ++shape) {
        unsigned type = types[format];
        size_t columns = shape == 0 ? 256 : format < 3 ? 259 : 512;
        size_t rb = format < 3 ? columns*(type == 0 ? 4 : 2) : row_bytes[format]*(columns/256);
        for (size_t i = 0; i < 5*rb; ++i) bytes[i] = (unsigned char)(i*13+7);
        for (size_t row = 0; row < 5; ++row) {
            unsigned char *w = bytes+row*rb;
            if (type == 0) for (size_t i = 0; i < columns; ++i) {
                float value = (float)((int)((row*3+i)%19)-9)/32; memcpy(w+4*i,&value,4);
            }
            else if (type == 1 || type == 30) for (size_t i = 0; i < columns; ++i) {
                w[2*i] = (unsigned char)(i%64); w[2*i+1] = type == 1 ? 0x30 : 0x3e;
            }
            else if (type == 2 || type == 8) for (size_t b = 0; b < columns/32; ++b) {
                w[b*(type == 2 ? 18 : 34)] = 0; w[b*(type == 2 ? 18 : 34)+1] = 0x30;
            }
            else for (size_t b = 0; b < columns/256; ++b) {
                if (type == 12) { w[b*144] = w[b*144+2] = 0; w[b*144+1] = w[b*144+3] = 0x30; }
                else { w[b*210+208] = 0; w[b*210+209] = 0x30; }
            }
        }
        nya_llm_tensor w = {0}; w.type = type; w.dimension_count = 2;
        w.dimensions[0] = columns; w.dimensions[1] = 5; w.data = bytes; w.data_size = 5*rb;
        nya_train_parameter *p = nya_train_parameter_create(7,columns,input); REQUIRE(p != NULL);
        size_t budget = 1024*1024;
        for (int tight = 0; tight < 2; ++tight) {
            nya_train_zero_grad(p); memset(expected,0,sizeof(expected));
            nya_train_graph *g = nya_train_graph_create(budget);
            nya_train_tensor *logits = nya_train_linear_mapped(nya_train_leaf(g,p),&w);
            REQUIRE(logits != NULL); memcpy(output,nya_train_data(logits),sizeof(output));
            float reference[7*5];
            for (size_t n = 0; n < 7; ++n) nya_llm_matvec(NULL,reference+n*5,&w,input+n*columns,columns,5);
            REQUIRE(memcmp(output,reference,sizeof(output)) == 0);
            nya_train_tensor *loss = nya_train_cross_entropy(logits,labels,NULL,7); REQUIRE(loss != NULL);
            budget = nya_train_memory_used(g);
            REQUIRE(nya_train_backward(loss) == 0);
            if (tight) REQUIRE(nya_train_memory_used(g) == budget); /* scalar fallback */
            float dy[7*5];
            for (size_t n = 0; n < 7; ++n) {
                double maximum = output[n*5], sum = 0;
                for (size_t o = 1; o < 5; ++o) if (output[n*5+o] > maximum) maximum = output[n*5+o];
                for (size_t o = 0; o < 5; ++o) sum += exp((double)output[n*5+o]-maximum);
                for (size_t o = 0; o < 5; ++o) dy[n*5+o] = (float)((-1.0/7)*
                    ((o == labels[n] ? 1.0 : 0.0)-exp((double)output[n*5+o]-maximum)/sum));
            }
            for (size_t o = 0; o < 5; ++o) for (size_t i = 0; i < columns; ++i) {
                float value = nya_llm_tensor_value(&w,o*columns+i);
                for (size_t n = 0; n < 7; ++n) expected[n*columns+i] += dy[n*5+o]*value;
            }
            REQUIRE(memcmp(expected,nya_train_parameter_gradient(p),7*columns*sizeof(float)) == 0);
            nya_train_graph_free(g);
        }
        nya_train_parameter_free(p);
    }
    return 0;
}

static int f32_cancellation(void)
{
    const float weights[] = {100000000.0f,1.0f,-100000000.0f};
    const float inputs[] = {1,1,1,1,1,1,1,1,1,1,1,1,1,1,1};
    nya_llm_tensor w = {0}; w.type = NYA_LLM_TENSOR_F32; w.dimension_count = 2;
    w.dimensions[0] = 3; w.dimensions[1] = 1;
    w.data = (const unsigned char *)weights; w.data_size = sizeof(weights);
    float reference = 0; nya_llm_matvec(NULL,&reference,&w,inputs,3,1);
    REQUIRE(reference == 1.0f);
    nya_train_graph *g = nya_train_graph_create(4096);
    nya_train_tensor *a = nya_train_input(g,5,3,inputs);
    nya_train_tensor *mapped = nya_train_linear_mapped(a,&w);
    nya_train_tensor *dense = nya_train_linear(a,nya_train_input(g,1,3,weights));
    REQUIRE(mapped != NULL && dense != NULL);
    for (size_t i = 0; i < 5; ++i) REQUIRE(nya_train_data(mapped)[i] == 1.0f && nya_train_data(dense)[i] == 1.0f);
    nya_train_graph_free(g); return 0;
}

/* Large enough to dispatch workers, with uneven row/quant-block partitions
   and a three-token tail. Compare complete multi-step trajectories bitwise. */
static int parallel_matrices(void)
{
    const unsigned types[] = {0,1,30,2,8,12,14,0};
    const size_t sizes[] = {1024,512,512,144,272,144,210,1024};
    const size_t thread_counts[] = {2,3,6};
    const size_t tokens = 7, columns = 768, rows = 257;
    float *input = malloc(tokens*columns*sizeof(float));
    unsigned char *bytes = malloc(rows*columns*4);
    float *output = malloc(tokens*rows*sizeof(float));
    REQUIRE(input && bytes && output);
    uint32_t labels[7] = {0,1,2,3,4,5,6};
    for (size_t i = 0; i < tokens*columns; ++i) input[i] = (float)((int)(i*7%17)-8)/128;
    REQUIRE(nya_train_executor_create(65) == NULL);
    REQUIRE(nya_train_executor_threads(NULL) == 1);
    for (size_t nt = 0; nt < 3; ++nt) {
        nya_train_executor *executor = nya_train_executor_create(thread_counts[nt]); REQUIRE(executor);
        REQUIRE(nya_train_executor_threads(executor) == thread_counts[nt]);
        for (size_t format = 0; format < 8; ++format) {
            unsigned type = types[format]; size_t rb = sizes[format]*(columns/256);
            for (size_t i = 0; i < rows*rb; ++i) bytes[i] = (unsigned char)(i*13+7);
            for (size_t row = 0; row < rows; ++row) {
                unsigned char *w = bytes+row*rb;
                if (type == 0) for (size_t j = 0; j < columns; ++j) {
                    float v = (float)((int)((row*3+j)%19)-9)/32; memcpy(w+4*j,&v,4);
                }
                else if (type == 1 || type == 30) for (size_t j = 0; j < columns; ++j) {
                    w[2*j] = (unsigned char)(j%64); w[2*j+1] = type == 1 ? 0x30 : 0x3e;
                }
                else if (type == 2 || type == 8) for (size_t b = 0; b < columns/32; ++b) {
                    w[b*(type == 2 ? 18 : 34)] = 0; w[b*(type == 2 ? 18 : 34)+1] = 0x30;
                }
                else for (size_t b = 0; b < columns/256; ++b) {
                    if (type == 12) { w[b*144] = w[b*144+2] = 0; w[b*144+1] = w[b*144+3] = 0x30; }
                    else { w[b*210+208] = 0; w[b*210+209] = 0x30; }
                }
            }
            nya_llm_tensor mapped = {0}; mapped.type = type; mapped.dimension_count = 2;
            mapped.dimensions[0] = columns; mapped.dimensions[1] = rows;
            mapped.data = bytes; mapped.data_size = rows*rb;
            nya_train_parameter *p[2][2]; nya_train_adamw optimizer[2];
            for (size_t run = 0; run < 2; ++run) {
                p[run][0] = nya_train_parameter_create(tokens,columns,input);
                p[run][1] = format == 7 ? nya_train_parameter_create(rows,columns,(const float *)(const void *)bytes) : NULL;
                REQUIRE(p[run][0] && (format != 7 || p[run][1]));
                nya_train_adamw_defaults(&optimizer[run]);
            }
            float loss_value = 0;
            for (size_t step = 0; step < 3; ++step) {
                for (size_t run = 0; run < 2; ++run) {
                    nya_train_zero_grad(p[run][0]); if (p[run][1]) nya_train_zero_grad(p[run][1]);
                    nya_train_graph *g = nya_train_graph_create_with_executor(16*1024*1024,run ? executor : NULL);
                    nya_train_tensor *x = nya_train_leaf(g,p[run][0]);
                    nya_train_tensor *y = format == 7 ? nya_train_linear(x,nya_train_leaf(g,p[run][1])) : nya_train_linear_mapped(x,&mapped);
                    REQUIRE(y != NULL);
                    if (!run) memcpy(output,nya_train_data(y),tokens*rows*sizeof(float));
                    else REQUIRE(memcmp(output,nya_train_data(y),tokens*rows*sizeof(float)) == 0);
                    nya_train_tensor *loss = nya_train_cross_entropy(y,labels,NULL,tokens); REQUIRE(loss);
                    if (!run) loss_value = nya_train_data(loss)[0];
                    else REQUIRE(loss_value == nya_train_data(loss)[0]);
                    if (step == 2 && run && format != 7) {
                        /* Workers must remain reusable when the graph has no
                           space for decoded scratch and backward goes scalar. */
                        size_t budget = nya_train_memory_used(g);
                        nya_train_graph_free(g);
                        g = nya_train_graph_create_with_executor(budget,executor);
                        loss = nya_train_cross_entropy(nya_train_linear_mapped(nya_train_leaf(g,p[run][0]),&mapped),labels,NULL,tokens);
                        REQUIRE(loss && nya_train_data(loss)[0] == loss_value);
                        REQUIRE(nya_train_memory_used(g) == budget);
                    }
                    REQUIRE(nya_train_backward(loss) == 0);
                    if (run) {
                        REQUIRE(memcmp(nya_train_parameter_gradient(p[0][0]),nya_train_parameter_gradient(p[1][0]),tokens*columns*sizeof(float)) == 0);
                        if (format == 7) REQUIRE(memcmp(nya_train_parameter_gradient(p[0][1]),nya_train_parameter_gradient(p[1][1]),rows*columns*sizeof(float)) == 0);
                    }
                    char error[160];
                    REQUIRE(nya_train_adamw_step(&optimizer[run],p[run],format == 7 ? 2 : 1,error,sizeof(error)) == 0);
                    nya_train_graph_free(g);
                }
                REQUIRE(memcmp(nya_train_parameter_data(p[0][0]),nya_train_parameter_data(p[1][0]),tokens*columns*sizeof(float)) == 0);
                if (format == 7) REQUIRE(memcmp(nya_train_parameter_data(p[0][1]),nya_train_parameter_data(p[1][1]),rows*columns*sizeof(float)) == 0);
            }
            for (size_t run = 0; run < 2; ++run) { nya_train_parameter_free(p[run][0]); nya_train_parameter_free(p[run][1]); }
        }
        nya_train_executor_free(executor);
    }
    free(input); free(bytes); free(output); return 0;
}

static int independent_executor(void *argument)
{
    int *status = argument;
    *status = parallel_matrices();
    return *status;
}

static int executor_lifecycle(void)
{
    for (size_t threads = 1; threads <= 6; ++threads) {
        size_t previous = 0;
        for (size_t worker = 0; worker < threads; ++worker) {
            size_t begin, end;
            nya_train_partition(SIZE_MAX,worker,threads,&begin,&end);
            REQUIRE(begin == previous && end >= begin); previous = end;
        }
        REQUIRE(previous == SIZE_MAX);
    }
    /* Each controller owns its executor and parameters. Concurrent lifetimes
       expose accidental process-global jobs and missed shutdown wakeups. */
    nya_thread thread; int status = -1;
    REQUIRE(nya_thread_create(&thread,independent_executor,&status) == 0);
    int local = parallel_matrices();
    REQUIRE(nya_thread_join(&thread) == 0);
    REQUIRE(local == 0 && status == 0);
    for (size_t i = 0; i < 8; ++i) {
        nya_train_executor *e = nya_train_executor_create(i%3+1); REQUIRE(e);
        REQUIRE(nya_train_graph_create_with_executor(1,e) == NULL);
        nya_train_graph *g = nya_train_graph_create_with_executor(1024,e); REQUIRE(g);
        nya_train_graph_free(g); nya_train_executor_free(e);
    }
    return 0;
}

int main(void)
{
    return executor_lifecycle() != 0 || f32_cancellation() != 0 || mapped_gradient_oracle() != 0 || linear_oracle() != 0 || attention_tile_gradients() != 0 || gradients() != 0 || attention_gradients() != 0 || frozen_matrix() != 0 ||
        lora() != 0 || checkpoint() != 0 || invalid_inputs() != 0;
}
