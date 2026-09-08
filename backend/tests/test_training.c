#include "training.h"
#include "training_internal.h"

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

int main(void)
{
    return gradients() != 0 || attention_gradients() != 0 || frozen_matrix() != 0 ||
        lora() != 0 || checkpoint() != 0 || invalid_inputs() != 0;
}
