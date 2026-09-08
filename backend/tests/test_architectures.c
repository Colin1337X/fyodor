/* Numerical integration tests deliberately access the private token step. This
   permits comparison of every logit, rather than only a greedy token decision. */
#include "../core/llm_cpu.c"
#include "gemma_reference.h"
#include "../core/multimodal.h"
#include "projector_reference.h"
#include "pretraining.h"
#include "model.h"

typedef struct fixture_tensor {
    char name[128];
    uint32_t dims[3], ndims;
    size_t count, offset;
    float *values;
} fixture_tensor;

static fixture_tensor tensors[160];
static size_t tensor_count;
static int fixture_assistant;
static int fixture_tied;

static int put_u32(FILE *f, uint32_t v)
{
    unsigned char b[4];
    for (size_t i = 0; i < 4; ++i) b[i] = (unsigned char)(v >> (8 * i));
    return fwrite(b, 1, 4, f) == 4 ? 0 : -1;
}
static int put_u64(FILE *f, uint64_t v)
{
    unsigned char b[8];
    for (size_t i = 0; i < 8; ++i) b[i] = (unsigned char)(v >> (8 * i));
    return fwrite(b, 1, 8, f) == 8 ? 0 : -1;
}
static int put_string(FILE *f, const char *v)
{
    char key[160];
    if (fixture_assistant && strncmp(v, "gemma4.", 7) == 0) {
        snprintf(key, sizeof(key), "gemma4-assistant.%s", v + 7);
        v = key;
    }
    size_t n = strlen(v);
    return put_u64(f, n) == 0 && fwrite(v, 1, n, f) == n ? 0 : -1;
}
static int put_float(FILE *f, float v)
{
    uint32_t bits;
    memcpy(&bits, &v, 4);
    return put_u32(f, bits);
}
static int meta_u32(FILE *f, const char *name, uint32_t v)
{
    return put_string(f, name) == 0 && put_u32(f, 4) == 0 && put_u32(f, v) == 0 ? 0 : -1;
}
static int meta_float(FILE *f, const char *name, float v)
{
    return put_string(f, name) == 0 && put_u32(f, 6) == 0 && put_float(f, v) == 0 ? 0 : -1;
}
static int meta_text(FILE *f, const char *name, const char *v)
{
    return put_string(f, name) == 0 && put_u32(f, 8) == 0 && put_string(f, v) == 0 ? 0 : -1;
}
static int meta_array(FILE *f, const char *name, uint32_t type, size_t n)
{
    return put_string(f, name) == 0 && put_u32(f, 9) == 0 && put_u32(f, type) == 0 && put_u64(f, n) == 0 ? 0 : -1;
}

/* Nonzero, asymmetric weights exercise attention, head layout, branch norms,
   tied K/V, PLE, expert selection/scaling, and changing token positions. */
static int add_tensor(const char *name, uint32_t columns, uint32_t rows, uint32_t experts, int norm)
{
    fixture_tensor *t;
    size_t salt = tensor_count + 1;
    if (tensor_count >= sizeof(tensors) / sizeof(tensors[0])) return -1;
    t = &tensors[tensor_count++];
    memset(t, 0, sizeof(*t));
    snprintf(t->name, sizeof(t->name), "%s", name);
    t->dims[0] = columns; t->dims[1] = rows; t->dims[2] = experts;
    t->ndims = experts != 0 ? 3 : rows != 0 ? 2 : 1;
    t->count = (size_t)columns * (rows == 0 ? 1 : rows) * (experts == 0 ? 1 : experts);
    t->values = (float *)malloc(t->count * sizeof(float));
    if (t->values == NULL) return -1;
    for (size_t i = 0; i < t->count; ++i) t->values[i] = norm ?
        1.0f + (float)((int)((i + salt) % 7) - 3) / 20.0f :
        (float)((int)((i * 17 + salt * 13) % 41) - 20) / 80.0f;
    return 0;
}

static void free_fixture(void)
{
    for (size_t i = 0; i < tensor_count; ++i) free(tensors[i].values);
    tensor_count = 0;
}

static int make_architecture(const char *path, int ple, int moe, int fused, int assistant, uint64_t *file_size)
{
    FILE *f = NULL;
    char name[128];
    size_t offset = 0;
    int result = -1;
    static const char *pieces[] = {"<unk>", "<bos>", "<eos>", "a", "b", "ab", "\n", "<turn|>"};
    free_fixture();
    fixture_assistant = assistant;
#define ADD(name_, x, y, z, norm) do { if (add_tensor(name_, x, y, z, norm) != 0) goto done; } while (0)
#define BLK(suffix, x, y, z, norm) do { snprintf(name, sizeof(name), "blk.%u." suffix, layer); ADD(name, x, y, z, norm); } while (0)
    ADD("token_embd.weight", 8, 8, 0, 0);
    ADD("output_norm.weight", 8, 0, 0, 1);
    if (!fixture_tied) ADD("output.weight", 8, 8, 0, 0);
    ADD("rope_freqs.weight", 4, 0, 0, 1);
    tensors[tensor_count-1].values[0] = 1.0f;
    for (size_t i = 1; i < 4; ++i) tensors[tensor_count-1].values[i] = 1e30f;
    if (assistant) {
        ADD("nextn.pre_projection.weight", 16, 8, 0, 0);
        ADD("nextn.post_projection.weight", 8, 8, 0, 0);
    }
    if (ple) {
        ADD("per_layer_token_embd.weight", 16, 8, 0, 0);
        ADD("per_layer_model_proj.weight", 8, 16, 0, 0);
        ADD("per_layer_proj_norm.weight", 4, 0, 0, 1);
    }
    for (uint32_t layer = 0; layer < 4; ++layer) {
        uint32_t hd = layer % 2 == 0 ? 4 : 8;
        BLK("attn_norm.weight", 8, 0, 0, 1);
        BLK("attn_q.weight", 8, hd * 2, 0, 0);
        BLK("attn_q_norm.weight", hd, 0, 0, 1);
        BLK("attn_output.weight", hd * 2, 8, 0, 0);
        BLK("post_attention_norm.weight", 8, 0, 0, 1);
        if (layer < 2 && !assistant) {
            BLK("attn_k.weight", 8, hd, 0, 0);
            if (layer == 0) BLK("attn_v.weight", 8, hd, 0, 0);
            BLK("attn_k_norm.weight", hd, 0, 0, 1);
        }
        BLK("ffn_norm.weight", 8, 0, 0, 1);
        BLK("ffn_gate.weight", 8, 12, 0, 0);
        BLK("ffn_up.weight", 8, 12, 0, 0);
        BLK("ffn_down.weight", 12, 8, 0, 0);
        BLK("post_ffw_norm.weight", 8, 0, 0, 1);
        BLK("layer_output_scale.weight", 1, 0, 0, 1);
        if (moe) {
            BLK("ffn_gate_inp.weight", 8, 3, 0, 0);
            BLK("ffn_gate_inp.scale", 8, 0, 0, 1);
            BLK("ffn_down_exps.scale", 3, 0, 0, 1);
            BLK("pre_ffw_norm_2.weight", 8, 0, 0, 1);
            BLK("post_ffw_norm_1.weight", 8, 0, 0, 1);
            BLK("post_ffw_norm_2.weight", 8, 0, 0, 1);
            if (fused) BLK("ffn_gate_up_exps.weight", 8, 12, 3, 0);
            else {
                BLK("ffn_gate_exps.weight", 8, 6, 3, 0);
                BLK("ffn_up_exps.weight", 8, 6, 3, 0);
            }
            BLK("ffn_down_exps.weight", 6, 8, 3, 0);
        }
        if (ple) {
            BLK("inp_gate.weight", 8, 4, 0, 0);
            BLK("proj.weight", 4, 8, 0, 0);
            BLK("post_norm.weight", 8, 0, 0, 1);
        }
    }
#undef BLK
#undef ADD
    f = fopen(path, "wb");
    if (f == NULL) goto done;
#define W(expr) do { if ((expr) != 0) goto done; } while (0)
    if (fwrite("GGUF", 1, 4, f) != 4) goto done;
    W(put_u32(f, 3)); W(put_u64(f, tensor_count)); W(put_u64(f, assistant ? 33 : 31));
    W(meta_u32(f,"general.file_type",0));
    W(meta_text(f, "general.architecture", assistant ? "gemma4-assistant" : "gemma4"));
    W(meta_text(f, "tokenizer.ggml.model", "gemma4"));
    W(meta_u32(f, "gemma4.context_length", 16));
    W(meta_u32(f, "gemma4.embedding_length", 8));
    W(meta_u32(f, "gemma4.block_count", 4));
    W(meta_u32(f, "gemma4.attention.head_count", 2));
    W(meta_u32(f, "gemma4.attention.head_count_kv", 1));
    W(meta_u32(f, "gemma4.feed_forward_length", 12));
    W(meta_u32(f, "gemma4.attention.key_length", 8));
    W(meta_u32(f, "gemma4.attention.value_length", 8));
    W(meta_u32(f, "gemma4.attention.key_length_swa", 4));
    W(meta_u32(f, "gemma4.attention.value_length_swa", 4));
    W(meta_u32(f, "gemma4.rope.dimension_count", 8));
    W(meta_u32(f, "gemma4.rope.dimension_count_swa", 4));
    W(meta_u32(f, "gemma4.attention.sliding_window", 2));
    W(meta_u32(f, "gemma4.attention.shared_kv_layers", assistant ? 4 : 2));
    if (assistant) {
        W(meta_u32(f, "gemma4.embedding_length_out", 8));
        W(meta_u32(f, "gemma4.nextn_predict_layers", 4));
    }
    W(meta_array(f, "gemma4.attention.sliding_window_pattern", 7, 4));
    for (size_t i = 0; i < 4; ++i) if (fputc(i % 2 == 0, f) == EOF) goto done;
    W(meta_u32(f, "gemma4.embedding_length_per_layer_input", ple ? 4 : 0));
    W(meta_u32(f, "gemma4.expert_count", moe ? 3 : 0));
    W(meta_u32(f, "gemma4.expert_used_count", moe ? 2 : 0));
    W(meta_u32(f, "gemma4.expert_feed_forward_length", moe ? 6 : 0));
    W(meta_float(f, "gemma4.final_logit_softcapping", assistant ? 0.0f : 5.0f));
    W(meta_u32(f, "tokenizer.ggml.bos_token_id", 1));
    W(meta_u32(f, "tokenizer.ggml.eos_token_id", 2));
    W(meta_u32(f, "tokenizer.ggml.unknown_token_id", 0));
    W(put_string(f, "tokenizer.ggml.add_space_prefix")); W(put_u32(f, 7));
    if (fputc(0, f) == EOF) goto done;
    W(meta_array(f, "tokenizer.ggml.tokens", 8, 8));
    for (size_t i = 0; i < 8; ++i) W(put_string(f, pieces[i]));
    W(meta_array(f, "tokenizer.ggml.scores", 6, 8));
    for (size_t i = 0; i < 8; ++i) W(put_float(f, 0.0f));
    W(meta_array(f, "tokenizer.ggml.token_type", 5, 8));
    for (size_t i = 0; i < 8; ++i) W(put_u32(f, i == 0 ? 2 : i == 1 || i == 2 || i == 7 ? 3 : 1));
    W(meta_array(f, "tokenizer.ggml.merges", 8, 1)); W(put_string(f, "a b"));
    for (size_t i = 0; i < tensor_count; ++i) {
        fixture_tensor *t = &tensors[i];
        t->offset = offset;
        offset = (offset + t->count * 4 + 31) / 32 * 32;
        W(put_string(f, t->name)); W(put_u32(f, t->ndims));
        for (size_t j = 0; j < t->ndims; ++j) W(put_u64(f, t->dims[j]));
        W(put_u32(f, 0)); W(put_u64(f, t->offset));
    }
    while (ftell(f) >= 0 && ftell(f) % 32 != 0) if (fputc(0, f) == EOF) goto done;
    for (size_t i = 0; i < tensor_count; ++i) {
        fixture_tensor *t = &tensors[i];
        for (size_t j = 0; j < t->count; ++j) W(put_float(f, t->values[j]));
        for (size_t n = t->count * 4; n % 32 != 0; ++n) if (fputc(0, f) == EOF) goto done;
    }
    if (ftell(f) < 0) goto done;
    *file_size = (uint64_t)ftell(f);
    result = 0;
done:
    if (f != NULL && fclose(f) != 0) result = -1;
    free_fixture();
    return result;
#undef W
}

/* Construct K-quants from logical scales and integer values, independently of
   the decoder's indexing. Two rows of two blocks catch both row and block
   strides. Exactly representable powers of two keep scalar comparisons exact. */
static int check_k_quants(void)
{
    unsigned char storage[1 + 4 * 210];
    float expected[1024], input[512], output[2];
    for (size_t i = 0; i < 512; ++i) input[i] = (float)((int)(i % 23) - 11) / 16.0f;
    for (int kind = 0; kind < 2; ++kind) {
        size_t stride = kind == 0 ? 144 : 210;
        nya_llm_tensor tensor = {0};
        memset(storage, 0, sizeof(storage));
        /* Deliberately unaligned byte storage also exercises the mapped-file
           load contract under alignment and undefined-behavior sanitizers. */
        tensor.data = storage + 1;
        tensor.type = kind == 0 ? NYA_LLM_TENSOR_Q4_K : NYA_LLM_TENSOR_Q6_K;
        for (size_t b = 0; b < 4; ++b) {
            unsigned char *p = storage + 1 + b * stride;
            if (kind == 0) {
                unsigned int scales[8], minima[8];
                p[1] = 0x38; p[3] = 0x34; /* d=1/2, dmin=1/4 */
                for (size_t g = 0; g < 8; ++g) {
                    scales[g] = (unsigned int)((g * 11 + b * 7) % 64);
                    minima[g] = (unsigned int)((g * 17 + b * 13) % 64);
                }
                for (size_t g = 0; g < 4; ++g) {
                    p[4 + g] = (unsigned char)(scales[g] | ((scales[g + 4] >> 4) << 6));
                    p[8 + g] = (unsigned char)(minima[g] | ((minima[g + 4] >> 4) << 6));
                    p[12 + g] = (unsigned char)((scales[g + 4] & 15) | ((minima[g + 4] & 15) << 4));
                }
                for (size_t pair = 0; pair < 4; ++pair) for (size_t j = 0; j < 32; ++j) {
                    unsigned int lo = (unsigned int)((j * 3 + pair + b) % 16);
                    unsigned int hi = (unsigned int)((j * 7 + pair * 3 + b) % 16);
                    p[16 + pair * 32 + j] = (unsigned char)(lo | (hi << 4));
                    expected[b * 256 + pair * 64 + j] = 0.5f * (float)(scales[pair * 2] * lo) - 0.25f * (float)minima[pair * 2];
                    expected[b * 256 + pair * 64 + 32 + j] = 0.5f * (float)(scales[pair * 2 + 1] * hi) - 0.25f * (float)minima[pair * 2 + 1];
                }
            } else {
                int scales[16];
                p[209] = 0x30; /* d=1/8 */
                for (size_t g = 0; g < 16; ++g) {
                    scales[g] = (int)((g * 37 + b * 53) % 256) - 128;
                    p[192 + g] = (unsigned char)scales[g];
                }
                for (size_t half = 0; half < 2; ++half) for (size_t j = 0; j < 32; ++j) {
                    unsigned int q[4];
                    for (size_t k = 0; k < 4; ++k) {
                        q[k] = (unsigned int)((j * 13 + k * 19 + half * 31 + b * 7) % 64);
                        size_t n = half * 128 + k * 32 + j;
                        expected[b * 256 + n] = 0.125f * (float)scales[n / 16] * (float)((int)q[k] - 32);
                    }
                    p[half * 64 + j] = (unsigned char)((q[0] & 15) | ((q[2] & 15) << 4));
                    p[half * 64 + 32 + j] = (unsigned char)((q[1] & 15) | ((q[3] & 15) << 4));
                    p[128 + half * 32 + j] = (unsigned char)((q[0] >> 4) | ((q[1] >> 4) << 2) | ((q[2] >> 4) << 4) | ((q[3] >> 4) << 6));
                }
            }
        }
        for (size_t i = 0; i < 1024; ++i) if (nya_llm_tensor_value(&tensor, i) != expected[i]) {
            fprintf(stderr, "K-quant %d scalar %zu differs\n", kind, i); return -1;
        }
        nya_llm_matvec(NULL, output, &tensor, input, 512, 2);
        for (size_t row = 0; row < 2; ++row) {
            double dot = 0.0;
            for (size_t j = 0; j < 512; ++j) dot += (double)expected[row * 512 + j] * input[j];
            if ((double)output[row] != dot) { fprintf(stderr, "K-quant matvec differs\n"); return -1; }
        }
        nya_compute_context *compute = nya_compute_create();
        if (compute != NULL && strcmp(nya_compute_name(compute), "cuda") == 0) {
            if (nya_compute_matvec_typed(compute, tensor.data, 2, 512, tensor.type, input, output) != 0) return -1;
            for (size_t row = 0; row < 2; ++row) {
                double dot = 0.0;
                for (size_t j = 0; j < 512; ++j) dot += (double)expected[row * 512 + j] * input[j];
                if ((double)output[row] != dot) { fprintf(stderr, "CUDA K-quant matvec differs\n"); return -1; }
            }
        }
        nya_compute_free(compute);
    }
    return 0;
}

/* Tiny asymmetric projector exercises CHW packing, all three affine LayerNorms,
   both position axes and the two different modality projections. */
static int check_projector(const char *prefix)
{
    char path[1200], error[512] = {0};
    FILE *f = NULL;
    size_t offset = 0;
    int result = -1;
    nya_llm_context *c = NULL;
    nya_model model = {0};
    nya_execution_response response = {0};
    nya_execution_request request = {0};
    float pixels[48], packed[56], audio[15];
    int64_t shape[3] = {4, 4, 3};
    free_fixture(); fixture_assistant = 0;
#define CHECK(v) do { if (!(v)) { fprintf(stderr, "Projector check line %d: %s\n", __LINE__, error); goto done; } } while (0)
    CHECK(snprintf(path, sizeof(path), "%s.projector", prefix) > 0);
    CHECK(add_tensor("mm.a.input_projection.weight", 5, 8, 0, 0) == 0);
    CHECK(add_tensor("mm.input_projection.weight", 8, 8, 0, 0) == 0);
    CHECK(add_tensor("v.patch_embd.bias", 8, 0, 0, 0) == 0);
    CHECK(add_tensor("v.patch_embd.weight", 12, 8, 0, 0) == 0);
    for (size_t i = 0; i < 3; ++i) {
        char name[64];
        uint32_t width = i == 0 ? 12 : 8;
        snprintf(name, sizeof(name), "v.patch_norm.%zu.bias", i + 1);
        CHECK(add_tensor(name, width, 0, 0, 0) == 0);
        snprintf(name, sizeof(name), "v.patch_norm.%zu.weight", i + 1);
        CHECK(add_tensor(name, width, 0, 0, 1) == 0);
    }
    CHECK(add_tensor("v.position_embd.weight", 8, 4, 2, 0) == 0);
    f = fopen(path, "wb"); CHECK(f != NULL);
    CHECK(fwrite("GGUF", 1, 4, f) == 4 && put_u32(f, 3) == 0 && put_u64(f, tensor_count) == 0 && put_u64(f, 9) == 0);
    CHECK(meta_text(f, "general.architecture", "clip") == 0);
    CHECK(meta_text(f, "clip.vision.projector_type", "gemma4uv") == 0);
    CHECK(meta_text(f, "clip.audio.projector_type", "gemma4ua") == 0);
    CHECK(meta_u32(f, "clip.vision.embedding_length", 8) == 0 && meta_u32(f, "clip.audio.embedding_length", 5) == 0);
    CHECK(meta_u32(f, "clip.vision.projection_dim", 8) == 0 && meta_u32(f, "clip.audio.projection_dim", 8) == 0);
    CHECK(meta_float(f, "clip.vision.attention.layer_norm_epsilon", 1e-6f) == 0);
    CHECK(meta_float(f, "clip.audio.attention.layer_norm_epsilon", 1e-6f) == 0);
    for (size_t i = 0; i < tensor_count; ++i) {
        fixture_tensor *t = &tensors[i];
        t->offset = offset; offset = (offset + t->count * 4 + 31) / 32 * 32;
        CHECK(put_string(f, t->name) == 0 && put_u32(f, t->ndims) == 0);
        for (size_t j = 0; j < t->ndims; ++j) CHECK(put_u64(f, t->dims[j]) == 0);
        CHECK(put_u32(f, 0) == 0 && put_u64(f, t->offset) == 0);
    }
    while (ftell(f) >= 0 && ftell(f) % 32 != 0) CHECK(fputc(0, f) != EOF);
    for (size_t i = 0; i < tensor_count; ++i) {
        fixture_tensor *t = &tensors[i];
        for (size_t j = 0; j < t->count; ++j) CHECK(put_float(f, t->values[j]) == 0);
        for (size_t n = t->count * 4; n % 32 != 0; ++n) CHECK(fputc(0, f) != EOF);
    }
    CHECK(ftell(f) > 0); model.file_size = (uint64_t)ftell(f);
    { int close_result = fclose(f); f = NULL; CHECK(close_result == 0); }
    CHECK(nya_llm_load_projector(path, model.file_size, &c, error, sizeof(error)) == 0);
    model.execution_context = c; model.inference_supported = 1; model.format = NYA_FORMAT_GGUF;
    memcpy(model.architecture, "clip", 5);
    for (size_t i = 0; i < 48; ++i) pixels[i] = (float)((i * 7) % 31) / 32.0f;
    for (size_t t = 0; t < 4; ++t) {
        packed[t * 14] = (float)(t % 2); packed[t * 14 + 1] = (float)(t / 2);
        for (size_t ch = 0; ch < 3; ++ch) for (size_t y = 0; y < 2; ++y) for (size_t x = 0; x < 2; ++x)
            packed[t * 14 + 2 + ch * 4 + y * 2 + x] = pixels[((t / 2 * 2 + y) * 4 + t % 2 * 2 + x) * 3 + ch];
    }
    request.input_name = "image_rgb"; request.output_name = "embeddings";
    request.input_type = NYA_TENSOR_FLOAT32; request.input_rank = 3; request.input_shape = shape;
    request.input_data = pixels; request.input_data_size = sizeof(pixels); request.max_output_bytes = 4096;
    for (int variant = 0; variant < 3; ++variant) {
        if (variant == 1) {
            request.input_name = "image_patches"; request.input_rank = 2; shape[0] = 4; shape[1] = 14;
            request.input_data = packed; request.input_data_size = sizeof(packed);
        } else if (variant == 2) {
            for (size_t i = 0; i < 15; ++i) audio[i] = (float)((int)(i * 13 % 23) - 11) / 8.0f;
            request.input_name = "audio_features"; shape[0] = 3; shape[1] = 5;
            request.input_data = audio; request.input_data_size = sizeof(audio);
        }
        CHECK(nya_execution_run(&model, &request, &response, error, sizeof(error)) == 0);
        CHECK(response.output_rank == 2 && response.output_shape[0] == (variant == 2 ? 3 : 4) && response.output_shape[1] == 8);
        for (size_t i = 0; i < response.output_data_size / 4; ++i) {
            double reference = variant == 2 ? projector_audio_reference[i] : projector_image_reference[i];
            CHECK(fabs((double)((float *)response.output_data)[i] - reference) < 2e-6);
        }
        nya_execution_response_free(&response);
    }
    /* Errors must leave output empty and safe to free. */
    audio[0] = NAN;
    CHECK(nya_execution_run(&model, &request, &response, error, sizeof(error)) != 0 && response.output_data == NULL);
    audio[0] = 0; request.max_output_bytes = 1;
    CHECK(nya_execution_run(&model, &request, &response, error, sizeof(error)) != 0 && response.output_data == NULL);
    request.max_output_bytes = 4096; request.input_name = "image_patches";
    request.input_data = packed; request.input_data_size = sizeof(packed); shape[0] = 4; shape[1] = 14;
    packed[0] = 4;
    CHECK(nya_execution_run(&model, &request, &response, error, sizeof(error)) != 0 && response.output_data == NULL);
    packed[0] = 0.5f;
    CHECK(nya_execution_run(&model, &request, &response, error, sizeof(error)) != 0 && response.output_data == NULL);
    result = 0;
done:
    if (f != NULL) fclose(f);
    nya_execution_response_free(&response);
    nya_llm_free(c); free_fixture();
    return result;
#undef CHECK
}

static int check_gemma_training(nya_llm_context *c, const char *path)
{
    char error[512] = {0}; const uint32_t ids[] = {1,3,4,5}, labels[] = {3,4,5,3};
    nya_model model = {0}; model.generation_supported = 1; model.generation_context = c;
    for (size_t rank = 0; rank <= 2; rank += 2) {
        nya_train_decoder *train = nya_train_decoder_from_model(&model,rank,4,16*1024*1024,error,sizeof(error));
        if (train == NULL) { fprintf(stderr,"Gemma train import: %s\n",error); return 1; }
        size_t pc; nya_train_parameter *const *parameters = nya_train_decoder_parameters(train,&pc);
        nya_train_adamw optimizer; nya_train_adamw_defaults(&optimizer); optimizer.learning_rate = 0.003f;
        float initial = 0, final = 0;
        for (size_t step = 0; step <= 40; ++step) {
            nya_train_graph *g = nya_train_graph_create(16*1024*1024);
            nya_train_tensor *logits = nya_train_decoder_forward(train,g,ids,4);
            if (logits == NULL) { fprintf(stderr,"Gemma train forward: %s\n",nya_train_error(g)); return 1; }
            if (step == 0 || step == 40) {
                nya_llm_context *compare = c, *exported = NULL;
                if (step != 0) {
                    char exported_path[1400]; snprintf(exported_path,sizeof(exported_path),"%s.train-%zu.gguf",path,rank);
                    FILE *f = fopen(exported_path,"wb");
                    if (f == NULL || nya_train_decoder_export(train,f,error,sizeof(error)) != 0) return 1;
                    long size = ftell(f); if (fclose(f) != 0 || size < 0 || nya_llm_load(exported_path,(uint64_t)size,&exported,error,sizeof(error)) != 0) return 1;
                    if (exported->file_type_begin != 0 || exported->gguf_metadata_count+1 != c->gguf_metadata_count) return 1;
                    compare = exported;
                }
                nya_llm_run_state state;
                if (nya_llm_state_create(compare,4,&state,error,sizeof(error)) != 0) return 1;
                for (size_t row = 0; row < 4; ++row) {
                    if (nya_llm_forward(compare,&state,ids[row],row,1) != 0) return 1;
                    for (size_t col = 0; col < 8; ++col) if (fabs((double)state.logits[col]-nya_train_data(logits)[row*8+col]) > 8e-5) {
                        fprintf(stderr,"Gemma training logit mismatch rank=%zu step=%zu row=%zu col=%zu\n",rank,step,row,col); return 1;
                    }
                }
                nya_llm_state_free(&state); nya_llm_free(exported);
            }
            nya_train_tensor *loss = nya_train_cross_entropy(logits,labels,NULL,4);
            if (loss == NULL) return 1;
            final = nya_train_data(loss)[0]; if (step == 0) initial = final;
            if (step < 40) {
                for (size_t j = 0; j < pc; ++j) nya_train_zero_grad(parameters[j]);
                if (nya_train_backward(loss) != 0) return 1;
                /* Independent central differences sample every parameter,
                   including shared KV, scalar scales and PLE paths. */
                if (step == 1) for (size_t j = 0; j < pc; ++j) {
                    float *data = nya_train_parameter_data(parameters[j]), original = data[0], values[2];
                    float analytic = nya_train_parameter_gradient(parameters[j])[0];
                    for (size_t sign = 0; sign < 2; ++sign) {
                        nya_train_graph *probe = nya_train_graph_create(16*1024*1024); data[0] = original+(sign == 0 ? 0.0005f : -0.0005f);
                        nya_train_tensor *value = nya_train_cross_entropy(nya_train_decoder_forward(train,probe,ids,4),labels,NULL,4);
                        if (value == NULL) return 1;
                        values[sign] = nya_train_data(value)[0]; nya_train_graph_free(probe);
                    }
                    data[0] = original;
                    if (fabs(((double)values[0]-values[1])/0.001-analytic) > 0.01*(1+fabs(analytic))) {
                        fprintf(stderr,"Gemma parameter gradient mismatch %zu: %.8g vs %.8g\n",j,(double)analytic,((double)values[0]-values[1])/0.001); return 1;
                    }
                }
                nya_train_graph_free(g);
                if (nya_train_adamw_step(&optimizer,parameters,pc,error,sizeof(error)) != 0) return 1;
            } else nya_train_graph_free(g);
        }
        if (!(final < initial*0.8f)) { fprintf(stderr,"Gemma training did not improve: %g -> %g\n",(double)initial,(double)final); return 1; }
        nya_train_decoder_free(train);
    }
    return 0;
}

int main(int argc, char **argv)
{
    const uint32_t prompt[] = {1, 3, 4, 5};
    if (argc != 2) return 2;
    if (check_k_quants() != 0) return 1;
    if (check_projector(argv[1]) != 0) return 1;
    {
        char path[1200], error[256]; uint64_t size; nya_llm_context *c;
        if (strlen(argv[1]) > 1100) return 1;
        snprintf(path,sizeof(path),"%s.tied",argv[1]); fixture_tied = 1;
        if (make_architecture(path,1,0,0,0,&size) != 0 || nya_llm_load(path,size,&c,error,sizeof(error)) != 0) return 1;
        if (c->output != c->token_embedding || check_gemma_training(c,path) != 0) return 1;
        nya_llm_free(c); fixture_tied = 0;
    }
    for (int variant = 0; variant < 4; ++variant) {
        char path[1200];
        uint64_t size;
        nya_llm_context *c = NULL;
        nya_llm_run_state s;
        char error[512] = {0};
        uint32_t *tokens = NULL;
        size_t count = 0;
        if (snprintf(path, sizeof(path), "%s.%d", argv[1], variant) < 0 || strlen(argv[1]) > 1100) return 1;
        if (make_architecture(path, variant & 1, variant >= 2, variant == 3, 0, &size) != 0 ||
            nya_llm_load(path, size, &c, error, sizeof(error)) != 0) {
            fprintf(stderr, "Gemma fixture %d: %s\n", variant, error); return 1;
        }
        if (nya_llm_tokenize(c, "ab<turn|>a\n\nb", &tokens, &count, error, sizeof(error)) != 0 ||
            count != 7 || tokens[0] != 1 || tokens[1] != 5 || tokens[2] != 7 ||
            tokens[3] != 3 || tokens[4] != 6 || tokens[5] != 6 || tokens[6] != 4 ||
            c->cache_width != 12 || !c->tokenizer.stop_tokens[7]) return 1;
        free(tokens);
        if (variant < 2 && check_gemma_training(c,path) != 0) return 1;
        if (nya_llm_state_create(c, 8, &s, error, sizeof(error)) != 0) return 1;
        for (size_t i = 0; i < 4; ++i) {
            if (nya_llm_forward(c, &s, prompt[i], i, 1) != 0) return 1;
            /* Goldens come from sequence-at-once float64 NumPy equations,
               whereas production executes one F32 token at a time. This tests
               causal masking/cache reuse against a different execution order. */
            for (size_t j = 0; j < 8; ++j) {
                if (!isfinite(s.logits[j]) || fabs((double)s.logits[j] - gemma_reference[variant][i][j]) > 3e-5) {
                    fprintf(stderr, "Gemma %d position %zu logit %zu differs: %.9g vs %.9g\n",
                        variant, i, j, (double)s.logits[j], gemma_reference[variant][i][j]);
                    return 1;
                }
            }
        }
        {
            float embeddings[16];
            nya_generation_soft_tokens span = {1, 2, 8, embeddings, 16, 1};
            nya_generation_request request = {0};
            request.soft_tokens = &span; request.soft_token_spans = 1;
            for (size_t j = 0; j < 16; ++j) embeddings[j] = (float)((int)(j % 11) - 5) / 16.0f;
            if (nya_llm_prefill(c, &s, prompt, 4, &request, error, sizeof(error)) != 0) return 1;
            for (size_t j = 0; j < 8; ++j) if (fabs((double)s.logits[j] - gemma_media_reference[variant][j]) > 3e-5) {
                fprintf(stderr, "Multimodal prefill %d logit %zu: %.9g vs %.9g\n", variant, j,
                    (double)s.logits[j], gemma_media_reference[variant][j]); return 1;
            }
            span.position = 3;
            if (nya_llm_prefill(c, &s, prompt, 4, &request, error, sizeof(error)) == 0) return 1;
            span.position = 1; embeddings[0] = INFINITY;
            if (nya_llm_prefill(c, &s, prompt, 4, &request, error, sizeof(error)) == 0) return 1;
        }
        nya_llm_state_free(&s);
        nya_llm_free(c);
    }
    {
        char target_path[1200], assistant_path[1200], error[512] = {0};
        uint64_t size;
        nya_llm_context *target = NULL, *assistant = NULL;
        nya_generation_request request = {0};
        nya_generation_response expected, actual;
        snprintf(target_path, sizeof(target_path), "%s.target", argv[1]);
        snprintf(assistant_path, sizeof(assistant_path), "%s.mtp", argv[1]);
        if (make_architecture(target_path, 0, 0, 0, 0, &size) != 0 ||
            nya_llm_load(target_path, size, &target, error, sizeof(error)) != 0 ||
            make_architecture(assistant_path, 0, 0, 0, 1, &size) != 0 ||
            nya_llm_load(assistant_path, size, &assistant, error, sizeof(error)) != 0) {
            fprintf(stderr, "MTP load: %s\n", error); return 1;
        }
        /* Disable stop tokens in this numerical fixture so that every window
           gets multiple chances to accept/reject, irrespective of random weights. */
        memset(target->tokenizer.stop_tokens, 0, 8);
        memset(assistant->tokenizer.stop_tokens, 0, 8);
        request.prompt = "a"; request.max_tokens = 12; request.max_output_bytes = 2048; request.top_p = 1.0f;
        if (nya_llm_generate(target, &request, &expected, error, sizeof(error)) != 0) return 1;
        for (size_t window = 1; window <= 8; ++window) {
            request.speculative_tokens = window;
            if (nya_llm_generate_draft(target, assistant, &request, &actual, error, sizeof(error)) != 0 ||
                actual.generated_tokens != expected.generated_tokens || strcmp(actual.text, expected.text) != 0 || actual.draft_tokens == 0) {
                fprintf(stderr, "MTP verification/rollback: %s\n", error); return 1;
            }
            free(actual.text);
        }
        free(expected.text);
        assistant->tokenizer.types[3] = 3;
        if (!nya_llm_tokenizers_match(&target->tokenizer, &assistant->tokenizer, 1) ||
            nya_llm_tokenizers_match(&target->tokenizer, &assistant->tokenizer, 0)) return 1;
        assistant->tokenizer.types[3] = 6;
        if (nya_llm_tokenizers_match(&target->tokenizer, &assistant->tokenizer, 1)) return 1;
        assistant->tokenizer.types[3] = 1;
        if (nya_llm_generate(assistant, &request, &actual, error, sizeof(error)) == 0) return 1;
        assistant->backbone_length += 1;
        if (nya_llm_generate_draft(target, assistant, &request, &actual, error, sizeof(error)) == 0) return 1;
        nya_llm_free(target); nya_llm_free(assistant);
    }
    return 0;
}
