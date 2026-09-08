#include "llm_internal.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* The fixture is a complete, deliberately tiny transformer. All attention and
   feed-forward projections are zero, so its residual stays equal to the all-one
   embedding. Output rows a/b then have known logits near 1 and 0.5. This gives
   an independent expected result for every supported weight representation. */
enum fixture_kind {
    FIXTURE_VALID,
    FIXTURE_BAD_OUTPUT,
    FIXTURE_BAD_TOKEN_COUNT,
    FIXTURE_EXTRA_TENSOR,
    FIXTURE_NONFINITE_OUTPUT,
    FIXTURE_MISSING_SCORES,
    FIXTURE_ADD_EOS,
    FIXTURE_INVALID_UTF8_OUTPUT
};

static int write_u32(FILE *file, uint32_t value)
{
    unsigned char bytes[4];
    size_t index;
    for (index = 0; index < 4; ++index) bytes[index] = (unsigned char)(value >> (index * 8));
    return fwrite(bytes, 1, 4, file) == 4 ? 0 : -1;
}

static int write_u64(FILE *file, uint64_t value)
{
    unsigned char bytes[8];
    size_t index;
    for (index = 0; index < 8; ++index) bytes[index] = (unsigned char)(value >> (index * 8));
    return fwrite(bytes, 1, 8, file) == 8 ? 0 : -1;
}

static int write_float(FILE *file, float value)
{
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return write_u32(file, bits);
}

static int write_string(FILE *file, const char *value)
{
    size_t length = strlen(value);
    if (write_u64(file, length) != 0) return -1;
    return fwrite(value, 1, length, file) == length ? 0 : -1;
}

static int metadata_u32(FILE *file, const char *key, uint32_t value)
{
    if (write_string(file, key) != 0 || write_u32(file, 4) != 0) return -1;
    return write_u32(file, value);
}

static int metadata_string(FILE *file, const char *key, const char *value)
{
    if (write_string(file, key) != 0 || write_u32(file, 8) != 0) return -1;
    return write_string(file, value);
}

/* Padding is written explicitly, so the mapped model has real bytes through the
   aligned data end and never depends on sparse-file or seek behavior. */
static int pad_to(FILE *file, size_t end)
{
    long current = ftell(file);
    if (current < 0 || (size_t)current > end) return -1;
    while ((size_t)current < end) {
        if (fputc(0, file) == EOF) return -1;
        current += 1;
    }
    return 0;
}

/* All fixture constants are exact powers of two, so no quantization tolerance
   is needed to compare token decisions across F32/F16/Q4_0/Q8_0. */
static int write_constant_row(FILE *file, uint32_t type, size_t columns, float value)
{
    size_t index;
    uint16_t half = value == 1.0f ? 0x3C00U : value == 0.03125f ? 0x2800U :
                    value == 0.015625f ? 0x2400U : 0;
    if (type == NYA_LLM_TENSOR_F32) {
        for (index = 0; index < columns; ++index) if (write_float(file, value) != 0) return -1;
    } else if (type == NYA_LLM_TENSOR_F16) {
        for (index = 0; index < columns; ++index) {
            if (fputc(half & 255U, file) == EOF || fputc(half >> 8, file) == EOF) return -1;
        }
    } else {
        size_t block;
        for (block = 0; block < columns / 32; ++block) {
            if (fputc(half & 255U, file) == EOF || fputc(half >> 8, file) == EOF) return -1;
            for (index = 0; index < (type == NYA_LLM_TENSOR_Q4_0 ? 16U : 32U); ++index) {
                int byte = type == NYA_LLM_TENSOR_Q4_0 ? 0x99 : 1;
                if (fputc(byte, file) == EOF) return -1;
            }
        }
    }
    return 0;
}

static int make_fixture(const char *path, uint32_t weight_type, enum fixture_kind kind, uint64_t *size)
{
    static const char *names[] = {
        "token_embd.weight", "output_norm.weight", "output.weight",
        "blk.0.attn_norm.weight", "blk.0.attn_q.weight", "blk.0.attn_k.weight",
        "blk.0.attn_v.weight", "blk.0.attn_output.weight", "blk.0.ffn_norm.weight",
        "blk.0.ffn_gate.weight", "blk.0.ffn_down.weight", "blk.0.ffn_up.weight",
        "output.bias"
    };
    static const char *pieces[] = {"<unk>", "<s>", "</s>", "\xE2\x96\x81", "a", "b", "ab", ("\xE2\x96\x81" "a")};
    size_t offsets[13], lengths[13], columns[13], rows[13];
    uint32_t types[13];
    size_t count = kind == FIXTURE_EXTRA_TENSOR ? 13 : 12;
    size_t index, row, offset = 0, data_start;
    long position;
    int result = -1;
    FILE *file = fopen(path, "wb");
    if (file == NULL) return -1;

    if (fwrite("GGUF", 1, 4, file) != 4 || write_u32(file, 3) != 0 ||
        write_u64(file, count) != 0 || write_u64(file, kind == FIXTURE_MISSING_SCORES ? 15 : 16) != 0) goto done;
    if (metadata_string(file, "general.architecture", "llama") != 0 ||
        metadata_string(file, "tokenizer.ggml.model", "llama") != 0 ||
        metadata_u32(file, "llama.embedding_length", 32) != 0 ||
        metadata_u32(file, "llama.feed_forward_length", 32) != 0 ||
        metadata_u32(file, "llama.block_count", 1) != 0 ||
        metadata_u32(file, "llama.attention.head_count", 4) != 0 ||
        metadata_u32(file, "llama.attention.head_count_kv", 2) != 0 ||
        metadata_u32(file, "llama.context_length", 16) != 0 ||
        metadata_u32(file, "tokenizer.ggml.unknown_token_id", 0) != 0 ||
        metadata_u32(file, "tokenizer.ggml.bos_token_id", 1) != 0 ||
        metadata_u32(file, "tokenizer.ggml.eos_token_id", 2) != 0) goto done;

    /* Put scores first to exercise metadata-order-independent cleanup. The
       malformed variant deliberately supplies fewer pieces than score entries. */
    if (kind != FIXTURE_MISSING_SCORES) {
        if (write_string(file, "tokenizer.ggml.scores") != 0 || write_u32(file, 9) != 0 ||
            write_u32(file, 6) != 0 || write_u64(file, 8) != 0) goto done;
        for (index = 0; index < 8; ++index) {
            float score = index == 7 ? 2.0f : index == 6 ? 1.0f : 0.0f;
            if (write_float(file, score) != 0) goto done;
        }
    }
    if (write_string(file, "tokenizer.ggml.tokens") != 0 || write_u32(file, 9) != 0 ||
        write_u32(file, 8) != 0 || write_u64(file, kind == FIXTURE_BAD_TOKEN_COUNT ? 3 : 8) != 0) goto done;
    for (index = 0; index < (kind == FIXTURE_BAD_TOKEN_COUNT ? 3U : 8U); ++index) {
        if (write_string(file, index == 4 && kind == FIXTURE_INVALID_UTF8_OUTPUT ? "<0xC3>" : pieces[index]) != 0) goto done;
    }
    if (write_string(file, "tokenizer.ggml.token_type") != 0 || write_u32(file, 9) != 0 ||
        write_u32(file, 5) != 0 || write_u64(file, 8) != 0) goto done;
    for (index = 0; index < 8; ++index) if (write_u32(file, index == 1 || index == 2 ? 3 : 1) != 0) goto done;
    if (write_string(file, "tokenizer.ggml.add_eos_token") != 0 || write_u32(file, 7) != 0 ||
        fputc(kind == FIXTURE_ADD_EOS, file) == EOF ||
        metadata_u32(file, "general.alignment", 32) != 0) goto done;

    for (index = 0; index < count; ++index) {
        int vector = index == 1 || index == 3 || index == 8 || index == 12;
        columns[index] = index == 2 && kind == FIXTURE_BAD_OUTPUT ? 16 : 32;
        rows[index] = vector ? 1 : index == 0 || index == 2 ? 8 : index == 5 || index == 6 ? 16 : 32;
        types[index] = vector ? NYA_LLM_TENSOR_F32 : weight_type;
        lengths[index] = columns[index] * rows[index];
        if (types[index] == NYA_LLM_TENSOR_F32) lengths[index] *= 4;
        else if (types[index] == NYA_LLM_TENSOR_F16) lengths[index] *= 2;
        else lengths[index] = lengths[index] / 32 * (types[index] == NYA_LLM_TENSOR_Q4_0 ? 18U : 34U);
        offsets[index] = offset;
        offset = (offset + lengths[index] + 31) & ~(size_t)31;
        if (write_string(file, names[index]) != 0 || write_u32(file, vector ? 1 : 2) != 0 ||
            write_u64(file, columns[index]) != 0 || (!vector && write_u64(file, rows[index]) != 0) ||
            write_u32(file, types[index]) != 0 || write_u64(file, offsets[index]) != 0) goto done;
    }
    position = ftell(file);
    if (position < 0) goto done;
    data_start = ((size_t)position + 31) & ~(size_t)31;
    if (pad_to(file, data_start) != 0) goto done;
    for (index = 0; index < count; ++index) {
        if (pad_to(file, data_start + offsets[index]) != 0) goto done;
        for (row = 0; row < rows[index]; ++row) {
            float value = index == 0 || index == 1 || index == 3 || index == 8 ? 1.0f :
                          index == 2 && row == 4 ? 0.03125f : index == 2 && row == 5 ? 0.015625f : 0.0f;
            if (index == 2 && kind == FIXTURE_NONFINITE_OUTPUT) value = NAN;
            if (write_constant_row(file, types[index], columns[index], value) != 0) goto done;
        }
    }
    position = ftell(file);
    if (position < 0) goto done;
    *size = (uint64_t)position;
    result = 0;
done:
    if (fclose(file) != 0) result = -1;
    return result;
}

static int fail(const char *message, const char *error)
{
    fprintf(stderr, "%s: %s\n", message, error);
    return 1;
}

static int compare_test_pieces(const void *left, const void *right)
{
    return strcmp(((const nya_llm_token_index *)left)->piece, ((const nya_llm_token_index *)right)->piece);
}

/* Compare the optimized merge queue with a deliberately slow reference that
 * rescans every adjacent pair. Its different control flow checks stale heap
 * entries and leftmost tie-breaking across hundreds of overlapping merges. */
static int test_tokenizer(void)
{
    char *pieces[] = {"<unk>", "<s>", "</s>", "a", "b", "ab", "aba", "ba", "\xc3\xa9\xc3\xa9"};
    size_t lengths[9];
    float scores[] = {0, 0, 0, 0, 0, 1, 2, 1, 3};
    int32_t types[] = {2, 3, 3, 1, 1, 1, 1, 1, 1};
    nya_llm_token_index sorted[9];
    nya_llm_context context = {0};
    nya_llm_tokenizer *tokenizer = &context.tokenizer;
    char error[512];
    uint32_t *actual = NULL;
    size_t actual_count = 0;
    uint32_t random = 12345;
    for (size_t i = 0; i < 9; ++i) {
        lengths[i] = strlen(pieces[i]);
        sorted[i].piece = pieces[i]; sorted[i].token = (uint32_t)i;
    }
    qsort(sorted, 9, sizeof(sorted[0]), compare_test_pieces);
    tokenizer->pieces = pieces;
    tokenizer->piece_lengths = lengths;
    tokenizer->scores = scores;
    tokenizer->types = types;
    tokenizer->sorted = sorted;
    tokenizer->vocabulary_size = 9;
    tokenizer->maximum_piece_length = 5;
    tokenizer->beginning_token = 1;
    tokenizer->end_token = 2;
    tokenizer->add_beginning_token = 1;
    for (size_t i = 0; i < 256; ++i) tokenizer->byte_tokens[i] = -1;
    /* No individual e-acute piece exists: fallback must follow the pair merge. */
    if (nya_llm_tokenize(&context, "\xc3\xa9\xc3\xa9", &actual, &actual_count, error, sizeof(error)) != 0 ||
        actual_count != 2 || actual[1] != 8) { free(actual); return fail("Unicode merge", error); }
    free(actual);
    for (size_t trial = 0; trial < 512; ++trial) {
        char prompt[33];
        char symbols[32][8];
        size_t symbol_count = trial % 32 + 1;
        for (size_t i = 0; i < symbol_count; ++i) {
            random = random * UINT32_C(1664525) + UINT32_C(1013904223);
            prompt[i] = (random >> 28) & 1 ? 'a' : 'b';
            symbols[i][0] = prompt[i]; symbols[i][1] = '\0';
        }
        prompt[symbol_count] = '\0';
        for (;;) {
            size_t best = SIZE_MAX;
            float best_score = -INFINITY;
            for (size_t pair = 0; pair + 1 < symbol_count; ++pair) {
                char merged[16];
                size_t left = strlen(symbols[pair]), right = strlen(symbols[pair + 1]);
                if (left + right >= sizeof(merged)) return 1;
                memcpy(merged, symbols[pair], left);
                memcpy(merged + left, symbols[pair + 1], right + 1);
                for (size_t token = 3; token < 9; ++token) {
                    if (strcmp(merged, pieces[token]) == 0 && scores[token] > best_score) {
                        best = pair; best_score = scores[token];
                    }
                }
            }
            if (best == SIZE_MAX) break;
            {
                size_t left_length = strlen(symbols[best]);
                size_t right_length = strlen(symbols[best + 1]);
                if (left_length + right_length >= sizeof(symbols[best])) return 1;
                memmove(symbols[best] + left_length, symbols[best + 1], right_length + 1);
            }
            for (size_t i = best + 1; i + 1 < symbol_count; ++i) memcpy(symbols[i], symbols[i + 1], sizeof(symbols[i]));
            symbol_count -= 1;
        }
        if (nya_llm_tokenize(&context, prompt, &actual, &actual_count, error, sizeof(error)) != 0 ||
            actual_count != symbol_count + 1) { free(actual); return fail("merge reference count", error); }
        for (size_t i = 0; i < symbol_count; ++i) {
            if (actual[i + 1] >= 9 || strcmp(pieces[actual[i + 1]], symbols[i]) != 0) {
                free(actual); return fail("merge reference token", prompt);
            }
        }
        free(actual);
    }
    {
        char prompt[8193];
        memset(prompt, 'b', sizeof(prompt) - 1); prompt[sizeof(prompt) - 1] = '\0';
        if (nya_llm_tokenize(&context, prompt, &actual, &actual_count, error, sizeof(error)) != 0 ||
            actual_count != 8193) { free(actual); return fail("long prompt", error); }
        for (size_t i = 1; i < actual_count; ++i) if (actual[i] != 4) { free(actual); return 1; }
        free(actual);
    }
    if (nya_llm_tokenize(&context, "\xc0\x80", &actual, &actual_count, error, sizeof(error)) == 0 ||
        actual != NULL || actual_count != 0) { free(actual); return fail("invalid prompt UTF-8", error); }
    return 0;
}

int main(int argument_count, char **arguments)
{
    static const uint32_t types[] = {NYA_LLM_TENSOR_F32, NYA_LLM_TENSOR_F16, NYA_LLM_TENSOR_Q4_0, NYA_LLM_TENSOR_Q8_0};
    nya_llm_context *context = NULL;
    nya_generation_request request;
    nya_generation_response response;
    uint64_t file_size;
    size_t index, token_count;
    uint32_t *tokens = NULL;
    char error[512] = "";
    if (argument_count != 2) return fail("usage", "test_generation SCRATCH_GGUF_PATH");
    if (test_tokenizer() != 0) return 1;
    memset(&request, 0, sizeof(request));
    request.prompt = "a";
    request.max_tokens = 4;
    request.top_p = 1.0f;
    request.max_output_bytes = 64;

    for (index = 0; index < sizeof(types) / sizeof(types[0]); ++index) {
        if (make_fixture(arguments[1], types[index], FIXTURE_VALID, &file_size) != 0 ||
            nya_llm_load(arguments[1], file_size, &context, error, sizeof(error)) != 0) return fail("valid fixture did not load", error);
        if (nya_llm_generate(context, &request, &response, error, sizeof(error)) != 0) return fail("greedy generation failed", error);
        if (strcmp(response.text, "aaaa") != 0 || response.prompt_tokens != 2 || response.generated_tokens != 4 ||
            response.stop_reason != NYA_GENERATION_STOP_LENGTH) return fail("unexpected greedy generation", response.text);
        nya_generation_response_free(&response);

        if (types[index] == NYA_LLM_TENSOR_F32) {
            /* A deliberately worse draft reverses a/b logits. Every greedy
               proposal is wrong, forcing rollback at each draft window. */
            nya_llm_context draft = *context;
            nya_llm_tensor draft_output = *context->output;
            float weights[8 * 32];
            size_t sampled_a = 0, accepted = 0;
            memcpy(weights, draft_output.data, sizeof(weights));
            for (size_t j = 0; j < 32; ++j) { weights[4 * 32 + j] = 0.015625f; weights[5 * 32 + j] = 0.03125f; }
            draft_output.data = (const unsigned char *)(const void *)weights;
            draft.output = &draft_output;
            draft.compute = NULL;
            for (size_t window = 1; window <= 8; ++window) {
                request.speculative_tokens = window;
                if (nya_llm_generate_draft(context, &draft, &request, &response, error, sizeof(error)) != 0 ||
                    strcmp(response.text, "aaaa") != 0 || response.accepted_draft_tokens != 0 ||
                    response.draft_tokens < 4) return fail("greedy draft rollback failed", error);
                nya_generation_response_free(&response);
            }
            /* The corrected distribution must follow p(a)=exp(1)/(exp(1)+
               exp(.5)) ~= .62246, although this draft favors b. Independent
               seeds exercise both acceptance and residual sampling. */
            request.speculative_tokens = 4;
            request.temperature = 1.0f; request.top_k = 2; request.max_tokens = 1;
            /* This is a sampling-distribution test, independent of the device
               arithmetic already checked above. Avoid thousands of GPU launch
               synchronizations solely to obtain the same constant logits. */
            nya_compute_context *saved_compute = context->compute;
            context->compute = NULL;
            for (request.seed = 1; request.seed <= 4096; ++request.seed) {
                if (nya_llm_generate_draft(context, &draft, &request, &response, error, sizeof(error)) != 0)
                    return fail("probabilistic speculation failed", error);
                sampled_a += strcmp(response.text, "a") == 0 ? 1U : 0U;
                accepted += response.accepted_draft_tokens;
                nya_generation_response_free(&response);
            }
            context->compute = saved_compute;
            if (sampled_a < 2400 || sampled_a > 2700 || accepted == 0 || accepted >= 4096)
                return fail("speculative output distribution is biased", "a fraction or rejection frequency outside tolerance");
            request.temperature = 0.0f; request.top_k = 0; request.max_tokens = 4;
            if (nya_llm_generate_draft(context, context, &request, &response, error, sizeof(error)) != 0 ||
                strcmp(response.text, "aaaa") != 0 || response.accepted_draft_tokens != 4)
                return fail("identical-model speculation failed", error);
            nya_generation_response_free(&response);
            draft.tokenizer.add_space_prefix = !draft.tokenizer.add_space_prefix;
            if (nya_llm_generate_draft(context, &draft, &request, &response, error, sizeof(error)) == 0 || response.text != NULL)
                return fail("incompatible draft tokenizer accepted", error);
            request.speculative_tokens = 0;
        }

        /* Top-k removes most of the original mass. Top-p=0.6 must retain only a
           after renormalization (a/(a+b) is about 0.622), for every random seed. */
        request.temperature = 1.0f;
        request.top_k = 2;
        request.top_p = 0.6f;
        for (request.seed = 1; request.seed <= 32; ++request.seed) {
            if (nya_llm_generate(context, &request, &response, error, sizeof(error)) != 0 ||
                strcmp(response.text, "aaaa") != 0) return fail("top-k then nucleus sampling failed", error);
            nya_generation_response_free(&response);
        }
        request.temperature = 0.0f;
        request.top_k = 0;
        request.top_p = 1.0f;
        nya_llm_free(context);
        context = NULL;
    }

    for (index = FIXTURE_BAD_OUTPUT; index <= FIXTURE_MISSING_SCORES; ++index) {
        if (index == FIXTURE_NONFINITE_OUTPUT) continue;
        if (make_fixture(arguments[1], NYA_LLM_TENSOR_F32, (enum fixture_kind)index, &file_size) != 0) return fail("fixture write failed", "");
        if (nya_llm_load(arguments[1], file_size, &context, error, sizeof(error)) == 0 || context != NULL || error[0] == '\0') {
            return fail("unsupported/malformed model was accepted", error);
        }
    }

    if (make_fixture(arguments[1], NYA_LLM_TENSOR_F32, FIXTURE_ADD_EOS, &file_size) != 0 ||
        nya_llm_load(arguments[1], file_size, &context, error, sizeof(error)) != 0) return fail("EOS fixture load failed", error);
    if (nya_llm_tokenize(context, "a", &tokens, &token_count, error, sizeof(error)) != 0 ||
        token_count != 3 || tokens[0] != 1 || tokens[1] != 7 || tokens[2] != 2) return fail("BOS/space/merge/EOS tokenization failed", error);
    free(tokens);
    request.max_tokens = 64;
    if (nya_llm_generate(context, &request, &response, error, sizeof(error)) != 0 ||
        response.generated_tokens != 13 || response.stop_reason != NYA_GENERATION_STOP_CONTEXT) return fail("context bound failed", error);
    nya_generation_response_free(&response);
    request.max_tokens = 4;
    request.temperature = NAN;
    if (nya_llm_generate(context, &request, &response, error, sizeof(error)) == 0 || response.text != NULL) return fail("NaN request was accepted", error);
    request.temperature = 0.0f;
    request.top_p = NAN;
    if (nya_llm_generate(context, &request, &response, error, sizeof(error)) == 0 || response.text != NULL) return fail("NaN top-p was accepted", error);
    request.top_p = 1.0f;
    request.max_output_bytes = 1;
    if (nya_llm_generate(context, &request, &response, error, sizeof(error)) == 0 || response.text != NULL) return fail("output bound was ignored", error);
    request.max_output_bytes = 64;
    request.temperature = 1.0e-38f;
    if (nya_llm_generate(context, &request, &response, error, sizeof(error)) != 0 || strcmp(response.text, "aaaa") != 0) return fail("small temperature failed", error);
    nya_generation_response_free(&response);
    nya_llm_free(context);
    context = NULL;

    if (make_fixture(arguments[1], NYA_LLM_TENSOR_F32, FIXTURE_NONFINITE_OUTPUT, &file_size) != 0 ||
        nya_llm_load(arguments[1], file_size, &context, error, sizeof(error)) != 0) return fail("nonfinite fixture load failed", error);
    request.temperature = 0.0f;
    if (nya_llm_generate(context, &request, &response, error, sizeof(error)) == 0 || response.text != NULL) return fail("nonfinite greedy logits were accepted", error);
    nya_llm_free(context);
    context = NULL;
    if (make_fixture(arguments[1], NYA_LLM_TENSOR_F32, FIXTURE_INVALID_UTF8_OUTPUT, &file_size) != 0 ||
        nya_llm_load(arguments[1], file_size, &context, error, sizeof(error)) != 0) return fail("byte fixture load", error);
    if (nya_llm_generate(context, &request, &response, error, sizeof(error)) == 0 || response.text != NULL)
        return fail("incomplete byte output accepted", error);
    nya_llm_free(context);
    if (remove(arguments[1]) != 0) return fail("fixture cleanup failed", "");
    return 0;
}
