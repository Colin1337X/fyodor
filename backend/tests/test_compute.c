#include "compute.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Use non-square matrices and non-workgroup-aligned columns. Signed fractional
 * values expose indexing and reduction bugs hidden by identity/all-one tests.
 * Repeated calls change the vector while keeping immutable cached weights. */
static int check_matvec(nya_compute_context *context, size_t rows, size_t columns)
{
    float *weights = malloc(rows * columns * sizeof(float));
    float *input = malloc(columns * sizeof(float));
    float *output = malloc(rows * sizeof(float));
    int result = -1;
    if (weights == NULL || input == NULL || output == NULL) goto done;
    for (size_t i = 0; i < rows * columns; ++i) weights[i] = (float)((int)(i % 23) - 11) / 17.0f;
    for (int pass = 0; pass < 3; ++pass) {
        for (size_t i = 0; i < columns; ++i) input[i] = (float)((int)(i % 13) - 6 + pass) / 7.0f;
        if (nya_compute_matvec(context, weights, rows, columns, input, output) != 0) {
            fprintf(stderr, "GPU declined test matrix %zu x %zu\n", rows, columns);
            goto done;
        }
        for (size_t row = 0; row < rows; ++row) {
            double expected = 0.0;
            for (size_t col = 0; col < columns; ++col) expected += (double)weights[row * columns + col] * input[col];
            if (!isfinite(output[row]) || fabs((double)output[row] - expected) > 0.0001 * (1.0 + fabs(expected))) {
                fprintf(stderr, "matvec mismatch row %zu: %.9g vs %.9g\n", row, (double)output[row], expected);
                goto done;
            }
        }
    }
    result = 0;
done:
    /* Cached weight addresses cannot be reused for a different matrix while a
     * context lives. Free its copies before releasing the source allocation. */
    nya_compute_free(context);
    free(weights); free(input); free(output);
    return result;
}

/* Alternate two live weight buffers and both growing/shrinking vector shapes.
 * This catches stale descriptors and accidental cache keys based only on size. */
static int check_resizing(void)
{
    nya_compute_context *context = nya_compute_create();
    float small[6], large[325], input[65], output[5];
    int result = -1;
    if (context == NULL) return -1;
    for (size_t i = 0; i < 6; ++i) small[i] = (float)i / 10.0f;
    for (size_t i = 0; i < 325; ++i) large[i] = (float)((int)(i % 17) - 8) / 19.0f;
    for (size_t pass = 0; pass < 6; ++pass) {
        size_t rows = pass % 2 == 0 ? 2 : 5;
        size_t columns = pass % 2 == 0 ? 3 : 65;
        const float *weights = pass % 2 == 0 ? small : large;
        for (size_t i = 0; i < columns; ++i) input[i] = (float)(i + pass) / 71.0f;
        if (nya_compute_matvec(context, weights, rows, columns, input, output) != 0) goto done;
        for (size_t row = 0; row < rows; ++row) {
            double expected = 0.0;
            for (size_t col = 0; col < columns; ++col) expected += (double)weights[row * columns + col] * input[col];
            if (!isfinite(output[row]) || fabs((double)output[row] - expected) > 0.0001 * (1.0 + fabs(expected))) goto done;
        }
    }
    result = 0;
done:
    nya_compute_free(context);
    return result;
}

int main(int argc, char **argv)
{
    nya_compute_context *context = nya_compute_create();
    float value = 1.0f;
    int require_vulkan = argc == 2 && strcmp(argv[1], "--vulkan") == 0;
    int require_cuda = argc == 2 && strcmp(argv[1], "--cuda") == 0;
    if (nya_compute_matvec(NULL, &value, 1, 1, &value, &value) != -1) return 1;
    if (context == NULL) {
        puts("compute: CPU fallback available");
        return require_vulkan || require_cuda ? 77 : 0;
    }
    int is_vulkan = strcmp(nya_compute_name(context), "vulkan") == 0 && nya_compute_vulkan_compiled();
    int is_cuda = strcmp(nya_compute_name(context), "cuda") == 0 && nya_compute_cuda_compiled();
    if ((!is_vulkan && !is_cuda) || (require_vulkan && !is_vulkan) || (require_cuda && !is_cuda) ||
        nya_compute_matvec(context, &value, SIZE_MAX, 2, &value, &value) != -1 ||
        nya_compute_matvec(context, &value, 0, 2, &value, &value) != -1 ||
        nya_compute_matvec(context, NULL, 1, 1, &value, &value) != -1) {
        nya_compute_free(context);
        return 1;
    }
    if (check_matvec(context, 7, 129) != 0) return 1;
    context = nya_compute_create();
    if (context == NULL || check_matvec(context, 257, 1025) != 0) return 1;
    if (check_resizing() != 0) return 1;
    puts("compute: GPU executed reference comparisons and cached-weight reuse");
    return 0;
}
