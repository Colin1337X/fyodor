/* Independent host double-precision Q4_K oracle for the optional experiment.
   No CUDA headers, runtime code, packed metadata helper or tensor-core code. */
#include <math.h>
#include <stddef.h>
#include <stdint.h>

#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT
#endif

static double half_value(const unsigned char *p)
{
    unsigned bits = (unsigned)p[0] | (unsigned)p[1] << 8;
    unsigned exponent = (bits >> 10) & 31U, fraction = bits & 1023U;
    double magnitude = exponent ? ldexp(1024.0 + fraction, (int)exponent - 25)
                                : ldexp((double)fraction, -24);
    if (exponent == 31) magnitude = fraction ? NAN : INFINITY;
    return bits & 32768U ? -magnitude : magnitude;
}

EXPORT int q4_oracle(const unsigned char *weights, const float *input,
    size_t columns, size_t rows, double *expected, double *bounds)
{
    if (!weights || !input || !expected || !bounds || !columns ||
        columns % 256 || rows > SIZE_MAX / (4 * sizeof(double)) ||
        columns > SIZE_MAX / (4 * sizeof(float)) ||
        rows > SIZE_MAX / (columns / 256 * 144)) return -1;
    size_t stride = columns / 256 * 144;
    for (size_t row = 0; row < rows; ++row) {
        double sum[4] = {0}, absolute[4] = {0};
        for (size_t block = 0; block < columns / 256; ++block) {
            const unsigned char *b = weights + row * stride + block * 144;
            double d = half_value(b), minimum = half_value(b + 2);
            if (!isfinite(d) || !isfinite(minimum)) return -2;
            for (unsigned group = 0; group < 8; ++group) {
                unsigned scale, offset;
                if (group < 4) {
                    scale = b[4 + group] & 63U;
                    offset = b[8 + group] & 63U;
                } else {
                    scale = (b[8 + group] & 15U) + 16U * (b[group] / 64U);
                    offset = b[8 + group] / 16U + 16U * (b[4 + group] / 64U);
                }
                for (unsigned j = 0; j < 32; ++j) {
                    unsigned packed = b[16 + group / 2 * 32 + j];
                    unsigned quant = group % 2 ? packed / 16U : packed % 16U;
                    double weight = d * scale * quant - minimum * offset;
                    size_t column = block * 256 + group * 32 + j;
                    for (unsigned token = 0; token < 4; ++token) {
                        double product = weight * input[(size_t)token * columns + column];
                        if (!isfinite(product)) return -3;
                        sum[token] += product;
                        absolute[token] += fabs(product);
                    }
                }
            }
        }
        for (unsigned token = 0; token < 4; ++token) {
            expected[(size_t)token * rows + row] = sum[token];
            bounds[(size_t)token * rows + row] = 2e-6 * absolute[token] + 1e-6;
        }
    }
    return 0;
}
