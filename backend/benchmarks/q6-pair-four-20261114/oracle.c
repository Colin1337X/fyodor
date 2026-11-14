/* Independent portable host-double Q6_K oracle. No CUDA/runtime helpers. */
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

EXPORT int q6_oracle(const unsigned char *weights, const float *input,
    size_t columns, size_t rows, double *expected, double *bounds)
{
    if (!weights || !input || !expected || !bounds || !columns ||
        columns % 256 || rows > SIZE_MAX / (4 * sizeof(double)) ||
        columns > SIZE_MAX / (4 * sizeof(float)) ||
        columns / 256 > SIZE_MAX / 210) return -1;
    size_t stride = columns / 256 * 210;
    if (rows > SIZE_MAX / stride) return -1;
    for (size_t row = 0; row < rows; ++row) {
        double sum[4] = {0}, absolute[4] = {0};
        for (size_t block = 0; block < columns / 256; ++block) {
            const unsigned char *b = weights + row * stride + block * 210;
            double d = half_value(b + 208);
            if (!isfinite(d)) return -2;
            for (unsigned j = 0; j < 256; ++j) {
                unsigned half = j / 128, quarter = j % 128 / 32, lane = j % 32;
                unsigned low = b[half * 64 + quarter % 2 * 32 + lane];
                unsigned high = b[128 + half * 32 + lane];
                unsigned low_quant = quarter < 2 ? low % 16 : low / 16;
                unsigned divisor = 1U << (quarter * 2);
                int quant = (int)(low_quant + 16 * ((high / divisor) % 4)) - 32;
                int scale = b[192 + j / 16];
                if (scale >= 128) scale -= 256;
                double weight = d * scale * quant;
                size_t column = block * 256 + j;
                for (unsigned token = 0; token < 4; ++token) {
                    double product = weight * input[(size_t)token * columns + column];
                    if (!isfinite(product)) return -3;
                    sum[token] += product;
                    absolute[token] += fabs(product);
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
