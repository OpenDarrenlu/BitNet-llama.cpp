#include <cstdint>
#include <cstddef>
#include <cstring>

#ifndef QK_I2_S
#define QK_I2_S 128  // 如果没有定义，根据实际情况定义
#endif

void ggml_vec_dot_i2_i8_s(int n, float *s, size_t bs, const void *vx, size_t bx, const void *vy, size_t by, int nrc) {
    const uint8_t *x = (const uint8_t *)vx;  // x: packed 2-bit values
    const int8_t *y = (const int8_t *)vy;    // y: full 8-bit values

    int nb = n / QK_I2_S;
    int total_sum = 0;

    for (int block = 0; block < nb; ++block) {
        int block_sum = 0;

        // For QK_I2_S = 128 → 128 / 4 = 32 bytes in packed x
        const uint8_t *x_block = x + block * (QK_I2_S / 4);
        const int8_t *y_block = y + block * QK_I2_S;

        for (int i = 0; i < QK_I2_S / 4; ++i) {
            uint8_t packed = x_block[i];

            for (int j = 0; j < 4; ++j) {
                int shift = (3 - j) * 2;
                uint8_t val2bit = (packed >> shift) & 0x03;

                // 注意：这里你可能需要查表把 2bit 值解码成实际数值（如 -2, -1, 0, +1 等）
                // 我们暂时假设 2bit 映射为：0 → -1, 1 → 0, 2 → +1, 3 → +2
                int x_val;
                switch (val2bit) {
                    case 0: x_val = -1; break;
                    case 1: x_val =  0; break;
                    case 2: x_val =  1; break;
                    case 3: x_val =  2; break;
                }

                int y_val = y_block[i * 4 + j];
                block_sum += x_val * y_val;
            }
        }

        total_sum += block_sum;
    }

    *s = (float)total_sum;
}
