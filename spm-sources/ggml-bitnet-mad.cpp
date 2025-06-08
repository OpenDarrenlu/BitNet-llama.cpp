#include <vector>
#include <type_traits>

#include "ggml-bitnet.h"
#include "ggml-quants.h"
#include <cmath>
#include <cstring>

#define QK_I2_S 128
#define QK_I2 128

size_t quantize_i2_s(const float * src, void * dst, int64_t nrow, int64_t n_per_row, const float * quant_weights) {
    // 2 bits per weight

    size_t row_size = ggml_row_size(GGML_TYPE_I2_S, n_per_row);

    int n = nrow * n_per_row;

    // f32 -> q8
    double max = 0;
    for (int i = 0; i < n; ++i) {
        max = fmax(max, (double)fabs((double)src[i]));
    }
    double i2_scale = max;

    uint8_t* q8 = (uint8_t*)malloc(n * sizeof(uint8_t));
    for (int i=0; i<n; i++) {
        if (fabs((double)(src[i])) < 1e-6) {
            q8[i] = 1;
            continue;
        }
        q8[i] = (double)src[i] * i2_scale > 0 ? 2 : 0;
    }

    memset(dst, 0, n * sizeof(uint8_t) / 4);

    // q8 -> 0, 1, 2
    //       |  |  |
    //      -1, 0, 1

    uint8_t* i2_weight = (uint8_t*)dst;
    for (int i = 0; i < n / QK_I2; i++) {
        for (int j = 0; j < QK_I2; j++) {
            int group_idx = j / 32;
            int group_pos = j % 32;
            uint8_t temp = (q8[i * QK_I2 + j] << (6 - 2 * group_idx));
            i2_weight[i * 32 + group_pos] |= temp;            
        }
    }

    float* scale_ptr = (float*)((char*)i2_weight + n / 4);
    scale_ptr[0] = i2_scale;

    free(q8);

    // 32B for alignment
    return nrow * row_size / 4 + 32;
}

void ggml_vec_dot_i2_i8_s(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
    const uint8_t * x = static_cast<const uint8_t *>(vx);
    const int8_t  * y = static_cast<const int8_t  *>(vy);

    // 计算完整块的数量（每块 128 个元素）
    const int nb = n / QK_I2_S;
    int32_t sum_total = 0;

    // 处理完整的块
    for (int i = 0; i < nb; ++i) {
        // 计算当前块在 x 和 y 中的偏移
        const uint8_t * x_block = x + i * 32;  // 每块占用 32 字节
        const int8_t  * y_block = y + i * 128; // 每块占用 128 字节
        
        // 处理块内的每个字节（32 字节/块）
        for (int j = 0; j < 32; ++j) {
            const uint8_t x_byte = x_block[j];
            
            // 从单个字节中提取 4 个 2 位元素
            const int e0 = (x_byte >> 6) & 0x03;
            const int e1 = (x_byte >> 4) & 0x03;
            const int e2 = (x_byte >> 2) & 0x03;
            const int e3 = (x_byte >> 0) & 0x03;
            
            // 计算点积（跨步访问 y 的 4 个区段）
            sum_total += e0 * y_block[j + 0];
            sum_total += e1 * y_block[j + 32];
            sum_total += e2 * y_block[j + 64];
            sum_total += e3 * y_block[j + 96];
        }
    }

    // 将整数结果转换为浮点数
    *s = static_cast<float>(sum_total);

}