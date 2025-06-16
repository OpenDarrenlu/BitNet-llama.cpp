#include <iostream>
#include <random>
#include <cmath>
#include <cstring>
#include <chrono>

// 假设128 = QK_I2_S（即x中每128个2-bit整数压缩成32字节）
#define QK_I2_S 128

// 模拟 i2（2-bit）输入的打包方式：4个值打包成1字节，每值2 bit。
void random_pack_i2(uint8_t* out, int num_values, std::mt19937& rng) {
    std::uniform_int_distribution<int> dist(0, 3);
    for (int i = 0; i < num_values; i += 4) {
        uint8_t byte = 0;
        for (int j = 0; j < 4; ++j) {
            int val = dist(rng);
            byte |= (val & 0x03) << ((3 - j) * 2);
        }
        *out++ = byte;
    }
}

// CPU fallback implementation (精度对比用)
void ggml_vec_dot_i2_i8_s_cpu(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
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

#include <immintrin.h>
// horizontally add 8 int32_t
static inline int hsum_i32_8(const __m256i a) {
    const __m128i sum128 = _mm_add_epi32(_mm256_castsi256_si128(a), _mm256_extractf128_si256(a, 1));
    const __m128i hi64 = _mm_unpackhi_epi64(sum128, sum128);
    const __m128i sum64 = _mm_add_epi32(hi64, sum128);
    const __m128i hi32  = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2, 3, 0, 1));
    return _mm_cvtsi128_si32(_mm_add_epi32(sum64, hi32));
}
void ggml_vec_dot_i2_i8_s(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
    const uint8_t *    x = (uint8_t *)vx;
    const int8_t  *    y = (int8_t *)vy;

    const int nb = n / QK_I2_S;
    const int group32_num = nb / 32;
    const int la_num = nb % 32;
    const int groupla_num = nb % 32 != 0 ? 1 : 0;

    __m256i mask = _mm256_set1_epi8(0x03);
    __m256i accu = _mm256_setzero_si256();

    for (int i=0; i < group32_num; i++){
        __m256i accu32 = _mm256_setzero_si256();
        for (int j=0; j < 32; j++) {
        // 128 index
        __m256i xq8_3 = _mm256_loadu_si256((const __m256i*)(x + i * 32 * 32 + j * 32));
        __m256i xq8_2 = _mm256_srli_epi16(xq8_3, 2);
        __m256i xq8_1 = _mm256_srli_epi16(xq8_3, 4);
        __m256i xq8_0 = _mm256_srli_epi16(xq8_3, 6);

        // each 32 index
        xq8_3 = _mm256_and_si256(xq8_3, mask);
        xq8_2 = _mm256_and_si256(xq8_2, mask);
        xq8_1 = _mm256_and_si256(xq8_1, mask);
        xq8_0 = _mm256_and_si256(xq8_0, mask);

        // each 32 index
        __m256i yq8_0 = _mm256_loadu_si256((const __m256i*)(y + i * 128 * 32 + j * 128 + 0));
        __m256i yq8_1 = _mm256_loadu_si256((const __m256i*)(y + i * 128 * 32 + j * 128 + 32));
        __m256i yq8_2 = _mm256_loadu_si256((const __m256i*)(y + i * 128 * 32 + j * 128 + 64));
        __m256i yq8_3 = _mm256_loadu_si256((const __m256i*)(y + i * 128 * 32 + j * 128 + 96));

        // 128 index accumulation add
        // split into 32 accumulation block
        // each block each 128 index accumulated 4index
        // each index maximum 256
        // each block maximum 4 * 256
        // each block accumulation maximum 127 * 256
        // each 32 group index (128 index in one group) needs cast to int32
        xq8_0 = _mm256_maddubs_epi16(xq8_0, yq8_0);
        xq8_1 = _mm256_maddubs_epi16(xq8_1, yq8_1);
        xq8_2 = _mm256_maddubs_epi16(xq8_2, yq8_2);
        xq8_3 = _mm256_maddubs_epi16(xq8_3, yq8_3);

        accu32 = _mm256_add_epi16(accu32, _mm256_add_epi16(xq8_0, xq8_1));
        accu32 = _mm256_add_epi16(accu32, _mm256_add_epi16(xq8_2, xq8_3));
        }
        accu = _mm256_add_epi32(_mm256_madd_epi16(accu32, _mm256_set1_epi16(1)), accu);
    }

    for (int i = 0; i < groupla_num; i++){
        __m256i accula = _mm256_setzero_si256();
        for (int j = 0; j < la_num; j++) {
        // 128 index
        __m256i xq8_3 = _mm256_loadu_si256((const __m256i*)(x + group32_num * 32 * 32 + j * 32));
        __m256i xq8_2 = _mm256_srli_epi16(xq8_3, 2);
        __m256i xq8_1 = _mm256_srli_epi16(xq8_3, 4);
        __m256i xq8_0 = _mm256_srli_epi16(xq8_3, 6);

        // each 32 index
        xq8_3 = _mm256_and_si256(xq8_3, mask);
        xq8_2 = _mm256_and_si256(xq8_2, mask);
        xq8_1 = _mm256_and_si256(xq8_1, mask);
        xq8_0 = _mm256_and_si256(xq8_0, mask);

        // each 32 index
        __m256i yq8_0 = _mm256_loadu_si256((const __m256i*)(y + group32_num * 128 * 32 + j * 128 + 0));
        __m256i yq8_1 = _mm256_loadu_si256((const __m256i*)(y + group32_num * 128 * 32 + j * 128 + 32));
        __m256i yq8_2 = _mm256_loadu_si256((const __m256i*)(y + group32_num * 128 * 32 + j * 128 + 64));
        __m256i yq8_3 = _mm256_loadu_si256((const __m256i*)(y + group32_num * 128 * 32 + j * 128 + 96));

        // 128 index accumulation add
        // split into 32 accumulation block
        // each block each 128 index accumulated 4index
        // each index maximum 256
        // each block maximum 4 * 256
        // each block accumulation maximum 127 * 256
        // each 32 group index (128 index in one group) needs cast to int32
        xq8_0 = _mm256_maddubs_epi16(xq8_0, yq8_0);
        xq8_1 = _mm256_maddubs_epi16(xq8_1, yq8_1);
        xq8_2 = _mm256_maddubs_epi16(xq8_2, yq8_2);
        xq8_3 = _mm256_maddubs_epi16(xq8_3, yq8_3);

        accula = _mm256_add_epi16(accula, _mm256_add_epi16(xq8_0, xq8_1));
        accula = _mm256_add_epi16(accula, _mm256_add_epi16(xq8_2, xq8_3));
        }
        accu = _mm256_add_epi32(accu, _mm256_madd_epi16(accula, _mm256_set1_epi16(1)));
    }
    int sumi = hsum_i32_8(accu);
    *s = (float)sumi;

}
int main() {
    constexpr int N = 1024; // 输入元素数量，必须是 QK_I2_S 的倍数
    constexpr int X_BYTES = N / 4; // 每4个 2bit value 合并成一个byte
    constexpr int Y_BYTES = N;

    std::vector<uint8_t> x(X_BYTES);
    std::vector<int8_t> y(Y_BYTES);

    std::random_device rd;
    std::mt19937 rng(rd());

    // 生成随机输入
    random_pack_i2(x.data(), N, rng);
    std::uniform_int_distribution<int> dist_y(-128, 127);
    for (int i = 0; i < N; ++i) {
        y[i] = dist_y(rng);
    }

    float result_cpu = 0.0f, result_avx = 0.0f;

    // 计时：CPU参考版本
    auto start_cpu = std::chrono::high_resolution_clock::now();
    ggml_vec_dot_i2_i8_s_cpu(N, &result_cpu, 0, x.data(), 0, y.data(), 0, 0);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpu_time = end_cpu - start_cpu;
    std::cout << "CPU version time: " << cpu_time.count() << " ms\n";

    // 计时：AVX优化版本
    auto start_avx = std::chrono::high_resolution_clock::now();
    ggml_vec_dot_i2_i8_s(N, &result_avx, 0, x.data(), 0, y.data(), 0, 0);
    auto end_avx = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> avx_time = end_avx - start_avx;
    std::cout << "AVX version time: " << avx_time.count() << " ms\n";

    // Compare
    std::cout << "CPU Result (fallback) : " << result_cpu << std::endl;
    std::cout << "AVX2 Result           : " << result_avx << std::endl;
    std::cout << "Absolute Error        : " << std::abs(result_cpu - result_avx) << std::endl;

    return 0;
}
