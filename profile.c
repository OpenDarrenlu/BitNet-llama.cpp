#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <immintrin.h>

void print_bits(int check) {
    for (int i=0; i<32; i++) {
        printf("%d ", (check >> (31 - i)) & (0x00000001));
    }
    printf("\n");
}

static inline __m128i mul_sum_i8_pairs(const __m128i x, const __m128i y) {
    // Get absolute values of x vectors
    const __m128i ax = _mm_sign_epi8(x, x);
    // Sign the values of the y vectors
    const __m128i sy = _mm_sign_epi8(y, x);
    // Perform multiplication and create 16-bit values
    const __m128i dot = _mm_maddubs_epi16(ax, sy);
    const __m128i ones = _mm_set1_epi16(1);
    return _mm_madd_epi16(ones, dot);
}

static inline int hsum_i32_8(const __m256i a) {
    const __m128i sum128 = _mm_add_epi32(_mm256_castsi256_si128(a), _mm256_extractf128_si256(a, 1));
    const __m128i hi64 = _mm_unpackhi_epi64(sum128, sum128);
    const __m128i sum64 = _mm_add_epi32(hi64, sum128);
    const __m128i hi32  = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2, 3, 0, 1));
    return _mm_cvtsi128_si32(_mm_add_epi32(sum64, hi32));
}

static inline __m256 sum_i16_pairs_float(const __m256i x) {
    const __m256i ones = _mm256_set1_epi16(1);
    const __m256i summed_pairs = _mm256_madd_epi16(ones, x);
    return _mm256_cvtepi32_ps(summed_pairs);
}

static inline __m256 mul_sum_us8_pairs_float(const __m256i ax, const __m256i sy) {
#if defined(__AVXVNNI__) || (defined(__AVX512VNNI__) && defined(__AVX512VL__))
    const __m256i zero = _mm256_setzero_si256();
    const __m256i summed_pairs = _mm256_dpbusd_epi32(zero, ax, sy);
    return _mm256_cvtepi32_ps(summed_pairs);
#else
    // Perform multiplication and create 16-bit values
    const __m256i dot = _mm256_maddubs_epi16(ax, sy);
    return sum_i16_pairs_float(dot);
#endif
}

static inline __m256 mul_sum_i8_pairs_float(const __m256i x, const __m256i y) {
#if __AVXVNNIINT8__
    const __m256i zero = _mm256_setzero_si256();
    const __m256i summed_pairs = _mm256_dpbssd_epi32(zero, x, y);
    return _mm256_cvtepi32_ps(summed_pairs);
#else
    // Get absolute values of x vectors
    const __m256i ax = _mm256_sign_epi8(x, x);
    // Sign the values of the y vectors
    const __m256i sy = _mm256_sign_epi8(y, x);
    return mul_sum_us8_pairs_float(ax, sy);
#endif
}

int main() {
    // uint8_t tmp = 0x02;
    // int check = tmp;
    // print_bits(check);
    // uint32_t tmp1 = check >> 28;
    // print_bits(tmp1);
    // float tmp2 = (tmp1 + 0.5f) * 0.25;
    // printf("%f\n", tmp2);

        // for (int j = 0; j < qk / 8; ++j) {
        //     sumi01 += x[i].qs[j + 0]  * ((y[i*8 + 0] >> 2*j) & 0x03 - 1);
        //     sumi02 += x[i].qs[j + 4]  * ((y[i*8 + 1] >> 2*j) & 0x03 - 1);
        //     sumi03 += x[i].qs[j + 8]  * ((y[i*8 + 2] >> 2*j) & 0x03 - 1);
        //     sumi04 += x[i].qs[j + 12] * ((y[i*8 + 3] >> 2*j) & 0x03 - 1);
        //     sumi11 += x[i].qs[j + 16] * ((y[i*8 + 4] >> 2*j) & 0x03 - 1);
        //     sumi12 += x[i].qs[j + 20] * ((y[i*8 + 5] >> 2*j) & 0x03 - 1);
        //     sumi13 += x[i].qs[j + 24] * ((y[i*8 + 6] >> 2*j) & 0x03 - 1);
        //     sumi14 += x[i].qs[j + 28] * ((y[i*8 + 7] >> 2*j) & 0x03 - 1);
        // }

    // __m256 acc = _mm256_setzero_ps();
    // const __m128i m2b = _mm_set1_epi8(0x03);
    // const __m256i mone = _mm256_set1_epi16(1);
    // uint16_t tmp[16];
    // tmp[0] = 0x1101;
    // tmp[1] = 0x1121;
    // tmp[2] = 0x1121;
    // tmp[3] = 0x1111;
    // tmp[4] = 0x1121;
    // tmp[5] = 0x1111;
    // tmp[6] = 0x1121;

    // __m256i aux256 = _mm256_set_epi16(5, tmp[0], 5, tmp[1], 5, tmp[2], 5, tmp[3], 5, tmp[4], 5, tmp[5], 5, tmp[6], 5, tmp[7]);
    // int res1 = _mm256_cvtsi256_si32(aux256);
    // __m256i tmpaux = _mm256_srli_epi32(aux256, 4);
    // int res = _mm256_cvtsi256_si32(tmpaux);
    // printf("%d\n", res1);
    // printf("%d\n", res);

    int num[8];
    // __m256i* tmp = (__m256i*)num;
    for (int i=0; i<8; i++) {
        num[i] = i;
    }

    int dst[8];
    for (int i=0; i<8; i++) {
        dst[i] = -1;
    }
    // __m128i* tmp1 = (__m128i*)dst;
    // hsum_i32_8(*tmp);
    __m256i tmpl = _mm256_loadu_si256((const __m256i *)num);
    __m256i qy = _mm256_loadu_si256((const __m256i *)dst);
    __m256 check = mul_sum_i8_pairs_float(tmpl, qy);
    
    for (int i=0; i<8; i++) {
        printf("%f ", check[i]);
    }
    printf("\n");
}