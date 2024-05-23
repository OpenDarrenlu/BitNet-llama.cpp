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

    __m256 acc = _mm256_setzero_ps();
    const __m128i m2b = _mm_set1_epi8(0x03);
    const __m256i mone = _mm256_set1_epi16(1);
    uint16_t tmp[16];
    __m256i aux256 = _mm256_set_epi16(0, tmp[0], 0, tmp[1], 0, tmp[2], 0, tmp[3], 0, tmp[4], 0, tmp[5], 0, tmp[6], 0, tmp[7]);
    __m256i tmpaux = _mm256_srli_epi32(aux256, 10);
    int res = _mm256_cvtsi256_si32(tmpaux);
    printf("%d\n", res);
}