#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>

static int iq2_compare_func(const void * left, const void * right) {
    const int * l = (const int *)left;
    const int * r = (const int *)right;
    return l[0] < r[0] ? -1 : l[0] > r[0] ? 1 : l[1] < r[1] ? -1 : l[1] > r[1] ? 1 : 0;
}

int main() {
    // int8_t pos[8];
    // int i = 54;
    // for (int k = 0; k < 8; ++k) {
    //     // put index()
    //     int l = (i >> 2*k) & 0x3;
    //     pos[k] = 2*l + 1;
    // }    

    // for (int i=0; i<8; i++) {
    //     printf("%d ", pos[i]);
    // }
    // printf("\n");
    // int dist2[6];
    // int grid_size = 3;
    // dist2[0] = -2;
    // dist2[1] = 0;
    // dist2[0] = 1;
    // dist2[1] = 1;
    // dist2[0] = -3;
    // dist2[1] = 2;

    // qsort(dist2, grid_size, 2*sizeof(int), iq2_compare_func);

    // for (int i=0; i<6; i++) {
    //     printf("%d ", dist2[i]);
    // }
    // printf("\n");
    // int block_size = 32;
    // uint16_t h = 0;
    // uint8_t qs[32];
    // uint16_t index[4];
    // index[0] = 42120; // 10100100 / 10001000
    // index[1] = 1;     // 00000000 / 00000001
    // index[2] = 2;     // 00000000 / 00000010
    // index[3] = 21312; // 01010011 / 01000000
    // for (int k = 0; k < block_size/8; ++k) {
    //     qs[k] = index[k] & 255;
    //     uint16_t tmp = (index[k] >> 8) << 3*k; // add zero when >> 8
    //     // k=0 -> >>8 <<0
    //     // k=1 -> >>8 <<3
    //     // k=2 -> >>8 <<6
    //     // k=3 -> >>8 <<9
    //     printf("tmp:%d\n", tmp);
    //     h |= tmp;
    // }

    // for (int i=0; i<4; i++) {
    //     printf("%d ", qs[i]);
    // }
    // printf("\n");
    // printf("h:%d\n", h); //1010011010100100

    int8_t tmp = 1;
    float delta = 0.125f;
    printf("%f\n", tmp + delta);
}

