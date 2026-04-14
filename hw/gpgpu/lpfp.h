#ifndef LPFP_H
#define LPFP_H

#include <math.h>
#include <stdint.h>
#include <string.h>

uint16_t f32_to_bf16(float f);
float bf16_to_f32(uint16_t bf);
uint8_t f32_to_e4m3(float f);
float e4m3_to_f32(uint8_t e4m3);
uint8_t f32_to_e5m2(float f);
float e5m2_to_f32(uint8_t e5m2);
uint8_t f32_to_e2m1(float f);
float e2m1_to_f32(uint8_t e2m1);

#endif /* LPFP_H */