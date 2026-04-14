#include "lpfp.h"

uint16_t f32_to_bf16(float f) {
    uint32_t bits;
    memcpy(&bits, &f, sizeof(float));
    
    uint32_t sign = bits & 0x80000000;
    uint32_t exp = (bits >> 23) & 0xFF;
    uint32_t mant = bits & 0x7FFFFF;
    
    if (exp == 0xFF) {
        if (mant == 0) {
            return sign ? 0xFF80 : 0x7F80;  // Inf
        }
        return 0x7FFF;  // NaN
    }
    
    if (exp == 0) {
        if (mant == 0) {
            return sign ? 0x8000 : 0x0000;
        }
        // Subnormal: 转换为正常数
        int clz = __builtin_clz(mant) - 8;
        mant <<= clz;
        exp = 1 - clz;
    }
    
    // BF16: 指数偏置从127转换到127 (相同), 尾数从23位截断到7位
    // 舍入到最近偶数 (RNE)
    uint32_t bf16_mant = mant >> 16;
    uint32_t rounding_bit = (mant >> 15) & 1;
    uint32_t sticky_bit = (mant & 0x7FFF) ? 1 : 0;
    
    // 舍入逻辑
    if (rounding_bit && (sticky_bit || (bf16_mant & 1))) {
        bf16_mant++;
        if (bf16_mant > 0x7F) {  // 尾数溢出
            bf16_mant = 0;
            exp++;
        }
    }
    
    // 检查指数溢出
    if (exp >= 0xFF) {
        return sign ? 0xFF80 : 0x7F80;  // Inf
    }
    
    return (sign >> 16) | (exp << 7) | bf16_mant;
}

float bf16_to_f32(uint16_t bf) {
    uint32_t bits = ((uint32_t)bf << 16);
    float f;
    memcpy(&f, &bits, sizeof(float));
    return f;
}

uint8_t f32_to_e4m3(float f) {
    uint32_t bits;
    memcpy(&bits, &f, sizeof(float));
    
    uint32_t sign = (bits >> 31) & 1;
    uint32_t exp = (bits >> 23) & 0xFF;
    uint32_t mant = bits & 0x7FFFFF;
    
    // 处理 NaN/Inf
    if (exp == 0xFF) {
        return sign ? 0xFF : 0x7F;  // 饱和到最大绝对值
    }
    
    // 处理 0
    if (exp == 0 && mant == 0) {
        return sign ? 0x80 : 0x00;
    }
    
    // 计算实际值
    float abs_f = fabsf(f);
    
    // E4M3 可表示的最大值
    const float max_val = 448.0f;
    if (abs_f >= max_val) {
        return sign ? 0xFF : 0x7F;
    }
    
    // 处理 subnormal (非常小的数)
    const float min_normal = 0.001953125f;  // 2^-9
    
    if (abs_f < min_normal) {
        // 量化到 subnormal 区域
        if (abs_f < min_normal / 16.0f) {
            return sign ? 0x80 : 0x00;  // 接近 0
        }
        // 查找最接近的 subnormal 值
        uint8_t best_idx = 0;
        float best_diff = abs_f;
        for (int i = 1; i < 8; i++) {
            float val = (i / 512.0f);  // subnormal 值
            float diff = fabsf(val - abs_f);
            if (diff < best_diff) {
                best_diff = diff;
                best_idx = i;
            }
        }
        return sign ? (0x80 | best_idx) : best_idx;
    }
    
    // 正常数: 找到最接近的表示
    // E4M3: 符号1, 指数4(bias=7), 尾数3
    // 可表示的指数范围: -7 到 7 (加上特殊值到 8)
    uint8_t best_encoding = 0;
    float best_diff = INFINITY;
    
    for (int e = 0; e < 16; e++) {
        int actual_exp = e - 7;
        if (actual_exp < -7 || actual_exp > 8) continue;
        
        float exp_val = powf(2.0f, actual_exp);
        
        for (int m = 0; m < 8; m++) {
            float val;
            if (e == 0) {
                // Subnormal
                if (m == 0) val = 0.0f;
                else val = (m / 8.0f) * powf(2.0f, -6);
            } else {
                // Normal
                val = (1.0f + m / 8.0f) * exp_val;
            }
            
            if (val > 448.0f) continue;
            
            float diff = fabsf(val - abs_f);
            if (diff < best_diff) {
                best_diff = diff;
                best_encoding = (e << 3) | m;
            }
        }
    }
    
    return sign ? (0x80 | best_encoding) : best_encoding;
}

float e4m3_to_f32(uint8_t e4m3) {
    uint8_t sign = (e4m3 >> 7) & 1;
    uint8_t exp = (e4m3 >> 3) & 0xF;
    uint8_t mant = e4m3 & 0x7;
    
    float result;
    
    if (exp == 0) {
        // Subnormal
        if (mant == 0) {
            result = 0.0f;
        } else {
            result = (mant / 8.0f) * powf(2.0f, -6);
        }
    } else if (exp == 0xF) {
        // 特殊值 (NaN/Inf 或 最大值)
        if (mant == 0x7) {
            result = 448.0f;
        } else {
            result = NAN;
        }
    } else {
        // Normal
        result = (1.0f + mant / 8.0f) * powf(2.0f, (int)exp - 7);
    }
    
    return sign ? -result : result;
}

/* E5M2 转换函数 (1-5-2 格式, bias=15) */
uint8_t f32_to_e5m2(float f) {
    uint32_t bits;
    memcpy(&bits, &f, sizeof(float));
    
    uint32_t sign = (bits >> 31) & 1;
    uint32_t exp = (bits >> 23) & 0xFF;
    uint32_t mant = bits & 0x7FFFFF;
    
    // 处理 NaN/Inf
    if (exp == 0xFF) {
        if (mant == 0) {
            return sign ? 0xFC : 0x7C;  // Inf
        }
        return 0x7E;  // NaN
    }
    
    // 处理 0
    if (exp == 0 && mant == 0) {
        return sign ? 0x80 : 0x00;
    }
    
    float abs_f = fabsf(f);
    
    // E5M2 最大值
    const float max_val = 57344.0f;
    if (abs_f >= max_val) {
        return sign ? 0xFC : 0x7C;
    }
    
    // 找到最接近的 E5M2 表示
    // E5M2: 符号1, 指数5(bias=15), 尾数2
    uint8_t best_encoding = 0;
    float best_diff = INFINITY;
    
    for (int e = 0; e < 32; e++) {
        int actual_exp = e - 15;
        if (actual_exp < -14 || actual_exp > 15) continue;
        
        float exp_val = powf(2.0f, actual_exp);
        
        for (int m = 0; m < 4; m++) {
            float val;
            if (e == 0) {
                // Subnormal
                val = (m / 4.0f) * powf(2.0f, -14);
            } else {
                // Normal
                val = (1.0f + m / 4.0f) * exp_val;
            }
            
            if (val > 57344.0f) continue;
            
            float diff = fabsf(val - abs_f);
            if (diff < best_diff) {
                best_diff = diff;
                best_encoding = (e << 2) | m;
            }
        }
    }
    
    return sign ? (0x80 | best_encoding) : best_encoding;
}

float e5m2_to_f32(uint8_t e5m2) {
    uint8_t sign = (e5m2 >> 7) & 1;
    uint8_t exp = (e5m2 >> 2) & 0x1F;
    uint8_t mant = e5m2 & 0x3;
    
    float result;
    
    if (exp == 0) {
        // Subnormal
        result = (mant / 4.0f) * powf(2.0f, -14);
    } else if (exp == 0x1F) {
        // 特殊值
        if (mant == 0) {
            result = INFINITY;
        } else {
            result = NAN;
        }
    } else {
        // Normal
        result = (1.0f + mant / 4.0f) * powf(2.0f, (int)exp - 15);
    }
    
    return sign ? -result : result;
}

/* E2M1 转换函数 (1-2-1 格式, bias=1) */
uint8_t f32_to_e2m1(float f) {
    uint32_t bits;
    memcpy(&bits, &f, sizeof(float));
    
    uint32_t sign = (bits >> 31) & 1;
    uint32_t exp = (bits >> 23) & 0xFF;
    // uint32_t mant = bits & 0x7FFFFF;
    
    // 处理 NaN/Inf
    if (exp == 0xFF) {
        return sign ? 0x8F : 0x07;  // 饱和到最大值
    }
    
    float abs_f = fabsf(f);
    
    // E2M1 最大值
    const float max_val = 6.0f;
    if (abs_f >= max_val) {
        return sign ? 0x8F : 0x07;
    }
    
    // E2M1: 符号1, 指数2(bias=1), 尾数1
    // 可表示值: 0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0
    uint8_t best_encoding = 0;
    float best_diff = INFINITY;
    
    for (int e = 0; e < 4; e++) {
        int actual_exp = e - 1;
        if (actual_exp < -1 || actual_exp > 2) continue;
        
        float exp_val = powf(2.0f, actual_exp);
        
        for (int m = 0; m < 2; m++) {
            float val;
            if (e == 0) {
                // Subnormal (只有 0)
                if (m == 0) {
                    val = 0.0f;
                } else {
                    continue;  // E2M1 subnormal 只有 0
                }
            } else {
                // Normal
                val = (1.0f + m / 2.0f) * exp_val;
            }
            
            if (val > 6.0f) continue;
            
            float diff = fabsf(val - abs_f);
            if (diff < best_diff) {
                best_diff = diff;
                best_encoding = (e << 1) | m;
            }
        }
    }
    
    // 特殊处理 6.0 (exp=2, mant=1 实际是 6.0)
    if (fabsf(6.0f - abs_f) < best_diff) {
        best_encoding = 0x7;  // 111: exp=3, mant=1 (实际是6.0)
    }
    
    return sign ? (0x80 | best_encoding) : best_encoding;
}

float e2m1_to_f32(uint8_t e2m1) {
    uint8_t sign = (e2m1 >> 3) & 1;
    uint8_t exp = (e2m1 >> 1) & 0x3;
    uint8_t mant = e2m1 & 0x1;
    
    float result;
    
    if (exp == 0) {
        // Subnormal: 只有 0
        result = 0.0f;
    } else {
        // Normal
        if (exp == 3 && mant == 1) {
            result = 6.0f;  // 特殊值
        } else {
            result = (1.0f + mant / 2.0f) * powf(2.0f, (int)exp - 1);
        }
    }
    
    return sign ? -result : result;
}
