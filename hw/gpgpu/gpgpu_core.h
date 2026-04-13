/*
 * QEMU GPGPU - RISC-V SIMT Core
 *
 * Copyright (c) 2024-2025
 *
 * This work is licensed under the terms of the GNU GPL, version 2 or later.
 * See the COPYING file in the top-level directory.
 *
 * 简化的 RV32I 指令解释器，用于 GPU 核心模拟。
 * 参考 NEMU 的设计思想，但独立实现。
 */

#ifndef HW_GPGPU_CORE_H
#define HW_GPGPU_CORE_H

#include "qemu/osdep.h"
#include "fpu/softfloat.h"
#include <math.h>

/* Tools */
#define BITMASK(bits) ((1ull << (bits)) - 1)
#define BITS(x, hi, lo) (((x) >> (lo)) & BITMASK((hi) - (lo) + 1)) // similar to x[hi:lo] in verilog
#define SEXT(x, len) ({ struct { int64_t n : len; } __x = { .n = x }; (uint64_t)__x.n; })
#ifndef ARRAY_SIZE
#define ARRAY_SIZE(arr) (sizeof(arr) / sizeof((arr)[0]))
#endif

/* Instruction Parsing Macros and tools */
static inline uint32_t pattern_to_mask(const char *pattern) {
    uint32_t mask = 0;
    const char *p = pattern;
    int bit = 31;
    
    // printf("[DEBUG] pattern_to_mask: parsing '%s'\n", pattern);
    
    while (*p && bit >= 0) {
        if (*p == '0' || *p == '1') {
            mask |= (1U << bit);
            // printf("  bit %2d: '%c' -> set mask bit\n", bit, *p);
        } else if (*p == '?') {
            // printf("  bit %2d: '?' -> don't care (mask bit not set)\n", bit);
        } else if (*p == ' ') {
            // printf("  bit %2d: space (ignored, bit not decremented)\n", bit);
            p++;
            continue;
        } else {
            // printf("  bit %2d: unexpected char '%c'\n", bit, *p);
        }
        if (*p != ' ') bit--;
        p++;
    }
    
    // printf("[DEBUG] pattern_to_mask: result = 0x%08x\n", mask);
    return mask;
}

static inline uint32_t pattern_to_match(const char *pattern) {
    uint32_t match = 0;
    const char *p = pattern;
    int bit = 31;
    
    // printf("[DEBUG] pattern_to_match: parsing '%s'\n", pattern);
    
    while (*p && bit >= 0) {
        if (*p == '1') {
            match |= (1U << bit);
            // printf("  bit %2d: '1' -> set match bit\n", bit);
        } else if (*p == '0') {
            // printf("  bit %2d: '0' -> clear match bit\n", bit);
        } else if (*p == '?') {
            // printf("  bit %2d: '?' -> don't care (match bit not set)\n", bit);
        } else if (*p == ' ') {
            // printf("  bit %2d: space (ignored, bit not decremented)\n", bit);
            p++;
            continue;
        } else {
            // printf("  bit %2d: unexpected char '%c'\n", bit, *p);
        }
        if (*p != ' ') bit--;
        p++;
    }
    
    // printf("[DEBUG] pattern_to_match: result = 0x%08x\n", match);
    return match;
}

#define MATCH_EBREAK 0x00100073

/* Memory IO function */
static inline uint32_t get_read_addr(void *addr, int len) {
    switch (len) {
        case 1:  return *(uint8_t  *)addr;
        case 2:  return *(uint16_t *)addr;
        case 4:  return *(uint32_t *)addr;
        default: return 0;
    }
}

/* Function table macros */
// 源操作数来自浮点寄存器的类型
#define IS_FP_SRC_TYPE(t) ((t) == TYPE_FR || (t) == TYPE_FI || (t) == TYPE_FS || (t) == TYPE_F4)
// 目的操作数是浮点寄存器的类型（用于 fcvt_s_w）
#define IS_FP_DST_TYPE(t) ((t) == TYPE_FR || (t) == TYPE_FI || (t) == TYPE_FS || (t) == TYPE_F4)

#define INIT_LANE_CONTEXT() \
    GPGPULane *l = &ctx->warp->lanes[lane_id]; \
    uint32_t src1_u32 = IS_FP_SRC_TYPE(ctx->type) ? l->fpr[ctx->rs1] : l->gpr[ctx->rs1]; \
    uint32_t src2_u32 = IS_FP_SRC_TYPE(ctx->type) ? l->fpr[ctx->rs2] : l->gpr[ctx->rs2]; \
    uint32_t src3_u32 = l->fpr[ctx->rs3]; \
    int32_t imm = ctx->imm; \
    float src1_f, src2_f, src3_f; \
    if (IS_FP_DST_TYPE(ctx->type)) { \
        memcpy(&src1_f, &src1_u32, sizeof(float)); \
        memcpy(&src2_f, &src2_u32, sizeof(float)); \
        memcpy(&src3_f, &src3_u32, sizeof(float)); \
    } else { \
        src1_f = 0; src2_f = 0; src3_f = 0; \
    } \
    uint32_t src1 = src1_u32; \
    uint32_t src2 = src2_u32; \
    uint32_t src3 = src3_u32; \
    (void)src1; (void)src2; (void)src3; (void)imm; (void)src1_f; (void)src2_f; (void)src3_f;

#define EXEC_FUNC(name, code) \
    static void __attribute__((unused)) exec_##name(exec_ctx_t *ctx, int lane_id) { \
        INIT_LANE_CONTEXT(); \
        code \
    }

static inline void get_write_addr(void *addr, int len, uint32_t data) {
    switch (len) {
        case 1: *(uint8_t  *)addr = data; return;
        case 2: *(uint16_t *)addr = data; return;
        case 4: *(uint32_t *)addr = data; return;
    }
}

static void out_of_bound(GPGPUState *s, uint32_t addr, int len) {
    if (addr + len > s->vram_size) {
        qemu_log_mask(LOG_GUEST_ERROR, "VRAM read out of bounds: addr=0x%08x, len=%d\n", addr, len);;
    }
}

static uint32_t __attribute__((unused)) vram_read(GPGPUState *s, uint32_t addr, int len) {
    out_of_bound(s, addr, len);
    return get_read_addr(s->vram_ptr + addr, len);
}

static void __attribute__((unused)) vram_write(GPGPUState *s, uint32_t addr, int len, uint32_t data) {
    out_of_bound(s, addr, len);
    get_write_addr(s->vram_ptr + addr, len, data);
}

/* BF16 转换函数 - 修正版 */
static inline uint16_t f32_to_bf16(float f) {
    uint32_t bits;
    memcpy(&bits, &f, sizeof(float));
    
    uint32_t sign = bits & 0x80000000;
    uint32_t exp = (bits >> 23) & 0xFF;
    uint32_t mant = bits & 0x7FFFFF;
    
    // 处理 NaN/Inf
    if (exp == 0xFF) {
        if (mant == 0) {
            return sign ? 0xFF80 : 0x7F80;  // Inf
        }
        return 0x7FFF;  // NaN
    }
    
    // 处理 0 和 subnormal
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

static inline float bf16_to_f32(uint16_t bf) {
    uint32_t bits = ((uint32_t)bf << 16);
    float f;
    memcpy(&f, &bits, sizeof(float));
    return f;
}

/* E4M3 转换函数 (1-4-3 格式, bias=7) */
static inline uint8_t f32_to_e4m3(float f) {
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

static inline float e4m3_to_f32(uint8_t e4m3) {
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
static inline uint8_t f32_to_e5m2(float f) {
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

static inline float e5m2_to_f32(uint8_t e5m2) {
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
static inline uint8_t f32_to_e2m1(float f) {
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

static inline float e2m1_to_f32(uint8_t e2m1) {
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

/* 前向声明 */
typedef struct GPGPUState GPGPUState;

/*
 * ============================================================================
 * 常量定义
 * ============================================================================
 */
#define GPGPU_WARP_SIZE     32      /* 每个 warp 的 lane 数量 */
#define GPGPU_NUM_REGS      32      /* RISC-V 通用寄存器数量 */
#define GPGPU_NUM_FREGS     32      /* RISC-V 浮点寄存器数量 */

/* 浮点 CSR 地址 */
#define CSR_FFLAGS          0x001
#define CSR_FRM             0x002
#define CSR_FCSR            0x003

/*
 * ============================================================================
 * mhartid CSR 定义
 * ============================================================================
 * 位域布局:
 *   31        13 12     5 4    0
 *   +---------+--------+------+
 *   | block   | warp   | tid  |
 *   | (19bit) | (8bit) | (5b) |
 *   +---------+--------+------+
 */
#define CSR_MHARTID             0xF14

#define MHARTID_THREAD_BITS     5
#define MHARTID_WARP_BITS       8
#define MHARTID_BLOCK_BITS      19
#define MHARTID_THREAD_MASK     0x1F
#define MHARTID_WARP_MASK       0xFF

#define MHARTID_ENCODE(block, warp, thread) \
    (((block) << 13) | ((warp) << 5) | ((thread) & 0x1F))
#define MHARTID_THREAD(id)      ((id) & 0x1F)
#define MHARTID_WARP(id)        (((id) >> 5) & 0xFF)
#define MHARTID_BLOCK(id)       ((id) >> 13)

/*
 * ============================================================================
 * Lane 状态结构
 * ============================================================================
 * 每个 Lane 相当于一个简化的 RISC-V 核心
 */
typedef struct GPGPULane {
    uint32_t gpr[GPGPU_NUM_REGS];   /* 通用寄存器 x0-x31 */
    uint32_t fpr[GPGPU_NUM_FREGS];  /* 浮点寄存器 f0-f31 */
    uint32_t pc;                     /* 程序计数器 */
    uint32_t mhartid;                /* 完整 hart ID (block|warp|lane) */
    uint32_t fcsr;                   /* fflags[4:0] | frm[7:5] */
    float_status fp_status;          /* softfloat 运行状态 */
    bool active;                     /* 是否活跃 */
} GPGPULane;

/*
 * ============================================================================
 * Warp 状态结构
 * ============================================================================
 * 一个 Warp 包含 32 个 Lane，它们锁步执行同一条指令
 */
typedef struct GPGPUWarp {
    GPGPULane lanes[GPGPU_WARP_SIZE];   /* 32 个 lane */
    uint32_t active_mask;                /* 活跃掩码，每位代表一个 lane */
    uint32_t thread_id_base;             /* 这个 warp 的起始 thread_id */
    uint32_t warp_id;                    /* warp 在 block 内的编号 */
    uint32_t block_id[3];                /* 所属 block 的 ID */
} GPGPUWarp;

/*
 * ============================================================================
 * CTRL 设备地址定义 (GPU 核心视角)
 * ============================================================================
 * GPU 核心通过访问这些地址来获取自己的线程 ID
 */
#define GPGPU_CORE_CTRL_BASE        0x80000000  /* CTRL 基地址 */
#define GPGPU_CORE_CTRL_THREAD_ID_X (GPGPU_CORE_CTRL_BASE + 0x00)
#define GPGPU_CORE_CTRL_THREAD_ID_Y (GPGPU_CORE_CTRL_BASE + 0x04)
#define GPGPU_CORE_CTRL_THREAD_ID_Z (GPGPU_CORE_CTRL_BASE + 0x08)
#define GPGPU_CORE_CTRL_BLOCK_ID_X  (GPGPU_CORE_CTRL_BASE + 0x10)
#define GPGPU_CORE_CTRL_BLOCK_ID_Y  (GPGPU_CORE_CTRL_BASE + 0x14)
#define GPGPU_CORE_CTRL_BLOCK_ID_Z  (GPGPU_CORE_CTRL_BASE + 0x18)
#define GPGPU_CORE_CTRL_BLOCK_DIM_X (GPGPU_CORE_CTRL_BASE + 0x20)
#define GPGPU_CORE_CTRL_BLOCK_DIM_Y (GPGPU_CORE_CTRL_BASE + 0x24)
#define GPGPU_CORE_CTRL_BLOCK_DIM_Z (GPGPU_CORE_CTRL_BASE + 0x28)
#define GPGPU_CORE_CTRL_GRID_DIM_X  (GPGPU_CORE_CTRL_BASE + 0x30)
#define GPGPU_CORE_CTRL_GRID_DIM_Y  (GPGPU_CORE_CTRL_BASE + 0x34)
#define GPGPU_CORE_CTRL_GRID_DIM_Z  (GPGPU_CORE_CTRL_BASE + 0x38)

/*
 * ============================================================================
 * 执行引擎 API
 * ============================================================================
 */

/**
 * gpgpu_core_init_warp - 初始化一个 warp
 * @warp: warp 状态指针
 * @pc: 初始程序计数器（内核代码地址）
 * @thread_id_base: 起始线程 ID
 * @block_id: block ID 数组 [x, y, z]
 * @num_threads: 活跃线程数量 (最多 32)
 * @warp_id: warp 在 block 内的编号
 * @block_id_linear: 线性化的 block ID
 */
void gpgpu_core_init_warp(GPGPUWarp *warp, uint32_t pc,
                          uint32_t thread_id_base, const uint32_t block_id[3],
                          uint32_t num_threads,
                          uint32_t warp_id, uint32_t block_id_linear);

/**
 * gpgpu_core_exec_warp - 执行一个 warp 直到完成
 * @s: GPGPU 设备状态
 * @warp: warp 状态指针
 * @max_cycles: 最大执行周期数（防止死循环）
 *
 * 返回: 0 成功，-1 错误（如非法指令）
 */
int gpgpu_core_exec_warp(GPGPUState *s, GPGPUWarp *warp, uint32_t max_cycles);

/**
 * gpgpu_core_exec_kernel - 执行完整的 kernel
 * @s: GPGPU 设备状态
 *
 * 根据 s->ker、nel 中配置的 grid/block 维度执行内核。
 * 返回: 0 成功，-1 错误
 */
int gpgpu_core_exec_kernel(GPGPUState *s);

#endif /* HW_GPGPU_CORE_H */
