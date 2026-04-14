#ifndef INST_H
#define INST_H

#include "qemu/osdep.h"

#define NUM_OF_INST 50
#define MATCH_EBREAK 0x00100073

/* Instruction Parsing Macros and tools */
static inline uint32_t pattern_to_mask(const char *pattern) {
    uint32_t mask = 0;
    const char *p = pattern;
    int bit = 31;
    
    while (*p && bit >= 0) {
        if (*p == '0' || *p == '1') {
            mask |= (1U << bit);
        } else if (*p == ' ') {
            p++;
            continue;
        }
        if (*p != ' ') bit--;
        p++;
    }
    
    return mask;
}

static inline uint32_t pattern_to_match(const char *pattern) {
    uint32_t match = 0;
    const char *p = pattern;
    int bit = 31;
    
    while (*p && bit >= 0) {
        if (*p == '1') {
            match |= (1U << bit);
        } else if (*p == ' ') {
            p++;
            continue;
        }
        if (*p != ' ') bit--;
        p++;
    }
    
    return match;
}

/* Function table macros */
#define IS_FP_SRC_TYPE(t) ((t) == TYPE_FR || (t) == TYPE_FI || (t) == TYPE_FS || (t) == TYPE_F4)
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

#endif /* INST_H */
