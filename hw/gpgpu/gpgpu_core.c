/*
 * QEMU GPGPU - RISC-V SIMT Core Implementation
 *
 * Copyright (c) 2024-2025
 *
 * This work is licensed under the terms of the GNU GPL, version 2 or later.
 * See the COPYING file in the top-level directory.
 */

#include "qemu/osdep.h"
#include "qemu/log.h"
#include "qemu/host-utils.h"
#include "gpgpu.h"
#include "gpgpu_core.h"

/* Define types of instruction */
typedef enum {
    TYPE_R, TYPE_I, TYPE_U, TYPE_S, TYPE_J, TYPE_B, TYPE_CSR,
    TYPE_FR, TYPE_FI, TYPE_FS, TYPE_F4,
} inst_type_t;

/* ======== Register ======== */
#define G(i) (l->gpr[ctx->i])
#define F(i) (l->fpr[ctx->i])

/* ======== Memory ======== */
#define Mw(addr, len, data) vram_write(ctx->s, addr, len, data)
#define Mr(addr, len) vram_read(ctx->s, addr, len)

/* ======== IMM ======== */
#define immI(i)  (SEXT(BITS(i, 31, 20), 12))
#define immU(i)  ((SEXT(BITS(i, 31, 12), 20) << 12))
#define immS(i)  ((SEXT(BITS(i, 31, 25), 7) << 5) | BITS(i, 11, 7))
#define immB(i)  ((SEXT(BITS(i, 31, 31), 1) << 12) | (BITS(i, 7, 7) << 11) | (BITS(i, 30, 25) << 5) | (BITS(i, 11, 8) << 1))
#define immJ(i)  ((SEXT(BITS(i, 31, 31), 1) << 20) | (BITS(i, 19, 12) << 12) | (BITS(i, 20, 20) << 11) | (BITS(i, 30, 21) << 1))
#define immCSR(i) (BITS(inst, 31, 20))

/* context of warp */
typedef struct exec_ctx {
    GPGPUState *s;
    GPGPUWarp *warp;
    int rd; int rs1; int rs2; int rs3;
    int32_t imm;
    int type;
} exec_ctx_t;

static void get_warp_ctx(exec_ctx_t *ctx, uint32_t inst, int type)
{
    ctx->rd  = BITS(inst, 11, 7);
    ctx->rs1 = BITS(inst, 19, 15);
    ctx->rs2 = BITS(inst, 24, 20);
    ctx->rs3 = BITS(inst, 31, 27);

    switch (type) {
        case TYPE_I: case TYPE_FI:
            ctx->imm = immI(inst); break;
        case TYPE_S: case TYPE_FS:
            ctx->imm = immS(inst); break;
        case TYPE_U:
            ctx->imm = immU(inst); break;
        case TYPE_B:
            ctx->imm = immB(inst); break;
        case TYPE_J:
            ctx->imm = immJ(inst); break;   
        case TYPE_R: case TYPE_FR: case TYPE_F4:
            ctx->imm = 0; break;
        case TYPE_CSR:
            ctx->imm = immCSR(inst); break;
        default:
            ctx->imm = 0; break;
    }
}

/* ================ Function Table ================ */
/* RV32I */
EXEC_FUNC(add,      { G(rd) = src1 + src2; })
EXEC_FUNC(addi,     { G(rd) = src1 + imm; })
EXEC_FUNC(slli,     { G(rd) = src1 << (imm & 0x1F); })
EXEC_FUNC(andi,     { G(rd) = src1 & imm; })
EXEC_FUNC(lui,      { G(rd) = imm; })
EXEC_FUNC(sw,       { Mw(src1 + imm, 4, src2); })
EXEC_FUNC(ebreak,   { /* nothing */ })
EXEC_FUNC(csrrs,    {
    switch ((uint16_t)imm) {
        case CSR_MHARTID: 
            G(rd) = l->mhartid;
            break;
        default:
            printf("[DEBUG] csrrs: unknown CSR 0x%x\n", (uint16_t)imm);
            break;
    }
})

/* RV32F */
EXEC_FUNC(fcvt_s_w, { 
    float result = (float)l->gpr[ctx->rs1];  // 直接读取整数寄存器
    memcpy(&F(rd), &result, sizeof(float));
})

EXEC_FUNC(fmul_s, { 
    // 监控读取
    if (lane_id == 1) {
        static int read_count = 0;
        static uint32_t last_value = 0;
        read_count++;
        
        printf("[MONITOR] READ fpr[1] lane1: value=0x%08x, count=%d\n", 
               l->fpr[1], read_count);
        
        if (read_count > 1 && l->fpr[1] != last_value) {
            printf("[MONITOR] *** fpr[1] changed from 0x%08x to 0x%08x ***\n", 
                   last_value, l->fpr[1]);
        }
        last_value = l->fpr[1];
    }
    
    float result = src1_f * src2_f; 
    memcpy(&F(rd), &result, sizeof(float));
})

EXEC_FUNC(fadd_s, { 
    float result = src1_f + src2_f; 
    printf("[DEBUG] fadd_s: lane=%d, src1_f=%f, src2_f=%f, result=%f, rd=%d\n", 
           lane_id, src1_f, src2_f, result, ctx->rd);
    memcpy(&F(rd), &result, sizeof(float));
    fflush(stdout);
})

EXEC_FUNC(fcvt_w_s, { 
    float f_val;
    memcpy(&f_val, &l->fpr[ctx->rs1], sizeof(float));
    G(rd) = (int32_t)f_val;
})


/* ================ Instruction Table ============== */
typedef void (*exec_func_t)(exec_ctx_t *ctx, int lane_id);
typedef struct opcode_entry {
    uint32_t mask;
    uint32_t match;
    exec_func_t exec;
    int type;
} opcode_entry_t;

#define INSTRUCTION_LIST \
    /* RV32I */ \
    X(add,      "0000000 ????? ????? 000 ????? 01100 11", TYPE_R); \
    X(addi,     "??????? ????? ????? 000 ????? 00100 11", TYPE_I); \
    X(slli,     "0000000 ????? ????? 001 ????? 00100 11", TYPE_I); \
    X(andi,     "??????? ????? ????? 111 ????? 00100 11", TYPE_I); \
    X(lui,      "??????? ????? ????? ??? ????? 01101 11", TYPE_U); \
    X(sw,       "??????? ????? ????? 010 ????? 01000 11", TYPE_S); \
    X(csrrs,    "??????? ????? ????? 010 ????? 11100 11", TYPE_CSR); \
    X(ebreak,   "0000000 00001 00000 000 00000 11100 11", TYPE_I); \
    /* RV32F */ \
    X(fadd_s,   "0000000 ????? ????? ??? ????? 10100 11", TYPE_FR); \
    X(fmul_s,   "0001000 ????? ????? ??? ????? 10100 11", TYPE_FR); \
    X(fcvt_s_w, "1101000 00000 ????? ??? ????? 10100 11", TYPE_FI); \
    X(fcvt_w_s, "1100000 00000 ????? ??? ????? 10100 11", TYPE_FI);

static opcode_entry_t opcode_table[16];
static size_t opcode_table_count = 0;

static void __attribute__((constructor)) init_opcode_table(void)
{
    int idx = 0;
    
    printf("\n=== Initializing Opcode Table ===\n");
    printf("Enum values: TYPE_R=%d, TYPE_I=%d, TYPE_U=%d, TYPE_S=%d, TYPE_J=%d, TYPE_B=%d, TYPE_CSR=%d, TYPE_FR=%d, TYPE_FI=%d, TYPE_FS=%d, TYPE_F4=%d\n",
           TYPE_R, TYPE_I, TYPE_U, TYPE_S, TYPE_J, TYPE_B, TYPE_CSR, TYPE_FR, TYPE_FI, TYPE_FS, TYPE_F4);
    fflush(stdout);
    
#define X(name, pattern, op_type) \
    do { \
        opcode_table[idx].mask = pattern_to_mask(pattern); \
        opcode_table[idx].match = pattern_to_match(pattern); \
        opcode_table[idx].exec = exec_##name; \
        opcode_table[idx].type = op_type; \
        printf("entry %2d: %-10s mask=0x%08x match=0x%08x type=%d (%s)\n", \
               idx, #name, opcode_table[idx].mask, opcode_table[idx].match, op_type, \
               op_type == TYPE_R ? "TYPE_R" : \
               op_type == TYPE_I ? "TYPE_I" : \
               op_type == TYPE_U ? "TYPE_U" : \
               op_type == TYPE_S ? "TYPE_S" : \
               op_type == TYPE_J ? "TYPE_J" : \
               op_type == TYPE_B ? "TYPE_B" : \
               op_type == TYPE_CSR ? "TYPE_CSR" : \
               op_type == TYPE_FR ? "TYPE_FR" : \
               op_type == TYPE_FI ? "TYPE_FI" : \
               op_type == TYPE_FS ? "TYPE_FS" : \
               op_type == TYPE_F4 ? "TYPE_F4" : "UNKNOWN"); \
        idx++; \
        if (idx >= 16) { \
            printf("WARNING: opcode_table size exceeded!\n"); \
        } \
    } while(0)
    
    INSTRUCTION_LIST
    
#undef X
    
    opcode_table_count = idx;
    printf("Total entries initialized: %ld\n", opcode_table_count);
    printf("================================\n\n");
    fflush(stdout);
}

static opcode_entry_t *lookup_opcode(uint32_t inst)
{
    for (size_t i = 0; i < opcode_table_count; i++) {
        if ((inst & opcode_table[i].mask) == opcode_table[i].match) {
            return &opcode_table[i];
        }
    }
    return NULL;
}

/* Only least instructions to pass the test are implemented */
static int exec_one_inst(GPGPUState *s, GPGPUWarp *warp, uint32_t inst)
{
    const opcode_entry_t *entry = lookup_opcode(inst);
    if (!entry) {
        qemu_log_mask(LOG_GUEST_ERROR, "Unsupported: 0x%08x\n", inst);
        return -1;
    }

    exec_ctx_t ctx = {
        .s = s,
        .warp = warp,
        .type = entry->type,
    };

    get_warp_ctx(&ctx, inst, entry->type);
    
    if (entry->match == MATCH_EBREAK) {
        return 1;
    }

    for (int lane = 0; lane < GPGPU_WARP_SIZE; lane++) {
        if (warp->active_mask & (1 << lane)) {
            entry->exec(&ctx, lane);
            warp->lanes[lane].gpr[0] = 0;
            warp->lanes[lane].fpr[0] = 0;
        }
    }

    return 0;
}

/* Initialize the warp */
void gpgpu_core_init_warp(GPGPUWarp *warp, uint32_t pc, 
                          uint32_t thread_id_base, const uint32_t block_id[3],
                          uint32_t num_threads,
                          uint32_t warp_id, uint32_t block_id_linear)
{   
    memset(warp, 0, sizeof(*warp));
    
    warp->thread_id_base = thread_id_base;
    warp->warp_id = warp_id;
    warp->block_id[0] = block_id[0];
    warp->block_id[1] = block_id[1];
    warp->block_id[2] = block_id[2];
    
    /* Set active mask */
    if (num_threads > GPGPU_WARP_SIZE) {
        warp->active_mask = 0xFFFFFFFF;
    } else {
        warp->active_mask = (1 << num_threads) - 1;
    }
    
    /* Initialize each lane */
    for (int i = 0; i < GPGPU_WARP_SIZE; i++) {
        GPGPULane *lane = &warp->lanes[i];
        lane->pc = pc;
        lane->mhartid = MHARTID_ENCODE(block_id_linear, warp_id, i);
        lane->active = (warp->active_mask & (1 << i)) != 0;
        /* x0 is always 0 */
        lane->gpr[0] = 0;
        /* f0 is always 0 */
        lane->fpr[0] = 0;
    }
}

/* warp execution */
int gpgpu_core_exec_warp(GPGPUState *s, GPGPUWarp *warp, uint32_t max_cycles)
{
    uint32_t cycles = 0;
    
    while (cycles < max_cycles) {

        uint32_t pc = warp->lanes[0].pc;
        if (pc >= s->vram_size) {
            return -1;
        }
        
        uint32_t inst = *(uint32_t *)(s->vram_ptr + pc);
        int ret = exec_one_inst(s, warp, inst);
        
        if (ret == 1) {
            return 0;
        } else if (ret == -1) {
            return -1;
        }
        
        for (int i = 0; i < GPGPU_WARP_SIZE; i++) {
            if (warp->active_mask & (1 << i)) {
                warp->lanes[i].pc += 4;
            }
        }
        
        cycles++;
    }
    
    return -1;
}

int gpgpu_core_exec_kernel(GPGPUState *s)
{
    uint32_t grid_dim[3] = {
        s->kernel.grid_dim[0],
        s->kernel.grid_dim[1],
        s->kernel.grid_dim[2]
    };
    uint32_t block_dim[3] = {
        s->kernel.block_dim[0],
        s->kernel.block_dim[1],
        s->kernel.block_dim[2]
    };
    
    uint32_t kernel_addr = s->kernel.kernel_addr;
    uint32_t threads_per_block = block_dim[0] * block_dim[1] * block_dim[2];
    
    for (uint32_t z = 0; z < grid_dim[2]; z++) {
        for (uint32_t y = 0; y < grid_dim[1]; y++) {
            for (uint32_t x = 0; x < grid_dim[0]; x++) {
                uint32_t block_id[3] = {x, y, z};
                uint32_t block_id_linear = z * grid_dim[0] * grid_dim[1] + y * grid_dim[0] + x;
                
                uint32_t num_warps = (threads_per_block + GPGPU_WARP_SIZE - 1) / GPGPU_WARP_SIZE;
                for (uint32_t warp_id = 0; warp_id < num_warps; warp_id++) {
                    GPGPUWarp warp;
                    uint32_t thread_id_base = warp_id * GPGPU_WARP_SIZE;
                    uint32_t num_threads = threads_per_block - thread_id_base;
                    if (num_threads > GPGPU_WARP_SIZE) {
                        num_threads = GPGPU_WARP_SIZE;
                    }
                    gpgpu_core_init_warp(&warp, kernel_addr, thread_id_base, 
                                        block_id, num_threads, 
                                        warp_id, block_id_linear);
                    int ret = gpgpu_core_exec_warp(s, &warp, 1000);
                    if (ret != 0) {
                        return -1;
                    }
                }
            }
        }
    }
    return 0;
}
