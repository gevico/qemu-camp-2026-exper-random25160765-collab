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
    printf("[DEBUG] csrrs: lane_id=%d, imm=0x%x, l->mhartid=0x%x\n", 
           lane_id, (uint16_t)imm, l->mhartid);
    fflush(stdout);
    
    switch ((uint16_t)imm) {
        case CSR_MHARTID: 
            G(rd) = l->mhartid;
            printf("[DEBUG] csrrs: lane %d: mhartid=0x%x -> GPR[%u]=0x%x\n", 
                   lane_id, l->mhartid, ctx->rd, l->mhartid);
            fflush(stdout);
            break;
        default:
            printf("[DEBUG] csrrs: unknown CSR 0x%x\n", (uint16_t)imm);
            break;
    }
})

/* RV32F */
EXEC_FUNC(fcvt_s_w, { 
    float result = (float)src1_u32; 
    printf("[DEBUG] fcvt_s_w: BEFORE memcpy, lane=%d, rd=%d, &fpr[rd]=%p\n", 
           lane_id, ctx->rd, &l->fpr[ctx->rd]);
    memcpy(&F(rd), &result, sizeof(float));
    printf("[DEBUG] fcvt_s_w: AFTER memcpy, fpr[%d]=0x%08x\n", 
           ctx->rd, l->fpr[ctx->rd]);
    // 立即验证
    float verify;
    memcpy(&verify, &l->fpr[ctx->rd], sizeof(float));
    printf("[DEBUG] fcvt_s_w: VERIFY float=%f\n", verify);
    fflush(stdout);
})

EXEC_FUNC(fmul_s, { 
    printf("[DEBUG] fmul_s: lane=%d, l=%p, &fpr[1]=%p\n", 
           lane_id, l, &l->fpr[1]);
    printf("[DEBUG] fmul_s: fpr[1]=0x%08x, fpr[2]=0x%08x\n", 
           l->fpr[1], l->fpr[2]);
    printf("[DEBUG] fmul_s: lane=%d, ctx->type=%d, IS_FP_TYPE=%d\n", 
           lane_id, ctx->type, IS_FP_TYPE(ctx->type));
    printf("[DEBUG] fmul_s: src1_u32=0x%08x, src2_u32=0x%08x\n", 
           src1_u32, src2_u32);
    printf("[DEBUG] fmul_s: before memcpy - src1_f=%f, src2_f=%f\n", 
           src1_f, src2_f);
    
    float result = src1_f * src2_f;
    
    printf("[DEBUG] fmul_s: after memcpy? src1_f=%f, src2_f=%f, result=%f, rd=%d\n", 
           src1_f, src2_f, result, ctx->rd);
    memcpy(&F(rd), &result, sizeof(float));
    printf("[DEBUG] fmul_s: stored to fpr[%d]=0x%08x\n", 
           ctx->rd, l->fpr[ctx->rd]);
    fflush(stdout);
})

EXEC_FUNC(fadd_s, { 
    float result = src1_f + src2_f; 
    printf("[DEBUG] fadd_s: lane=%d, src1_f=%f, src2_f=%f, result=%f, rd=%d\n", 
           lane_id, src1_f, src2_f, result, ctx->rd);
    memcpy(&F(rd), &result, sizeof(float));
    fflush(stdout);
})

EXEC_FUNC(fcvt_w_s, { 
    printf("[DEBUG] fcvt_w_s: lane=%d, src1_f=%f, rd=%d\n", 
           lane_id, src1_f, ctx->rd);
    G(rd) = (int32_t)src1_f;
    printf("[DEBUG] fcvt_w_s: result=%d\n", (int32_t)src1_f);
    fflush(stdout);
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
            printf("[DEBUG] lookup_opcode: inst=0x%08x matched entry %zu, mask=0x%08x, match=0x%08x, exec=%p\n",
                   inst, i, opcode_table[i].mask, opcode_table[i].match, opcode_table[i].exec);
            return &opcode_table[i];
        }
    }
    return NULL;
}

/* Only least instructions to pass the test are implemented */
static int exec_one_inst(GPGPUState *s, GPGPUWarp *warp, uint32_t inst)
{
    printf("[DEBUG] exec_one_inst: inst=0x%08x\n", inst);
    fflush(stdout);

    const opcode_entry_t *entry = lookup_opcode(inst);
    if (!entry) {
        printf("[ERROR] exec_one_inst: Unsupported instruction 0x%08x\n", inst);
        qemu_log_mask(LOG_GUEST_ERROR, "Unsupported: 0x%08x\n", inst);
        return -1;
    }

    exec_ctx_t ctx = {
        .s = s,
        .warp = warp,
        .type = entry->type,
    };
    
    printf("[DEBUG] exec_one_inst: opcode matched, entry->match=0x%x, type=%d\n", 
           entry->match, entry->type);
    fflush(stdout);

    get_warp_ctx(&ctx, inst, entry->type);
    printf("[DEBUG] exec_one_inst: get_warp_ctx completed\n");
    fflush(stdout);
    
    if (entry->match == MATCH_EBREAK) {
        printf("[DEBUG] exec_one_inst: EBREAK instruction, returning 1\n");
        fflush(stdout);
        return 1;
    }

    printf("[DEBUG] exec_one_inst: executing for %d active lanes (active_mask=0x%x)\n", 
           __builtin_popcount(warp->active_mask), warp->active_mask);
    fflush(stdout);

    for (int lane = 0; lane < GPGPU_WARP_SIZE; lane++) {
        if (warp->active_mask & (1 << lane)) {
            printf("[DEBUG]   lane %d: before exec, ", lane);
            
            // 打印关键寄存器的值（根据指令类型）
            if (entry->type == TYPE_R || entry->type == TYPE_I) {
                printf("rs1=%u, rs2=%u, rd=%u", 
                       ctx.rs1, ctx.rs2, ctx.rd);
                if (entry->type == TYPE_I) {
                    printf(", imm=0x%x", ctx.imm);
                }
            } else if (entry->type == TYPE_S) {
                printf("rs1=%u, rs2=%u, imm=0x%x", 
                       ctx.rs1, ctx.rs2, ctx.imm);
            } else if (entry->type == TYPE_U) {
                printf("rd=%u, imm=0x%x", ctx.rd, ctx.imm);
            }
            
            printf("\n");
            fflush(stdout);
            
            entry->exec(&ctx, lane);
            
            printf("[DEBUG]   lane %d: after exec, ", lane);
            if (entry->type == TYPE_R || entry->type == TYPE_I || entry->type == TYPE_U) {
                if (ctx.rd != 0) {
                    printf("rd=%u, value=0x%x", ctx.rd, warp->lanes[lane].gpr[ctx.rd]);
                } else {
                    printf("rd=0 (x0)");
                }
            } else if (entry->type == TYPE_S) {
                printf("store completed");
            }
            printf("\n");
            fflush(stdout);
            
            warp->lanes[lane].gpr[0] = 0;
            warp->lanes[lane].fpr[0] = 0;
        }
    }
    
    printf("[DEBUG] exec_one_inst: completed, returning 0\n");
    fflush(stdout);
    return 0;
}

/* Initialize the warp */
void gpgpu_core_init_warp(GPGPUWarp *warp, uint32_t pc, 
                          uint32_t thread_id_base, const uint32_t block_id[3],
                          uint32_t num_threads,
                          uint32_t warp_id, uint32_t block_id_linear)
{
    printf("[DEBUG] gpgpu_core_init_warp: start\n");
    printf("[DEBUG]   pc=0x%x, thread_id_base=%u, num_threads=%u, warp_id=%u, block_id_linear=%u\n",
           pc, thread_id_base, num_threads, warp_id, block_id_linear);
    printf("[DEBUG]   block_id=(%u,%u,%u)\n", block_id[0], block_id[1], block_id[2]);
    fflush(stdout);
    
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
    printf("[DEBUG]   active_mask = 0x%x (binary: ", warp->active_mask);
    for (int i = 31; i >= 0; i--) {
        if (i == 7) printf(" ");
        printf("%d", (warp->active_mask >> i) & 1);
    }
    printf(")\n");
    fflush(stdout);
    
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
        
        if (i < 8 || (i >= 24 && i < 32)) {  // 打印前8个和后8个
            printf("[DEBUG]   lane[%2d]: active=%d, mhartid=0x%x (block=%u, warp=%u, lane=%u)\n",
                   i, lane->active, lane->mhartid,
                   block_id_linear, warp_id, i);
        }
    }
    fflush(stdout);
    
    printf("[DEBUG] gpgpu_core_init_warp: done\n");
    fflush(stdout);
}

/* warp execution */
int gpgpu_core_exec_warp(GPGPUState *s, GPGPUWarp *warp, uint32_t max_cycles)
{
    uint32_t cycles = 0;
    
    printf("[DEBUG] gpgpu_core_exec_warp: entry\n");
    printf("[DEBUG]   warp->active_mask = 0x%x\n", warp->active_mask);
    printf("[DEBUG]   warp->pc = 0x%x\n", warp->lanes[0].pc);
    printf("[DEBUG]   s->vram_ptr = %p\n", s->vram_ptr);
    printf("[DEBUG]   s->vram_size = 0x%lx\n", s->vram_size);
    fflush(stdout);
    
    while (cycles < max_cycles) {
        printf("[DEBUG] exec_warp: cycle=%u, pc=0x%x\n", cycles, warp->lanes[0].pc);
        fflush(stdout);
        
        uint32_t pc = warp->lanes[0].pc;
        if (pc >= s->vram_size) {
            printf("[ERROR] PC 0x%x >= VRAM size 0x%lx\n", pc, s->vram_size);
            fflush(stdout);
            return -1;
        }
        
        printf("[DEBUG] Reading instruction from vram_ptr+0x%x = %p\n", pc, s->vram_ptr + pc);
        fflush(stdout);
        
        uint32_t inst = *(uint32_t *)(s->vram_ptr + pc);
        printf("[DEBUG] inst = 0x%08x\n", inst);
        fflush(stdout);
        
        printf("[DEBUG] Calling exec_one_inst...\n");
        fflush(stdout);
        int ret = exec_one_inst(s, warp, inst);
        printf("[DEBUG] exec_one_inst returned %d\n", ret);
        fflush(stdout);
        
        if (ret == 1) {
            printf("[DEBUG] exec_one_inst returned 1 (kernel finished)\n");
            return 0;
        } else if (ret == -1) {
            printf("[ERROR] exec_one_inst returned -1\n");
            return -1;
        }
        
        printf("[DEBUG] Updating PCs for active lanes (active_mask=0x%x)\n", warp->active_mask);
        fflush(stdout);
        for (int i = 0; i < GPGPU_WARP_SIZE; i++) {
            if (warp->active_mask & (1 << i)) {
                warp->lanes[i].pc += 4;
                if (i < 5) {  // 只打印前5个lane
                    printf("[DEBUG]   lane[%d] pc -> 0x%x\n", i, warp->lanes[i].pc);
                }
            }
        }
        fflush(stdout);
        
        cycles++;
        printf("[DEBUG] cycles=%u, max_cycles=%u\n", cycles, max_cycles);
        fflush(stdout);
    }
    
    printf("[ERROR] Exceeded max cycles (%u)\n", max_cycles);
    return -1;
}

int gpgpu_core_exec_kernel(GPGPUState *s)
{
    printf("[DEBUG] gpgpu_core_exec_kernel: start\n");
    fflush(stdout);
    
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
    
    printf("[DEBUG] grid_dim: (%u, %u, %u)\n", grid_dim[0], grid_dim[1], grid_dim[2]);
    printf("[DEBUG] block_dim: (%u, %u, %u)\n", block_dim[0], block_dim[1], block_dim[2]);
    printf("[DEBUG] kernel_addr: 0x%lx\n", s->kernel.kernel_addr);
    fflush(stdout);
    
    uint32_t kernel_addr = s->kernel.kernel_addr;
    uint32_t threads_per_block = block_dim[0] * block_dim[1] * block_dim[2];
    printf("[DEBUG] threads_per_block: %u\n", threads_per_block);
    fflush(stdout);
    
    for (uint32_t z = 0; z < grid_dim[2]; z++) {
        for (uint32_t y = 0; y < grid_dim[1]; y++) {
            for (uint32_t x = 0; x < grid_dim[0]; x++) {
                printf("[DEBUG] Processing block (%u, %u, %u)\n", x, y, z);
                fflush(stdout);
                
                uint32_t block_id[3] = {x, y, z};
                uint32_t block_id_linear = z * grid_dim[0] * grid_dim[1] + y * grid_dim[0] + x;
                
                uint32_t num_warps = (threads_per_block + GPGPU_WARP_SIZE - 1) / GPGPU_WARP_SIZE;
                printf("[DEBUG] num_warps: %u\n", num_warps);
                fflush(stdout);
                
                for (uint32_t warp_id = 0; warp_id < num_warps; warp_id++) {
                    printf("[DEBUG] Processing warp_id: %u\n", warp_id);
                    fflush(stdout);
                    
                    GPGPUWarp warp;
                    uint32_t thread_id_base = warp_id * GPGPU_WARP_SIZE;
                    uint32_t num_threads = threads_per_block - thread_id_base;
                    if (num_threads > GPGPU_WARP_SIZE) {
                        num_threads = GPGPU_WARP_SIZE;
                    }
                    
                    printf("[DEBUG] Calling gpgpu_core_init_warp...\n");
                    fflush(stdout);
                    gpgpu_core_init_warp(&warp, kernel_addr, thread_id_base, 
                                        block_id, num_threads, 
                                        warp_id, block_id_linear);
                    printf("[DEBUG] gpgpu_core_init_warp done\n");
                    fflush(stdout);
                    
                    printf("[DEBUG] Calling gpgpu_core_exec_warp...\n");
                    fflush(stdout);
                    int ret = gpgpu_core_exec_warp(s, &warp, 1000);
                    printf("[DEBUG] gpgpu_core_exec_warp returned: %d\n", ret);
                    fflush(stdout);
                    
                    if (ret != 0) {
                        printf("[ERROR] gpgpu_core_exec_warp failed with %d\n", ret);
                        return -1;
                    }
                }
            }
        }
    }
    
    printf("[DEBUG] gpgpu_core_exec_kernel: success\n");
    return 0;
}
