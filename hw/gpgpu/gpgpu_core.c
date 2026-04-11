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
#include "gpgpu.h"
#include "gpgpu_core.h"

/* Define the format of instruction */
typedef union {
    uint32_t raw;
    struct {
        uint32_t opcode : 7;
        uint32_t rd : 5;
        uint32_t funct3 : 3;
        uint32_t rs1 : 5;
        uint32_t rs2 : 5;
        uint32_t funct7 : 7;
    } r_type;
    struct {
        uint32_t opcode : 7;
        uint32_t rd : 5;
        uint32_t funct3 : 3;
        uint32_t rs1 : 5;
        int32_t imm : 12;
    } i_type;
    struct {
        uint32_t opcode : 7;
        uint32_t imm1 : 5;
        uint32_t funct3 : 3;
        uint32_t rs1 : 5;
        uint32_t rs2 : 5;
        int32_t imm2 : 7;
    } s_type;
    struct {
        uint32_t opcode : 7;
        uint32_t rd : 5;
        int32_t imm : 20;
    } u_type;
} rv32i_inst_t;

/* Execute single instruction */
static int exec_one_inst(GPGPUState *s, GPGPUWarp *warp, uint32_t inst)
{
    rv32i_inst_t i = { .raw = inst };
    
    switch (i.r_type.opcode) {
    case 0x33: /* R-type */
        switch (i.r_type.funct3) {
        case 0x0: /* add */
            if (i.r_type.funct7 == 0x00) {
                for (int lane = 0; lane < GPGPU_WARP_SIZE; lane++) {
                    if (warp->active_mask & (1 << lane)) {
                        GPGPULane *l = &warp->lanes[lane];
                        if (i.r_type.rd != 0) {
                            l->gpr[i.r_type.rd] = l->gpr[i.r_type.rs1] + l->gpr[i.r_type.rs2];
                        }
                    }
                }
            }
            break;
        }
        break;
    
    case 0x13: /* I-type */
        switch (i.r_type.funct3) {
        case 0x0: /* addi */
            for (int lane = 0; lane < GPGPU_WARP_SIZE; lane++) {
                if (warp->active_mask & (1 << lane)) {
                    GPGPULane *l = &warp->lanes[lane];
                    if (i.r_type.rd != 0) {
                        l->gpr[i.r_type.rd] = l->gpr[i.r_type.rs1] + i.i_type.imm;
                    }
                }
            }
            break;
        case 0x1: /* slli */
            if (i.i_type.imm >> 5 == 0) {
                for (int lane = 0; lane < GPGPU_WARP_SIZE; lane++) {
                    if (warp->active_mask & (1 << lane)) {
                        GPGPULane *l = &warp->lanes[lane];
                        if (i.r_type.rd != 0) {
                            l->gpr[i.r_type.rd] = l->gpr[i.r_type.rs1] << (i.i_type.imm & 0x1F);
                        }
                    }
                }
            }
            break;
        case 0x7: /* andi */
            for (int lane = 0; lane < GPGPU_WARP_SIZE; lane++) {
                if (warp->active_mask & (1 << lane)) {
                    GPGPULane *l = &warp->lanes[lane];
                    if (i.r_type.rd != 0) {
                        l->gpr[i.r_type.rd] = l->gpr[i.r_type.rs1] & i.i_type.imm;
                    }
                }
            }
            break;
        }
        break;
    
    case 0x37: /* U-type (lui) */
        for (int lane = 0; lane < GPGPU_WARP_SIZE; lane++) {
            if (warp->active_mask & (1 << lane)) {
                GPGPULane *l = &warp->lanes[lane];
                if (i.r_type.rd != 0) {
                    l->gpr[i.r_type.rd] = i.u_type.imm << 12;
                }
            }
        }
        break;
    
    case 0x23: /* S-type (sw) */
        if (i.r_type.funct3 == 0x2) {
            for (int lane = 0; lane < GPGPU_WARP_SIZE; lane++) {
                if (warp->active_mask & (1 << lane)) {
                    GPGPULane *l = &warp->lanes[lane];
                    uint32_t addr = l->gpr[i.r_type.rs1] + ((i.s_type.imm2 << 5) | i.s_type.imm1);
                    if (addr + 4 <= s->vram_size) {
                        *(uint32_t *)(s->vram_ptr + addr) = l->gpr[i.r_type.rs2];
                    }
                }
            }
        }
        break;
    
    case 0x73: /* System instructions */
        if (i.r_type.funct3 == 0x1) { /* csrrs */
            uint16_t csr_addr = (uint16_t)i.i_type.imm;
            if (csr_addr == CSR_MHARTID) {
                for (int lane = 0; lane < GPGPU_WARP_SIZE; lane++) {
                    if (warp->active_mask & (1 << lane)) {
                        GPGPULane *l = &warp->lanes[lane];
                        if (i.r_type.rd != 0) {
                            l->gpr[i.r_type.rd] = l->mhartid;
                        }
                    }
                }
            }
        } else if (i.r_type.funct3 == 0x0 && i.i_type.imm == 0x001) { /* ebreak */
            return 1; /* stop execution */
        }
        break;
    
    default:
        qemu_log_mask(LOG_GUEST_ERROR, "GPGPU: Unsupported instruction: 0x%08x\n", inst);
        return -1;
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
    
    // uint32_t num_blocks = grid_dim[0] * grid_dim[1] * grid_dim[2];
    uint32_t threads_per_block = block_dim[0] * block_dim[1] * block_dim[2];
    
    for (uint32_t z = 0; z < grid_dim[2]; z++) {
        for (uint32_t y = 0; y < grid_dim[1]; y++) {
            for (uint32_t x = 0; x < grid_dim[0]; x++) {

                /* Get the id of block */
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
