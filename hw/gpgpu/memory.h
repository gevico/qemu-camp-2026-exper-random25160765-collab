#ifndef MEMORY_H
#define MEMORY_H

#include "qemu/osdep.h"

/* Memory IO function */
static inline uint32_t get_read_addr(void *addr, int len) {
    switch (len) {
        case 1:  return *(uint8_t  *)addr;
        case 2:  return *(uint16_t *)addr;
        case 4:  return *(uint32_t *)addr;
        default: return 0;
    }
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

#endif /* MEMORY_H */