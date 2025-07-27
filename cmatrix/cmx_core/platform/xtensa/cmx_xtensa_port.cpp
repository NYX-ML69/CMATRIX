#include "cmx_xtensa_port.hpp"
#include <xtensa/config/core.h>
#include <xtensa/hal.h>
#include <xtensa/tie/xt_timer.h>
#include <cstring>
#include <cstdio>

namespace cmx::platform::xtensa {

static uint32_t g_cpu_freq_hz = 0;
static uint64_t g_boot_time_cycles = 0;

void cmx_platform_init() {
    // Get CPU frequency from configuration
    g_cpu_freq_hz = xthal_get_ccount_freq();
    if (g_cpu_freq_hz == 0) {
        g_cpu_freq_hz = 240000000; // Default 240MHz for ESP32
    }
    
    // Record boot time
    g_boot_time_cycles = xthal_get_ccount();
    
    // Initialize any platform-specific hardware
    // Enable cycle counter if not already enabled
    XT_WSR_CCOMPARE0(0xFFFFFFFF);
}

uint64_t cmx_get_timestamp_us() {
    uint32_t current_cycles = xthal_get_ccount();
    uint64_t elapsed_cycles = current_cycles - g_boot_time_cycles;
    
    // Convert cycles to microseconds
    return (elapsed_cycles * 1000000ULL) / g_cpu_freq_hz;
}

void cmx_log(const char* message) {
    // Simple UART output - platform specific implementation
    // For now, use a basic approach
    if (message) {
        // In real implementation, this would use UART peripheral
        // For simulation/debugging, could use printf
        #ifdef CMX_DEBUG
        printf("[CMX] %s\n", message);
        #endif
        
        // Could also implement circular buffer for log storage
        // or send to external debugger interface
    }
}

void cmx_yield() {
    // On Xtensa, we can use waiti instruction to wait for interrupt
    // This puts CPU in low-power state until next interrupt
    #if XCHAL_HAVE_INTERRUPTS
    asm volatile ("waiti 0" ::: "memory");
    #else
    // If no interrupt support, just a memory barrier
    asm volatile ("" ::: "memory");
    #endif
}

uint32_t cmx_get_cpu_freq_hz() {
    return g_cpu_freq_hz;
}

// Critical Section Implementation
CriticalSection::CriticalSection() {
    #if XCHAL_HAVE_INTERRUPTS
    saved_int_level_ = XTOS_SET_INTLEVEL(XCHAL_EXCM_LEVEL);
    #else
    saved_int_level_ = 0;
    #endif
}

CriticalSection::~CriticalSection() {
    #if XCHAL_HAVE_INTERRUPTS
    XTOS_RESTORE_INTLEVEL(saved_int_level_);
    #endif
}

void cmx_memory_barrier() {
    asm volatile ("memw" ::: "memory");
}

void cmx_data_cache_flush(void* addr, size_t size) {
    #if XCHAL_DCACHE_SIZE > 0
    char* start = static_cast<char*>(addr);
    char* end = start + size;
    
    // Align to cache line boundaries
    start = reinterpret_cast<char*>(
        reinterpret_cast<uintptr_t>(start) & ~(XCHAL_DCACHE_LINESIZE - 1)
    );
    
    for (char* p = start; p < end; p += XCHAL_DCACHE_LINESIZE) {
        asm volatile ("dhwb %0, 0" :: "a"(p) : "memory");
    }
    #endif
}

void cmx_instruction_cache_invalidate(void* addr, size_t size) {
    #if XCHAL_ICACHE_SIZE > 0
    char* start = static_cast<char*>(addr);
    char* end = start + size;
    
    // Align to cache line boundaries
    start = reinterpret_cast<char*>(
        reinterpret_cast<uintptr_t>(start) & ~(XCHAL_ICACHE_LINESIZE - 1)
    );
    
    for (char* p = start; p < end; p += XCHAL_ICACHE_LINESIZE) {
        asm volatile ("ihi %0, 0" :: "a"(p) : "memory");
    }
    #endif
}

void cmx_platform_reset() {
    // Perform software reset
    // This is platform-specific - for ESP32 it might be different
    #ifdef XCHAL_HAVE_EXCEPTIONS
    // Trigger a reset exception
    asm volatile ("ill");
    #else
    // Infinite loop as fallback
    while (true) {
        asm volatile ("nop");
    }
    #endif
}

size_t cmx_get_free_stack_size() {
    // Get current stack pointer
    void* sp;
    asm volatile ("mov %0, a1" : "=a"(sp));
    
    // This is a rough estimate - in real implementation,
    // you'd need to know the stack boundaries
    extern char _stack_start[];  // Linker symbol
    extern char _stack_end[];    // Linker symbol
    
    uintptr_t current_sp = reinterpret_cast<uintptr_t>(sp);
    uintptr_t stack_start = reinterpret_cast<uintptr_t>(_stack_start);
    
    if (current_sp > stack_start) {
        return current_sp - stack_start;
    }
    
    return 0; // Stack overflow condition
}

const PlatformInfo& cmx_get_platform_info() {
    static PlatformInfo info = {
        .platform_name = "Xtensa",
        .cpu_model = "Xtensa LX6/LX7",
        .cpu_freq_hz = g_cpu_freq_hz,
        .total_memory_bytes = 512 * 1024,     // Platform specific
        .available_memory_bytes = 256 * 1024, // Platform specific
        .has_fpu = XCHAL_HAVE_FP != 0,
        .has_dma = true,  // Most Xtensa platforms have DMA
        .num_cores = 1    // Can be overridden for multi-core variants
    };
    
    return info;
}

} // namespace cmx::platform::xtensa