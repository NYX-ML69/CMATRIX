#include "cmx_cortex_m_port.hpp"

// CMSIS headers - minimal set
#ifdef __CORTEX_M
#include "core_cm4.h"  // Adjust for your specific Cortex-M variant
#endif

namespace cmx {
namespace platform {
namespace cortex_m {

namespace {
    // Static variables for platform state
    volatile uint32_t g_systick_ms = 0;
    volatile uint64_t g_micros_counter = 0;
    PlatformConfig g_config = {};
    bool g_initialized = false;
    
    // Stack canary for stack monitoring
    extern uint32_t _estack;
    extern uint32_t _sstack;
    static const uint32_t STACK_CANARY = 0xDEADBEEF;
}

// SysTick interrupt handler
extern "C" void SysTick_Handler() {
    g_systick_ms++;
    
    // Update microseconds counter (assuming 1ms SysTick)
    g_micros_counter += 1000;
}

InitStatus init(const PlatformConfig& config) {
    if (g_initialized) {
        return InitStatus::SUCCESS;
    }
    
    g_config = config;
    
    // Configure system clock (simplified - platform specific)
    SystemCoreClockUpdate();
    
    // Configure SysTick for 1ms interrupts
    if (SysTick_Config(config.cpu_freq_hz / config.systick_freq_hz) != 0) {
        return InitStatus::TIMER_INIT_FAILED;
    }
    
    // Enable DWT cycle counter if requested
    if (config.enable_dwt_cycle_counter) {
        #ifdef DWT
        CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
        DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;
        DWT->CYCCNT = 0;
        #endif
    }
    
    // Enable cache if available and requested
    if (config.enable_cache) {
        #ifdef __ICACHE_PRESENT
        SCB_EnableICache();
        #endif
        #ifdef __DCACHE_PRESENT
        SCB_EnableDCache();
        #endif
    }
    
    // Initialize status LED GPIO (simplified)
    // Note: Real implementation would configure GPIO registers
    
    // Place stack canary
    *((volatile uint32_t*)&_sstack) = STACK_CANARY;
    
    g_initialized = true;
    return InitStatus::SUCCESS;
}

void deinit() {
    if (!g_initialized) return;
    
    // Disable SysTick
    SysTick->CTRL = 0;
    
    // Disable caches
    #ifdef __ICACHE_PRESENT
    SCB_DisableICache();
    #endif
    #ifdef __DCACHE_PRESENT
    SCB_DisableDCache();
    #endif
    
    g_initialized = false;
}

void delay_ms(uint32_t ms) {
    uint32_t start = g_systick_ms;
    while ((g_systick_ms - start) < ms) {
        __WFI();  // Wait for interrupt to save power
    }
}

void delay_us(uint32_t us) {
    // Use DWT cycle counter for precise microsecond delays
    #ifdef DWT
    if (g_config.enable_dwt_cycle_counter) {
        uint32_t start = DWT->CYCCNT;
        uint32_t cycles = (g_config.cpu_freq_hz / 1000000) * us;
        while ((DWT->CYCCNT - start) < cycles) {
            // Busy wait
        }
        return;
    }
    #endif
    
    // Fallback: approximate delay using CPU cycles
    volatile uint32_t count = (g_config.cpu_freq_hz / 1000000) * us / 4;
    while (count--) {
        __NOP();
    }
}

uint32_t get_millis() {
    return g_systick_ms;
}

uint64_t get_micros() {
    uint32_t ms1, ms2;
    uint32_t tick1, tick2;
    
    // Read with double-check to avoid race condition
    do {
        ms1 = g_systick_ms;
        tick1 = SysTick->VAL;
        ms2 = g_systick_ms;
        tick2 = SysTick->VAL;
    } while (ms1 != ms2 || tick1 < tick2);
    
    // Convert SysTick value to microseconds within current ms
    uint32_t us_in_tick = ((SysTick->LOAD - tick1) * 1000) / SysTick->LOAD;
    
    return (uint64_t)ms1 * 1000 + us_in_tick;
}

uint32_t get_cpu_cycles() {
    #ifdef DWT
    if (g_config.enable_dwt_cycle_counter) {
        return DWT->CYCCNT;
    }
    #endif
    return 0;
}

uint32_t enter_critical() {
    uint32_t mask = __get_PRIMASK();
    __disable_irq();
    return mask;
}

void exit_critical(uint32_t mask) {
    __set_PRIMASK(mask);
}

void set_status_led(bool on) {
    // Platform-specific GPIO manipulation
    // This is a placeholder - real implementation would set GPIO pins
    if (g_config.status_led.port != 0) {
        // GPIO register access would go here
        // Example: *((volatile uint32_t*)(g_config.status_led.port + GPIO_ODR_OFFSET)) 
        //          = on ? (1 << g_config.status_led.pin) : 0;
    }
}

void toggle_status_led() {
    // Platform-specific GPIO toggle
    if (g_config.status_led.port != 0) {
        // Toggle implementation would go here
    }
}

void system_reset() {
    NVIC_SystemReset();
    while(1); // Should never reach here
}

bool in_interrupt() {
    return (__get_IPSR() != 0);
}

size_t get_free_ram() {
    // Simple stack pointer based estimation
    extern char _heap_start;
    extern char _heap_end;
    
    char* stack_ptr = (char*)__get_MSP();
    char* heap_start = &_heap_start;
    
    if (stack_ptr > heap_start) {
        return static_cast<size_t>(stack_ptr - heap_start);
    }
    
    return 0;
}

size_t get_stack_usage() {
    // Check stack canary and estimate usage
    volatile uint32_t* canary_ptr = (volatile uint32_t*)&_sstack;
    
    if (*canary_ptr != STACK_CANARY) {
        // Stack overflow detected!
        return SIZE_MAX;
    }
    
    // Walk up the stack to find high water mark
    uint32_t* current_sp = (uint32_t*)__get_MSP();
    uint32_t* stack_end = (uint32_t*)&_estack;
    
    return static_cast<size_t>((char*)stack_end - (char*)current_sp);
}

} // namespace cortex_m
} // namespace platform
} // namespace cmx