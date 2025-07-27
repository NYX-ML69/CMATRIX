#include "cmx_cortex_m_port.hpp"
#include <cstdint>
#include <cstring>

// Linker symbols (defined in linker script)
extern uint32_t _sidata;    // Start of initialized data in flash
extern uint32_t _sdata;     // Start of data section in RAM
extern uint32_t _edata;     // End of data section in RAM
extern uint32_t _sbss;      // Start of BSS section
extern uint32_t _ebss;      // End of BSS section
extern uint32_t _sstack;    // Start of stack
extern uint32_t _estack;    // End of stack (stack top)

namespace cmx {
namespace platform {
namespace cortex_m {

namespace {
    // Default platform configuration
    constexpr PlatformConfig DEFAULT_CONFIG = {
        .cpu_freq_hz = 168000000,  // 168 MHz (adjust for your MCU)
        .systick_freq_hz = 1000,   // 1 kHz SysTick (1ms interrupts)
        .status_led = {0, 0, false}, // No LED by default
        .enable_dwt_cycle_counter = true,
        .enable_cache = true
    };
}

/**
 * @brief Early hardware initialization
 * Called before C++ constructors and main()
 */
extern "C" void early_init() {
    // Disable global interrupts during startup
    __disable_irq();
    
    // Configure flash wait states and prefetch
    // Platform-specific code would go here
    
    // Enable FPU if present
    #if (__FPU_PRESENT == 1) && (__FPU_USED == 1)
    SCB->CPACR |= ((3UL << 10*2)|(3UL << 11*2));  // Enable CP10 and CP11 coprocessors
    __DSB();
    __ISB();
    #endif
    
    // Configure vector table offset if needed
    // SCB->VTOR = FLASH_BASE | VECT_TAB_OFFSET;
}

/**
 * @brief Initialize RAM sections
 * Copy initialized data from flash to RAM and zero BSS
 */
extern "C" void init_ram() {
    // Copy initialized data from flash to RAM
    uint32_t* src = &_sidata;
    uint32_t* dest = &_sdata;
    uint32_t* end = &_edata;
    
    while (dest < end) {
        *dest++ = *src++;
    }
    
    // Zero out BSS section
    dest = &_sbss;
    end = &_ebss;
    while (dest < end) {
        *dest++ = 0;
    }
    
    // Fill stack with pattern for debugging (optional)
    #ifdef DEBUG
    constexpr uint32_t STACK_FILL_PATTERN = 0xA5A5A5A5;
    dest = &_sstack;
    end = (uint32_t*)((char*)&_estack - 512); // Leave some space for current usage
    while (dest < end) {
        *dest++ = STACK_FILL_PATTERN;
    }
    #endif
}

/**
 * @brief System initialization
 * Configure clocks, peripherals, and platform
 */
extern "C" void system_init() {
    // Initialize system clocks
    SystemInit();  // CMSIS function
    
    // Early platform initialization
    early_init();
    
    // Initialize RAM sections
    init_ram();
    
    // Initialize CMX platform
    InitStatus status = init(DEFAULT_CONFIG);
    if (status != InitStatus::SUCCESS) {
        // Handle initialization failure
        // Could blink LED in error pattern or trigger reset
        while (1) {
            // Error loop
            for (volatile int i = 0; i < 1000000; i++);
            toggle_status_led();
        }
    }
    
    // Enable global interrupts
    __enable_irq();
}

/**
 * @brief Main application entry point
 * Override this weak function in your application
 */
extern "C" __attribute__((weak)) int cmx_main() {
    // Default main loop - blink status LED
    while (1) {
        set_status_led(true);
        delay_ms(500);
        set_status_led(false);
        delay_ms(500);
    }
    return 0;
}

/**
 * @brief Default fault handler
 * Called when a hard fault occurs
 */
extern "C" __attribute__((weak)) void hard_fault_handler() {
    // Disable interrupts
    __disable_irq();
    
    // Try to blink LED rapidly to indicate error
    for (int i = 0; i < 10; i++) {
        set_status_led(true);
        delay_ms(100);
        set_status_led(false);
        delay_ms(100);
    }
    
    // Reset system
    system_reset();
}

/**
 * @brief Default NMI handler
 */
extern "C" __attribute__((weak)) void nmi_handler() {
    // Handle non-maskable interrupt
    while (1);
}

/**
 * @brief Memory management fault handler
 */
extern "C" __attribute__((weak)) void mem_manage_handler() {
    hard_fault_handler();
}

/**
 * @brief Bus fault handler
 */
extern "C" __attribute__((weak)) void bus_fault_handler() {
    hard_fault_handler();
}

/**
 * @brief Usage fault handler
 */
extern "C" __attribute__((weak)) void usage_fault_handler() {
    hard_fault_handler();
}

/**
 * @brief Assert handler for debug builds
 */
extern "C" __attribute__((weak)) void assert_failed(const char* file, uint32_t line) {
    #ifdef DEBUG
    // In debug mode, halt execution
    __disable_irq();
    while (1);
    #else
    // In release mode, reset system
    system_reset();
    #endif
}

} // namespace cortex_m
} // namespace platform
} // namespace cmx

// C-style entry points for interrupt handlers
extern "C" {
    void Reset_Handler() {
        cmx::platform::cortex_m::system_init();
        cmx::platform::cortex_m::cmx_main();
        while (1); // Should never reach here
    }
    
    void HardFault_Handler() {
        cmx::platform::cortex_m::hard_fault_handler();
    }
    
    void NMI_Handler() {
        cmx::platform::cortex_m::nmi_handler();
    }
    
    void MemManage_Handler() {
        cmx::platform::cortex_m::mem_manage_handler();
    }
    
    void BusFault_Handler() {
        cmx::platform::cortex_m::bus_fault_handler();
    }
    
    void UsageFault_Handler() {
        cmx::platform::cortex_m::usage_fault_handler();
    }
}