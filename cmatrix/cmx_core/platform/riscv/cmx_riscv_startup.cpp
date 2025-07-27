#include <cstdint>
#include <cstring>

// External symbols from linker script
extern "C" {
    // Data section symbols
    extern uint32_t __data_start__;
    extern uint32_t __data_end__;
    extern uint32_t __data_load__;
    
    // BSS section symbols
    extern uint32_t __bss_start__;
    extern uint32_t __bss_end__;
    
    // Stack symbols
    extern uint32_t __stack_top__;
    extern uint32_t __stack_bottom__;
    
    // Heap symbols (optional)
    extern uint32_t __heap_start__;
    extern uint32_t __heap_end__;
    
    // Main function
    int main();
    
    // Global constructors/destructors
    extern void (*__init_array_start__)();
    extern void (*__init_array_end__)();
    extern void (*__fini_array_start__)();
    extern void (*__fini_array_end__)();
}

namespace cmx::platform::riscv {

// Forward declarations
void system_early_init() noexcept;
void copy_data_section() noexcept;
void zero_bss_section() noexcept;
void call_global_constructors() noexcept;
void call_global_destructors() noexcept;
void setup_stack() noexcept;

// Exception and interrupt handlers (weak definitions)
extern "C" {
    void __attribute__((weak)) default_exception_handler();
    void __attribute__((weak)) default_interrupt_handler();
    
    // Standard RISC-V exception handlers
    void __attribute__((weak)) instruction_address_misaligned_handler();
    void __attribute__((weak)) instruction_access_fault_handler();
    void __attribute__((weak)) illegal_instruction_handler();
    void __attribute__((weak)) breakpoint_handler();
    void __attribute__((weak)) load_address_misaligned_handler();
    void __attribute__((weak)) load_access_fault_handler();
    void __attribute__((weak)) store_address_misaligned_handler();
    void __attribute__((weak)) store_access_fault_handler();
    void __attribute__((weak)) environment_call_from_u_mode_handler();
    void __attribute__((weak)) environment_call_from_s_mode_handler();
    void __attribute__((weak)) environment_call_from_m_mode_handler();
    void __attribute__((weak)) instruction_page_fault_handler();
    void __attribute__((weak)) load_page_fault_handler();
    void __attribute__((weak)) store_page_fault_handler();
}

// Interrupt/exception vector table
extern "C" const void* __attribute__((section(".vectors"))) interrupt_vector_table[] = {
    reinterpret_cast<const void*>(instruction_address_misaligned_handler),  // 0
    reinterpret_cast<const void*>(instruction_access_fault_handler),        // 1
    reinterpret_cast<const void*>(illegal_instruction_handler),             // 2
    reinterpret_cast<const void*>(breakpoint_handler),                      // 3
    reinterpret_cast<const void*>(load_address_misaligned_handler),         // 4
    reinterpret_cast<const void*>(load_access_fault_handler),               // 5
    reinterpret_cast<const void*>(store_address_misaligned_handler),        // 6
    reinterpret_cast<const void*>(store_access_fault_handler),              // 7
    reinterpret_cast<const void*>(environment_call_from_u_mode_handler),    // 8
    reinterpret_cast<const void*>(environment_call_from_s_mode_handler),    // 9
    nullptr,                                                                // 10 - reserved
    reinterpret_cast<const void*>(environment_call_from_m_mode_handler),    // 11
    reinterpret_cast<const void*>(instruction_page_fault_handler),          // 12
    reinterpret_cast<const void*>(load_page_fault_handler),                 // 13
    nullptr,                                                                // 14 - reserved
    reinterpret_cast<const void*>(store_page_fault_handler),                // 15
};

// Early system initialization
void system_early_init() noexcept {
    // Disable interrupts during startup
    asm volatile("csrci mstatus, 0x8"); // Clear MIE bit in mstatus
    
    // Set up trap vector base address
    uintptr_t vector_base = reinterpret_cast<uintptr_t>(interrupt_vector_table);
    asm volatile("csrw mtvec, %0" :: "r"(vector_base));
    
    // Initialize machine-mode CSRs
    // Clear pending interrupts
    asm volatile("csrw mip, zero");
    
    // Set up basic machine-mode environment
    // Enable machine-mode timer, software, and external interrupts
    asm volatile("csrw mie, %0" :: "r"(0x888)); // MTIE | MSIE | MEIE
    
    // Memory barrier to ensure initialization order
    asm volatile("fence" ::: "memory");
}

// Copy initialized data from flash to RAM
void copy_data_section() noexcept {
    uint32_t* src = &__data_load__;
    uint32_t* dst = &__data_start__;
    uint32_t* end = &__data_end__;
    
    // Copy data section word by word for better performance
    while (dst < end) {
        *dst++ = *src++;
    }
    
    // Memory barrier to ensure data is copied before proceeding
    asm volatile("fence" ::: "memory");
}

// Zero out the BSS section
void zero_bss_section() noexcept {
    uint32_t* start = &__bss_start__;
    uint32_t* end = &__bss_end__;
    
    // Zero BSS section word by word
    while (start < end) {
        *start++ = 0;
    }
    
    // Memory barrier to ensure BSS is zeroed before proceeding
    asm volatile("fence" ::: "memory");
}

// Set up stack pointer
void setup_stack() noexcept {
    // Stack pointer should already be set by reset vector
    // But we can verify and adjust if needed
    register uint32_t sp_val;
    asm volatile("mv %0, sp" : "=r"(sp_val));
    
    // Verify stack is within valid range
    uint32_t stack_top = reinterpret_cast<uint32_t>(&__stack_top__);
    uint32_t stack_bottom = reinterpret_cast<uint32_t>(&__stack_bottom__);
    
    if (sp_val > stack_top || sp_val <= stack_bottom) {
        // Reset stack pointer to top of stack
        asm volatile("mv sp, %0" :: "r"(stack_top));
    }
}

// Call global constructors
void call_global_constructors() noexcept {
    void (**init_func)() = &__init_array_start__;
    void (**init_end)() = &__init_array_end__;
    
    while (init_func < init_end) {
        if (*init_func) {
            (*init_func)();
        }
        init_func++;
    }
}

// Call global destructors (typically not used in embedded systems)
void call_global_destructors() noexcept {
    void (**fini_func)() = &__fini_array_start__;
    void (**fini_end)() = &__fini_array_end__;
    
    while (fini_func < fini_end) {
        if (*fini_func) {
            (*fini_func)();
        }
        fini_func++;
    }
}

// Main startup function
extern "C" void __attribute__((section(".text.startup"))) _start() {
    // Early hardware initialization
    system_early_init();
    
    // Set up stack pointer
    setup_stack();
    
    // Copy initialized data from flash to RAM
    copy_data_section();
    
    // Zero out BSS section
    zero_bss_section();
    
    // Call global constructors
    call_global_constructors();
    
    // Enable interrupts before calling main (if desired)
    // asm volatile("csrsi mstatus, 0x8"); // Set MIE bit in mstatus
    
    // Call main function
    int result = main();
    
    // Call global destructors (optional in embedded systems)
    call_global_destructors();
    
    // Handle main return (typically not used in embedded systems)
    // In embedded systems, main usually doesn't return
    // If it does, we can halt or reset the system
    (void)result; // Suppress unused variable warning
    
    // Disable interrupts and halt
    asm volatile("csrci mstatus, 0x8"); // Clear MIE bit
    while (true) {
        asm volatile("wfi"); // Wait for interrupt (low power mode)
    }
}

// Default exception/interrupt handlers
extern "C" {
    void __attribute__((weak)) default_exception_handler() {
        // Default exception handler - halt system
        asm volatile("csrci mstatus, 0x8"); // Disable interrupts
        while (true) {
            asm volatile("wfi");
        }
    }
    
    void __attribute__((weak)) default_interrupt_handler() {
        // Default interrupt handler - just return
        return;
    }
    
    // Weak definitions for specific exception handlers
    void __attribute__((weak)) instruction_address_misaligned_handler() {
        default_exception_handler();
    }
    
    void __attribute__((weak)) instruction_access_fault_handler() {
        default_exception_handler();
    }
    
    void __attribute__((weak)) illegal_instruction_handler() {
        default_exception_handler();
    }
    
    void __attribute__((weak)) breakpoint_handler() {
        default_exception_handler();
    }
    
    void __attribute__((weak)) load_address_misaligned_handler() {
        default_exception_handler();
    }
    
    void __attribute__((weak)) load_access_fault_handler() {
        default_exception_handler();
    }
    
    void __attribute__((weak)) store_address_misaligned_handler() {
        default_exception_handler();
    }
    
    void __attribute__((weak)) store_access_fault_handler() {
        default_exception_handler();
    }
    
    void __attribute__((weak)) environment_call_from_u_mode_handler() {
        default_exception_handler();
    }
    
    void __attribute__((weak)) environment_call_from_s_mode_handler() {
        default_exception_handler();
    }
    
    void __attribute__((weak)) environment_call_from_m_mode_handler() {
        default_exception_handler();
    }
    
    void __attribute__((weak)) instruction_page_fault_handler() {
        default_exception_handler();
    }
    
    void __attribute__((weak)) load_page_fault_handler() {
        default_exception_handler();
    }
    
    void __attribute__((weak)) store_page_fault_handler() {
        default_exception_handler();
    }
}

} // namespace cmx::platform::riscv