#include "cmx_riscv_port.hpp"
#include <cstring>

namespace cmx::platform::riscv {

// Platform-specific constants
static constexpr uint32_t DEFAULT_CPU_FREQ_HZ = 100000000; // 100 MHz default
static constexpr uint32_t CYCLES_PER_MICROSECOND = DEFAULT_CPU_FREQ_HZ / 1000000;

// CSR manipulation helpers
template<uint32_t csr>
inline uint32_t read_csr() noexcept {
    uint32_t value;
    asm volatile("csrr %0, %1" : "=r"(value) : "i"(csr));
    return value;
}

template<uint32_t csr>
inline void write_csr(uint32_t value) noexcept {
    asm volatile("csrw %0, %1" :: "i"(csr), "r"(value));
}

template<uint32_t csr>
inline uint32_t set_csr_bits(uint32_t mask) noexcept {
    uint32_t old_value;
    asm volatile("csrrs %0, %1, %2" : "=r"(old_value) : "i"(csr), "r"(mask));
    return old_value;
}

template<uint32_t csr>
inline uint32_t clear_csr_bits(uint32_t mask) noexcept {
    uint32_t old_value;
    asm volatile("csrrc %0, %1, %2" : "=r"(old_value) : "i"(csr), "r"(mask));
    return old_value;
}

// System initialization
void system_init() noexcept {
    // Initialize basic system components
    // This can be extended based on specific RISC-V implementation
    
    // Clear any pending interrupts
    // Platform-specific interrupt controller setup would go here
    
    // Initialize GPIO if needed
    // gpio_init_all();
    
    // Memory barriers to ensure initialization order
    memory_barrier();
}

void system_reset() noexcept {
    // Perform software reset
    // Implementation depends on specific RISC-V platform
    
    // For now, just halt
    system_halt();
}

void system_halt() noexcept {
    // Disable interrupts and halt
    disable_interrupts();
    
    while (true) {
        wfi(); // Wait for interrupt (low power mode)
    }
}

// Interrupt control
InterruptState disable_interrupts() noexcept {
    // RISC-V uses mstatus.MIE bit (bit 3) for global interrupt enable
    constexpr uint32_t MSTATUS_MIE = 0x8;
    uint32_t old_status = clear_csr_bits<0x300>(MSTATUS_MIE); // mstatus CSR = 0x300
    return (old_status & MSTATUS_MIE) ? InterruptState::ENABLED : InterruptState::DISABLED;
}

void restore_interrupts(InterruptState previous_state) noexcept {
    if (previous_state == InterruptState::ENABLED) {
        constexpr uint32_t MSTATUS_MIE = 0x8;
        set_csr_bits<0x300>(MSTATUS_MIE);
    }
}

bool interrupts_enabled() noexcept {
    constexpr uint32_t MSTATUS_MIE = 0x8;
    return (read_csr<0x300>() & MSTATUS_MIE) != 0;
}

// Delay functions
void delay_cycles(uint64_t cycles) noexcept {
    if (cycles == 0) return;
    
    uint64_t start_cycles = read_cycle_counter();
    uint64_t target_cycles = start_cycles + cycles;
    
    // Handle potential overflow
    if (target_cycles < start_cycles) {
        // Wait for overflow
        while (read_cycle_counter() >= start_cycles) {
            nop();
        }
        start_cycles = 0;
        target_cycles = cycles - (UINT64_MAX - start_cycles);
    }
    
    while (read_cycle_counter() < target_cycles) {
        nop();
    }
}

void delay_microseconds(uint32_t microseconds) noexcept {
    if (microseconds == 0) return;
    
    uint64_t cycles = static_cast<uint64_t>(microseconds) * CYCLES_PER_MICROSECOND;
    delay_cycles(cycles);
}

void delay_milliseconds(uint32_t milliseconds) noexcept {
    if (milliseconds == 0) return;
    
    // Break down large delays to avoid overflow
    while (milliseconds >= 1000) {
        delay_microseconds(1000000); // 1 second
        milliseconds -= 1000;
    }
    
    if (milliseconds > 0) {
        delay_microseconds(milliseconds * 1000);
    }
}

// GPIO functions (basic implementation - needs platform-specific adaptation)
// These are placeholder implementations that need to be adapted for specific RISC-V platforms

// Placeholder GPIO register addresses - replace with actual platform values
static volatile uint32_t* const GPIO_BASE = reinterpret_cast<volatile uint32_t*>(0x10060000);
static constexpr uint32_t GPIO_INPUT_VAL_OFFSET  = 0x00;
static constexpr uint32_t GPIO_INPUT_EN_OFFSET   = 0x04;
static constexpr uint32_t GPIO_OUTPUT_EN_OFFSET  = 0x08;
static constexpr uint32_t GPIO_OUTPUT_VAL_OFFSET = 0x0C;

bool gpio_init(uint32_t pin, PinDirection direction) noexcept {
    if (pin >= 32) return false; // Assuming 32 GPIO pins max
    
    gpio_set_direction(pin, direction);
    return true;
}

void gpio_set_direction(uint32_t pin, PinDirection direction) noexcept {
    if (pin >= 32) return;
    
    volatile uint32_t* input_en_reg = GPIO_BASE + (GPIO_INPUT_EN_OFFSET / 4);
    volatile uint32_t* output_en_reg = GPIO_BASE + (GPIO_OUTPUT_EN_OFFSET / 4);
    
    uint32_t pin_mask = 1U << pin;
    
    if (direction == PinDirection::INPUT) {
        *input_en_reg |= pin_mask;
        *output_en_reg &= ~pin_mask;
    } else {
        *output_en_reg |= pin_mask;
        *input_en_reg &= ~pin_mask;
    }
}

void gpio_write(uint32_t pin, PinState state) noexcept {
    if (pin >= 32) return;
    
    volatile uint32_t* output_val_reg = GPIO_BASE + (GPIO_OUTPUT_VAL_OFFSET / 4);
    uint32_t pin_mask = 1U << pin;
    
    if (state == PinState::HIGH) {
        *output_val_reg |= pin_mask;
    } else {
        *output_val_reg &= ~pin_mask;
    }
}

PinState gpio_read(uint32_t pin) noexcept {
    if (pin >= 32) return PinState::LOW;
    
    volatile uint32_t* input_val_reg = GPIO_BASE + (GPIO_INPUT_VAL_OFFSET / 4);
    uint32_t pin_mask = 1U << pin;
    
    return (*input_val_reg & pin_mask) ? PinState::HIGH : PinState::LOW;
}

void gpio_toggle(uint32_t pin) noexcept {
    if (pin >= 32) return;
    
    PinState current_state = gpio_read(pin);
    gpio_write(pin, (current_state == PinState::HIGH) ? PinState::LOW : PinState::HIGH);
}

// Cache operations (if available - most basic RISC-V cores don't have caches)
void icache_invalidate() noexcept {
    // Most basic RISC-V implementations don't have instruction cache
    // This would be platform-specific
    instruction_barrier();
}

void dcache_clean() noexcept {
    // Most basic RISC-V implementations don't have data cache
    // This would be platform-specific
    memory_barrier();
}

void dcache_invalidate() noexcept {
    // Most basic RISC-V implementations don't have data cache
    // This would be platform-specific
    memory_barrier();
}

void dcache_clean_invalidate() noexcept {
    dcache_clean();
    dcache_invalidate();
}

// CPU feature detection
bool has_float_extension() noexcept {
    // Check misa CSR for F extension (bit 5)
    uint32_t misa = read_csr<0x301>(); // misa CSR = 0x301
    return (misa & (1U << 5)) != 0;
}

bool has_double_extension() noexcept {
    // Check misa CSR for D extension (bit 3)
    uint32_t misa = read_csr<0x301>();
    return (misa & (1U << 3)) != 0;
}

bool has_multiply_extension() noexcept {
    // Check misa CSR for M extension (bit 12)
    uint32_t misa = read_csr<0x301>();
    return (misa & (1U << 12)) != 0;
}

bool has_atomic_extension() noexcept {
    // Check misa CSR for A extension (bit 0)
    uint32_t misa = read_csr<0x301>();
    return (misa & (1U << 0)) != 0;
}

bool has_compressed_extension() noexcept {
    // Check misa CSR for C extension (bit 2)
    uint32_t misa = read_csr<0x301>();
    return (misa & (1U << 2)) != 0;
}

} // namespace cmx::platform::riscv