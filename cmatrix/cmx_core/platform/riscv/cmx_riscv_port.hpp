#pragma once

#include <cstdint>
#include <cstddef>

namespace cmx::platform::riscv {

// RISC-V CSR (Control and Status Register) definitions
constexpr uint32_t CSR_CYCLE = 0xC00;
constexpr uint32_t CSR_CYCLEH = 0xC80;
constexpr uint32_t CSR_TIME = 0xC01;
constexpr uint32_t CSR_TIMEH = 0xC81;
constexpr uint32_t CSR_INSTRET = 0xC02;
constexpr uint32_t CSR_INSTRETH = 0xC82;

// Memory barrier types
enum class MemoryBarrier : uint32_t {
    READ_WRITE = 0x33,  // Full fence
    READ_ONLY = 0x11,   // Read fence
    WRITE_ONLY = 0x22   // Write fence
};

// GPIO pin state
enum class PinState : uint8_t {
    LOW = 0,
    HIGH = 1
};

// GPIO pin direction
enum class PinDirection : uint8_t {
    INPUT = 0,
    OUTPUT = 1
};

// Interrupt control
enum class InterruptState : bool {
    DISABLED = false,
    ENABLED = true
};

// Core system functions
void system_init() noexcept;
void system_reset() noexcept;
void system_halt() noexcept;

// Memory barrier operations
inline void memory_barrier(MemoryBarrier type = MemoryBarrier::READ_WRITE) noexcept {
    asm volatile("fence %0, %0" :: "i"(static_cast<uint32_t>(type)) : "memory");
}

inline void instruction_barrier() noexcept {
    asm volatile("fence.i" ::: "memory");
}

// Cycle counter access
inline uint64_t read_cycle_counter() noexcept {
    uint32_t low, high1, high2;
    
    do {
        asm volatile("csrr %0, %1" : "=r"(high1) : "i"(CSR_CYCLEH));
        asm volatile("csrr %0, %1" : "=r"(low) : "i"(CSR_CYCLE));
        asm volatile("csrr %0, %1" : "=r"(high2) : "i"(CSR_CYCLEH));
    } while (high1 != high2);
    
    return (static_cast<uint64_t>(high1) << 32) | low;
}

inline uint64_t read_instruction_counter() noexcept {
    uint32_t low, high1, high2;
    
    do {
        asm volatile("csrr %0, %1" : "=r"(high1) : "i"(CSR_INSTRETH));
        asm volatile("csrr %0, %1" : "=r"(low) : "i"(CSR_INSTRET));
        asm volatile("csrr %0, %1" : "=r"(high2) : "i"(CSR_INSTRETH));
    } while (high1 != high2);
    
    return (static_cast<uint64_t>(high1) << 32) | low;
}

// Time access
inline uint64_t read_time() noexcept {
    uint32_t low, high1, high2;
    
    do {
        asm volatile("csrr %0, %1" : "=r"(high1) : "i"(CSR_TIMEH));
        asm volatile("csrr %0, %1" : "=r"(low) : "i"(CSR_TIME));
        asm volatile("csrr %0, %1" : "=r"(high2) : "i"(CSR_TIMEH));
    } while (high1 != high2);
    
    return (static_cast<uint64_t>(high1) << 32) | low;
}

// Interrupt control
InterruptState disable_interrupts() noexcept;
void restore_interrupts(InterruptState previous_state) noexcept;
bool interrupts_enabled() noexcept;

// RAII interrupt guard
class InterruptGuard {
    InterruptState previous_state_;
public:
    InterruptGuard() noexcept : previous_state_(disable_interrupts()) {}
    ~InterruptGuard() noexcept { restore_interrupts(previous_state_); }
    
    // Non-copyable, non-movable
    InterruptGuard(const InterruptGuard&) = delete;
    InterruptGuard& operator=(const InterruptGuard&) = delete;
    InterruptGuard(InterruptGuard&&) = delete;
    InterruptGuard& operator=(InterruptGuard&&) = delete;
};

// Delay functions
void delay_cycles(uint64_t cycles) noexcept;
void delay_microseconds(uint32_t microseconds) noexcept;
void delay_milliseconds(uint32_t milliseconds) noexcept;

// GPIO functions (basic implementation)
bool gpio_init(uint32_t pin, PinDirection direction) noexcept;
void gpio_set_direction(uint32_t pin, PinDirection direction) noexcept;
void gpio_write(uint32_t pin, PinState state) noexcept;
PinState gpio_read(uint32_t pin) noexcept;
void gpio_toggle(uint32_t pin) noexcept;

// Cache operations (if available)
void icache_invalidate() noexcept;
void dcache_clean() noexcept;
void dcache_invalidate() noexcept;
void dcache_clean_invalidate() noexcept;

// Platform-specific optimized operations
inline void nop() noexcept {
    asm volatile("nop");
}

inline void wfi() noexcept {  // Wait for interrupt
    asm volatile("wfi");
}

inline void yield() noexcept {
    // Yield to other threads/processes if in multitasking environment
    nop();
}

// Compiler barriers
inline void compiler_barrier() noexcept {
    asm volatile("" ::: "memory");
}

// Platform identification
constexpr const char* platform_name() noexcept {
    return "RISC-V";
}

constexpr const char* platform_version() noexcept {
    return "1.0.0";
}

// CPU feature detection
bool has_float_extension() noexcept;
bool has_double_extension() noexcept;
bool has_multiply_extension() noexcept;
bool has_atomic_extension() noexcept;
bool has_compressed_extension() noexcept;

// Performance hint macros
#define CMX_LIKELY(x)   __builtin_expect(!!(x), 1)
#define CMX_UNLIKELY(x) __builtin_expect(!!(x), 0)
#define CMX_PREFETCH(addr, rw, locality) __builtin_prefetch(addr, rw, locality)

} // namespace cmx::platform::riscv