#pragma once

#include <cstdint>
#include <chrono>

namespace cmx::platform::riscv {

/**
 * @brief RISC-V Timer abstraction for delays and profiling
 * 
 * Provides high-resolution timing functions using RISC-V cycle counter
 * or hardware timers. Used by cmx_profile for layer execution timing.
 */
class Timer {
public:
    using cycles_t = uint64_t;
    using microseconds_t = uint64_t;
    using nanoseconds_t = uint64_t;

    /**
     * @brief Initialize the timer subsystem
     * @return true if initialization successful, false otherwise
     */
    static bool initialize();

    /**
     * @brief Get current cycle count from RISC-V cycle counter
     * @return Current cycle count
     */
    static inline cycles_t get_cycles() {
        cycles_t cycles;
        asm volatile("rdcycle %0" : "=r"(cycles));
        return cycles;
    }

    /**
     * @brief Get current time in microseconds
     * @return Current time in microseconds since initialization
     */
    static microseconds_t get_microseconds();

    /**
     * @brief Get current time in nanoseconds
     * @return Current time in nanoseconds since initialization
     */
    static nanoseconds_t get_nanoseconds();

    /**
     * @brief Convert cycles to microseconds
     * @param cycles Number of cycles to convert
     * @return Equivalent time in microseconds
     */
    static inline microseconds_t cycles_to_microseconds(cycles_t cycles) {
        return (cycles * 1000000ULL) / cpu_frequency_hz_;
    }

    /**
     * @brief Convert cycles to nanoseconds
     * @param cycles Number of cycles to convert
     * @return Equivalent time in nanoseconds
     */
    static inline nanoseconds_t cycles_to_nanoseconds(cycles_t cycles) {
        return (cycles * 1000000000ULL) / cpu_frequency_hz_;
    }

    /**
     * @brief Busy-wait delay for specified microseconds
     * @param microseconds Number of microseconds to delay
     */
    static void delay_microseconds(microseconds_t microseconds);

    /**
     * @brief Busy-wait delay for specified nanoseconds
     * @param nanoseconds Number of nanoseconds to delay
     */
    static void delay_nanoseconds(nanoseconds_t nanoseconds);

    /**
     * @brief Busy-wait delay for specified number of cycles
     * @param cycles Number of CPU cycles to delay
     */
    static inline void delay_cycles(cycles_t cycles) {
        cycles_t start = get_cycles();
        while ((get_cycles() - start) < cycles) {
            asm volatile("nop");
        }
    }

    /**
     * @brief Get CPU frequency in Hz
     * @return CPU frequency in Hz
     */
    static inline uint64_t get_cpu_frequency() {
        return cpu_frequency_hz_;
    }

    /**
     * @brief Set CPU frequency (must be called during initialization)
     * @param freq_hz CPU frequency in Hz
     */
    static void set_cpu_frequency(uint64_t freq_hz);

private:
    static uint64_t cpu_frequency_hz_;
    static cycles_t init_cycles_;
    static bool initialized_;
};

/**
 * @brief RAII timer class for profiling code blocks
 * 
 * Usage:
 * {
 *     ProfileTimer timer;
 *     // ... code to profile ...
 *     auto elapsed_us = timer.elapsed_microseconds();
 * }
 */
class ProfileTimer {
public:
    ProfileTimer() : start_cycles_(Timer::get_cycles()) {}

    /**
     * @brief Get elapsed time since construction in cycles
     * @return Elapsed cycles
     */
    inline Timer::cycles_t elapsed_cycles() const {
        return Timer::get_cycles() - start_cycles_;
    }

    /**
     * @brief Get elapsed time since construction in microseconds
     * @return Elapsed microseconds
     */
    inline Timer::microseconds_t elapsed_microseconds() const {
        return Timer::cycles_to_microseconds(elapsed_cycles());
    }

    /**
     * @brief Get elapsed time since construction in nanoseconds
     * @return Elapsed nanoseconds
     */
    inline Timer::nanoseconds_t elapsed_nanoseconds() const {
        return Timer::cycles_to_nanoseconds(elapsed_cycles());
    }

    /**
     * @brief Reset the timer to current time
     */
    inline void reset() {
        start_cycles_ = Timer::get_cycles();
    }

private:
    Timer::cycles_t start_cycles_;
};

} // namespace cmx::platform::riscv