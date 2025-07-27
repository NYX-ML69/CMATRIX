#pragma once

#include <cstdint>
#include <cstddef>

namespace cmx {
namespace platform {
namespace cortex_m {

/**
 * @brief Platform initialization status codes
 */
enum class InitStatus : uint8_t {
    SUCCESS = 0,
    CLOCK_INIT_FAILED,
    TIMER_INIT_FAILED,
    DMA_INIT_FAILED,
    MEMORY_INIT_FAILED
};

/**
 * @brief GPIO pin configuration for debugging/status LEDs
 */
struct GpioConfig {
    uint32_t port;      ///< GPIO port base address
    uint16_t pin;       ///< Pin number (0-15)
    bool active_high;   ///< True if LED is active high
};

/**
 * @brief Platform-specific configuration structure
 */
struct PlatformConfig {
    uint32_t cpu_freq_hz;           ///< CPU frequency in Hz
    uint32_t systick_freq_hz;       ///< SysTick frequency in Hz
    GpioConfig status_led;          ///< Optional status LED config
    bool enable_dwt_cycle_counter;  ///< Enable DWT cycle counter for profiling
    bool enable_cache;              ///< Enable instruction/data cache if available
};

/**
 * @brief Initialize the Cortex-M platform
 * @param config Platform configuration parameters
 * @return InitStatus indicating success or failure reason
 */
InitStatus init(const PlatformConfig& config);

/**
 * @brief Deinitialize platform resources
 */
void deinit();

/**
 * @brief Blocking delay in milliseconds
 * @param ms Milliseconds to delay
 */
void delay_ms(uint32_t ms);

/**
 * @brief Blocking delay in microseconds
 * @param us Microseconds to delay
 */
void delay_us(uint32_t us);

/**
 * @brief Get system uptime in milliseconds
 * @return Milliseconds since platform initialization
 */
uint32_t get_millis();

/**
 * @brief Get high-resolution timestamp in microseconds
 * @return Microseconds timestamp (may wrap around)
 */
uint64_t get_micros();

/**
 * @brief Get CPU cycle count (if DWT is enabled)
 * @return Current CPU cycle count
 */
uint32_t get_cpu_cycles();

/**
 * @brief Enter critical section (disable interrupts)
 * @return Previous interrupt mask state
 */
uint32_t enter_critical();

/**
 * @brief Exit critical section (restore interrupts)
 * @param mask Previous interrupt mask state from enter_critical()
 */
void exit_critical(uint32_t mask);

/**
 * @brief Set status LED state
 * @param on True to turn LED on, false to turn off
 */
void set_status_led(bool on);

/**
 * @brief Toggle status LED
 */
void toggle_status_led();

/**
 * @brief Trigger system reset
 */
void system_reset() __attribute__((noreturn));

/**
 * @brief Check if running in interrupt context
 * @return True if currently in an interrupt handler
 */
bool in_interrupt();

/**
 * @brief Get available RAM in bytes
 * @return Free RAM size in bytes
 */
size_t get_free_ram();

/**
 * @brief Get stack usage statistics
 * @return Stack high water mark in bytes
 */
size_t get_stack_usage();

} // namespace cortex_m
} // namespace platform
} // namespace cmx