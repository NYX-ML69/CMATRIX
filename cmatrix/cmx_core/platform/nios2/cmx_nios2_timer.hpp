#pragma once

#include <cstdint>

namespace cmx {
namespace platform {
namespace nios2 {

/**
 * @brief Timer configuration for Nios II systems
 */
struct TimerConfig {
    static constexpr uint32_t TIMER_FREQ_HZ = 50000000;    // 50MHz default system clock
    static constexpr uint32_t US_PER_SECOND = 1000000;
    static constexpr uint32_t MS_PER_SECOND = 1000;
};

/**
 * @brief Timer statistics for performance monitoring
 */
struct TimerStats {
    uint64_t total_delay_calls;
    uint64_t total_delay_time_us;
    uint64_t max_delay_time_us;
    uint32_t timer_overflows;
};

/**
 * @brief Initialize the timer subsystem
 * Must be called before using any timer functions
 * @param system_freq_hz System clock frequency in Hz (0 = use default)
 */
void cmx_timer_init(uint32_t system_freq_hz = 0);

/**
 * @brief Get current timestamp in microseconds
 * @return Current time in microseconds since timer initialization
 */
uint64_t cmx_now_us();

/**
 * @brief Get current timestamp in milliseconds
 * @return Current time in milliseconds since timer initialization
 */
uint64_t cmx_now_ms();

/**
 * @brief Delay execution for specified milliseconds
 * @param ms Number of milliseconds to delay
 */
void cmx_delay_ms(uint32_t ms);

/**
 * @brief Delay execution for specified microseconds
 * @param us Number of microseconds to delay
 */
void cmx_delay_us(uint32_t us);

/**
 * @brief Non-blocking delay check
 * @param start_time_us Start time from cmx_now_us()
 * @param duration_us Desired delay duration in microseconds
 * @return true if delay period has elapsed, false otherwise
 */
bool cmx_delay_elapsed(uint64_t start_time_us, uint32_t duration_us);

/**
 * @brief Get timer statistics for performance monitoring
 * @return Const reference to timer statistics
 */
const TimerStats& cmx_timer_get_stats();

/**
 * @brief Reset timer statistics
 */
void cmx_timer_reset_stats();

/**
 * @brief High-precision timestamp for benchmarking
 * @return Raw timer counter value
 */
uint64_t cmx_get_raw_timer();

/**
 * @brief Convert raw timer ticks to microseconds
 * @param ticks Raw timer counter value
 * @return Time in microseconds
 */
uint64_t cmx_ticks_to_us(uint64_t ticks);

/**
 * @brief Convert microseconds to raw timer ticks
 * @param us Time in microseconds
 * @return Raw timer counter value
 */
uint64_t cmx_us_to_ticks(uint64_t us);

/**
 * @brief Calibrate timer frequency by measuring against known delay
 * @param reference_delay_ms Known delay in milliseconds
 * @return Measured frequency in Hz, 0 if calibration failed
 */
uint32_t cmx_timer_calibrate(uint32_t reference_delay_ms = 100);

/**
 * @brief Check if timer hardware is available and functioning
 * @return true if timer is working correctly, false otherwise
 */
bool cmx_timer_self_test();

/**
 * @brief Get the current timer frequency setting
 * @return Timer frequency in Hz
 */
uint32_t cmx_timer_get_frequency();

} // namespace nios2
} // namespace platform
} // namespace cmx