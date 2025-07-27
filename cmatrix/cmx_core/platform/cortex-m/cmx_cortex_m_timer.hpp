#pragma once

#include <cstdint>

namespace cmx {
namespace platform {
namespace cortex_m {

/**
 * @brief High-resolution timer configuration options
 */
struct TimerConfig {
    uint32_t frequency_hz;      ///< Timer frequency in Hz
    bool enable_interrupts;     ///< Enable timer interrupts
    uint8_t interrupt_priority; ///< Interrupt priority (0-15)
};

/**
 * @brief Timer callback function type
 * @param timer_id Timer identifier that triggered the callback
 */
using TimerCallback = void (*)(uint32_t timer_id);

/**
 * @brief High-resolution hardware timer wrapper for Cortex-M
 * 
 * Provides microsecond and millisecond precision timing for CMX runtime
 * profiling and scheduling operations. Uses SysTick and/or hardware timers.
 */
class Timer {
public:
    /**
     * @brief Initialize the timer subsystem
     * @param config Timer configuration parameters
     * @return true if initialization successful, false otherwise
     */
    static bool init(const TimerConfig& config = {1000000, false, 15});
    
    /**
     * @brief Deinitialize timer subsystem
     */
    static void deinit();
    
    /**
     * @brief Get current millisecond timestamp
     * @return Milliseconds since timer initialization
     */
    static uint32_t get_millis();
    
    /**
     * @brief Get current microsecond timestamp
     * @return Microseconds since timer initialization
     */
    static uint64_t get_micros();
    
    /**
     * @brief Blocking delay in milliseconds
     * @param ms Milliseconds to delay
     */
    static void delay_ms(uint32_t ms);
    
    /**
     * @brief Blocking delay in microseconds
     * @param us Microseconds to delay
     */
    static void delay_us(uint32_t us);
    
    /**
     * @brief Start a one-shot timer
     * @param timeout_ms Timeout in milliseconds
     * @param callback Callback function to execute on timeout
     * @param timer_id Optional timer ID for callback identification
     * @return true if timer started successfully
     */
    static bool start_oneshot(uint32_t timeout_ms, TimerCallback callback, uint32_t timer_id = 0);
    
    /**
     * @brief Start a periodic timer
     * @param period_ms Period in milliseconds
     * @param callback Callback function to execute on each period
     * @param timer_id Optional timer ID for callback identification
     * @return true if timer started successfully
     */
    static bool start_periodic(uint32_t period_ms, TimerCallback callback, uint32_t timer_id = 0);
    
    /**
     * @brief Stop a running timer
     * @param timer_id Timer ID to stop
     */
    static void stop_timer(uint32_t timer_id);
    
    /**
     * @brief Reset timestamp counters to zero
     */
    static void reset_counters();
    
    /**
     * @brief Get timer resolution in microseconds
     * @return Timer resolution (e.g., 1 for 1μs resolution)
     */
    static uint32_t get_resolution_us();
    
    /**
     * @brief Check if timer subsystem is initialized
     * @return true if initialized, false otherwise
     */
    static bool is_initialized();

private:
    Timer() = delete;
    ~Timer() = delete;
    Timer(const Timer&) = delete;
    Timer& operator=(const Timer&) = delete;
};

/**
 * @brief RAII timer profiler helper class
 * 
 * Automatically measures execution time of a code block
 */
class ProfileTimer {
public:
    /**
     * @brief Start profiling timer
     */
    ProfileTimer();
    
    /**
     * @brief Stop profiling timer and get elapsed time
     */
    ~ProfileTimer() = default;
    
    /**
     * @brief Get elapsed time in microseconds
     * @return Elapsed microseconds since construction
     */
    uint64_t elapsed_us() const;
    
    /**
     * @brief Get elapsed time in milliseconds
     * @return Elapsed milliseconds since construction
     */
    uint32_t elapsed_ms() const;
    
    /**
     * @brief Reset the profiling timer
     */
    void reset();

private:
    uint64_t start_time_us_;
};

} // namespace cortex_m
} // namespace platform
} // namespace cmx