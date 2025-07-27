#pragma once

#include <cstdint>

namespace cmx::platform::zephyr {

/**
 * @brief Initialize the timer subsystem
 * 
 * Sets up timing infrastructure and calibrates timing sources.
 * Must be called before using other timer functions.
 */
void cmx_timer_init();

/**
 * @brief Get current timestamp in microseconds
 * 
 * @return uint64_t Current timestamp in microseconds since system boot
 */
uint64_t cmx_now_us();

/**
 * @brief Get current timestamp in milliseconds
 * 
 * @return uint64_t Current timestamp in milliseconds since system boot
 */
uint64_t cmx_now_ms();

/**
 * @brief Delay execution for specified milliseconds
 * 
 * This function yields the CPU to other threads during the delay.
 * 
 * @param ms Milliseconds to delay
 */
void cmx_delay_ms(uint32_t ms);

/**
 * @brief Delay execution for specified microseconds
 * 
 * This function performs a busy wait for precise timing.
 * Use sparingly as it blocks the CPU.
 * 
 * @param us Microseconds to delay
 */
void cmx_delay_us(uint32_t us);

/**
 * @brief Get system tick frequency
 * 
 * @return uint32_t Ticks per second
 */
uint32_t cmx_get_tick_frequency();

/**
 * @brief Convert ticks to microseconds
 * 
 * @param ticks Number of system ticks
 * @return uint64_t Equivalent microseconds
 */
uint64_t cmx_ticks_to_us(uint64_t ticks);

/**
 * @brief Convert microseconds to ticks
 * 
 * @param us Microseconds
 * @return uint64_t Equivalent system ticks
 */
uint64_t cmx_us_to_ticks(uint64_t us);

/**
 * @brief High-resolution timer class for profiling
 * 
 * Provides start/stop timing functionality for performance measurement.
 */
class CmxTimer {
public:
    /**
     * @brief Construct a new timer
     */
    CmxTimer();

    /**
     * @brief Start the timer
     */
    void start();

    /**
     * @brief Stop the timer and return elapsed time
     * 
     * @return uint64_t Elapsed time in microseconds
     */
    uint64_t stop();

    /**
     * @brief Get elapsed time without stopping
     * 
     * @return uint64_t Elapsed time in microseconds since start()
     */
    uint64_t elapsed() const;

    /**
     * @brief Reset the timer
     */
    void reset();

    /**
     * @brief Check if timer is currently running
     * 
     * @return true Timer is running
     * @return false Timer is stopped
     */
    bool is_running() const;

private:
    uint64_t start_time_;
    uint64_t stop_time_;
    bool running_;
};

/**
 * @brief Performance profiler class
 * 
 * Accumulates timing statistics over multiple measurements.
 */
class CmxProfiler {
public:
    /**
     * @brief Construct a new profiler
     * 
     * @param name Name for this profiler instance
     */
    explicit CmxProfiler(const char* name);

    /**
     * @brief Start a measurement
     */
    void start();

    /**
     * @brief Stop current measurement and add to statistics
     */
    void stop();

    /**
     * @brief Get average time per measurement
     * 
     * @return uint64_t Average time in microseconds
     */
    uint64_t get_average_us() const;

    /**
     * @brief Get minimum recorded time
     * 
     * @return uint64_t Minimum time in microseconds
     */
    uint64_t get_min_us() const;

    /**
     * @brief Get maximum recorded time
     * 
     * @return uint64_t Maximum time in microseconds
     */
    uint64_t get_max_us() const;

    /**
     * @brief Get total number of measurements
     * 
     * @return uint32_t Number of measurements
     */
    uint32_t get_count() const;

    /**
     * @brief Reset all statistics
     */
    void reset();

    /**
     * @brief Print statistics to log
     */
    void print_stats() const;

private:
    const char* name_;
    CmxTimer timer_;
    uint64_t total_time_;
    uint64_t min_time_;
    uint64_t max_time_;
    uint32_t count_;
};

/**
 * @brief RAII timer for automatic profiling
 * 
 * Automatically starts timing on construction and stops on destruction.
 */
class CmxScopedTimer {
public:
    /**
     * @brief Construct and start timing
     * 
     * @param profiler Profiler to add timing to (optional)
     */
    explicit CmxScopedTimer(CmxProfiler* profiler = nullptr);

    /**
     * @brief Destruct and stop timing
     */
    ~CmxScopedTimer();

    /**
     * @brief Get elapsed time
     * 
     * @return uint64_t Elapsed time in microseconds
     */
    uint64_t elapsed() const;

private:
    CmxTimer timer_;
    CmxProfiler* profiler_;
};

/**
 * @brief Watchdog timer for detecting timeouts
 */
class CmxWatchdog {
public:
    /**
     * @brief Construct a watchdog timer
     * 
     * @param timeout_ms Timeout in milliseconds
     */
    explicit CmxWatchdog(uint32_t timeout_ms);

    /**
     * @brief Start the watchdog
     */
    void start();

    /**
     * @brief Feed/reset the watchdog
     */
    void feed();

    /**
     * @brief Check if watchdog has timed out
     * 
     * @return true Timeout occurred
     * @return false Still within timeout period
     */
    bool has_timeout();

    /**
     * @brief Stop the watchdog
     */
    void stop();

private:
    uint32_t timeout_ms_;
    uint64_t last_feed_time_;
    bool active_;
};

} // namespace cmx::platform::zephyr