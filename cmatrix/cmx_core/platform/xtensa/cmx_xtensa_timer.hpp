#pragma once

#include <cstdint>
#include <cstddef>

namespace cmx::platform::xtensa {

/**
 * @brief Timer callback function type
 */
using TimerCallback = void(*)(void* user_data);

/**
 * @brief Timer handle for managing periodic timers
 */
struct TimerHandle {
    uint8_t timer_id;
    bool active;
    bool periodic;
    uint64_t interval_us;
    uint64_t next_trigger_us;
    TimerCallback callback;
    void* user_data;
};

/**
 * @brief Initialize timer subsystem
 */
void timer_init();

/**
 * @brief Get current timestamp in microseconds
 * @return Current timestamp since system start
 */
uint64_t cmx_now_us();

/**
 * @brief Get current timestamp in milliseconds
 * @return Current timestamp since system start
 */
uint64_t cmx_now_ms();

/**
 * @brief Blocking delay in milliseconds
 * @param ms Milliseconds to delay
 */
void cmx_delay_ms(uint32_t ms);

/**
 * @brief Blocking delay in microseconds
 * @param us Microseconds to delay
 */
void cmx_delay_us(uint32_t us);

/**
 * @brief Non-blocking delay check
 * @param start_time_us Start time in microseconds
 * @param delay_us Delay duration in microseconds
 * @return true if delay period has elapsed
 */
bool cmx_delay_elapsed(uint64_t start_time_us, uint32_t delay_us);

/**
 * @brief Create a periodic timer
 * @param interval_us Timer interval in microseconds
 * @param callback Function to call on timer expiry
 * @param user_data User data passed to callback
 * @return Timer handle, nullptr if failed
 */
TimerHandle* cmx_timer_create_periodic(uint64_t interval_us, 
                                      TimerCallback callback, 
                                      void* user_data);

/**
 * @brief Create a one-shot timer
 * @param delay_us Delay before timer fires in microseconds
 * @param callback Function to call on timer expiry
 * @param user_data User data passed to callback
 * @return Timer handle, nullptr if failed
 */
TimerHandle* cmx_timer_create_oneshot(uint64_t delay_us,
                                     TimerCallback callback,
                                     void* user_data);

/**
 * @brief Start or restart a timer
 * @param handle Timer handle
 * @return true if started successfully
 */
bool cmx_timer_start(TimerHandle* handle);

/**
 * @brief Stop a timer
 * @param handle Timer handle
 */
void cmx_timer_stop(TimerHandle* handle);

/**
 * @brief Delete a timer
 * @param handle Timer handle
 */
void cmx_timer_delete(TimerHandle* handle);

/**
 * @brief Update timer interval
 * @param handle Timer handle
 * @param new_interval_us New interval in microseconds
 * @return true if updated successfully
 */
bool cmx_timer_set_interval(TimerHandle* handle, uint64_t new_interval_us);

/**
 * @brief Check if timer is active
 * @param handle Timer handle
 * @return true if timer is active
 */
bool cmx_timer_is_active(const TimerHandle* handle);

/**
 * @brief Get time remaining until next timer trigger
 * @param handle Timer handle
 * @return Microseconds until next trigger, 0 if not active
 */
uint64_t cmx_timer_time_remaining(const TimerHandle* handle);

/**
 * @brief Process timer callbacks (called from main loop)
 */
void cmx_timer_process();

/**
 * @brief High-precision benchmark timer
 */
class BenchmarkTimer {
public:
    BenchmarkTimer();
    
    void start();
    void stop();
    void reset();
    
    uint64_t get_elapsed_us() const;
    uint64_t get_elapsed_ns() const;
    double get_elapsed_ms() const;
    
    bool is_running() const;
    
private:
    uint64_t start_time_us_;
    uint64_t elapsed_us_;
    bool running_;
};

/**
 * @brief Timer statistics
 */
struct TimerStats {
    uint32_t active_timers;
    uint32_t total_timers_created;
    uint32_t timer_callbacks_executed;
    uint64_t total_callback_time_us;
    uint32_t max_callback_time_us;
    uint32_t timer_overruns;
};

/**
 * @brief Get timer subsystem statistics
 * @return Timer statistics
 */
const TimerStats& cmx_timer_get_stats();

/**
 * @brief Reset timer statistics
 */
void cmx_timer_reset_stats();

/**
 * @brief Watchdog timer functionality
 */
class WatchdogTimer {
public:
    /**
     * @brief Initialize watchdog timer
     * @param timeout_ms Timeout in milliseconds
     * @param callback Optional callback before reset
     */
    WatchdogTimer(uint32_t timeout_ms, TimerCallback callback = nullptr);
    
    /**
     * @brief Feed/kick the watchdog
     */
    void feed();
    
    /**
     * @brief Enable watchdog
     */
    void enable();
    
    /**
     * @brief Disable watchdog
     */
    void disable();
    
    /**
     * @brief Check if watchdog is enabled
     */
    bool is_enabled() const;
    
    /**
     * @brief Get time since last feed
     * @return Milliseconds since last feed
     */
    uint32_t get_time_since_feed() const;
    
private:
    uint32_t timeout_ms_;
    uint64_t last_feed_time_us_;
    TimerCallback callback_;
    bool enabled_;
};

/**
 * @brief Rate limiter using timer
 */
class RateLimiter {
public:
    /**
     * @brief Initialize rate limiter
     * @param max_rate_hz Maximum rate in Hz
     */
    explicit RateLimiter(uint32_t max_rate_hz);
    
    /**
     * @brief Check if action is allowed
     * @return true if action should proceed, false if rate limited
     */
    bool allow();
    
    /**
     * @brief Reset rate limiter
     */
    void reset();
    
    /**
     * @brief Set new rate limit
     * @param max_rate_hz New maximum rate in Hz
     */
    void set_rate(uint32_t max_rate_hz);
    
private:
    uint64_t interval_us_;
    uint64_t last_action_time_us_;
};

/**
 * @brief Profiling timer for measuring execution time
 */
class ProfileTimer {
public:
    ProfileTimer(const char* name);
    ~ProfileTimer();
    
    void begin();
    void end();
    
    uint64_t get_total_time_us() const;
    uint32_t get_call_count() const;
    uint64_t get_average_time_us() const;
    uint64_t get_min_time_us() const;
    uint64_t get_max_time_us() const;
    
    void reset();
    
private:
    const char* name_;
    uint64_t start_time_us_;
    uint64_t total_time_us_;
    uint64_t min_time_us_;
    uint64_t max_time_us_;
    uint32_t call_count_;
    bool active_;
};

/**
 * @brief Timeout helper class
 */
class Timeout {
public:
    /**
     * @brief Create timeout
     * @param timeout_us Timeout in microseconds
     */
    explicit Timeout(uint64_t timeout_us);
    
    /**
     * @brief Check if timeout has occurred
     * @return true if timeout has occurred
     */
    bool expired() const;
    
    /**
     * @brief Reset timeout
     */
    void reset();
    
    /**
     * @brief Get remaining time
     * @return Microseconds remaining, 0 if expired
     */
    uint64_t remaining() const;
    
private:
    uint64_t timeout_us_;
    uint64_t start_time_us_;
};

/**
 * @brief Timer configuration
 */
struct TimerConfig {
    uint32_t max_timers;
    bool enable_statistics;
    bool enable_profiling;
    uint32_t timer_resolution_us;
};

/**
 * @brief Configure timer subsystem
 * @param config Timer configuration
 * @return true if configured successfully
 */
bool cmx_timer_configure(const TimerConfig& config);

/**
 * @brief Get timer resolution in microseconds
 * @return Timer resolution
 */
uint32_t cmx_timer_get_resolution();

/**
 * @brief Calibrate timer against external reference
 * @param reference_freq_hz Reference frequency in Hz
 * @return Calibration factor (1.0 = perfect calibration)
 */
double cmx_timer_calibrate(uint32_t reference_freq_hz);

} // namespace cmx::platform::xtensa

// Convenience macros
#define CMX_BENCHMARK_START() cmx::platform::xtensa::BenchmarkTimer _bench; _bench.start()
#define CMX_BENCHMARK_END() _bench.stop(); cmx_log("Benchmark: " + std::to_string(_bench.get_elapsed_us()) + "us")

#define CMX_PROFILE_FUNCTION() cmx::platform::xtensa::ProfileTimer _prof(__FUNCTION__)
#define CMX_PROFILE_SCOPE(name) cmx::platform::xtensa::ProfileTimer _prof(name)

#define CMX_TIMEOUT(us) cmx::platform::xtensa::Timeout _timeout(us)
#define CMX_TIMEOUT_CHECK() _timeout.expired()