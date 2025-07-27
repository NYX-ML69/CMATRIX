#pragma once

#include <cstdint>
#include <cstddef>
#include <functional>

// Forward declarations
struct esp_timer_handle_t;
typedef struct esp_timer_handle_t* esp_timer_handle_t;

namespace cmx {
namespace platform {
namespace esp32 {

/**
 * @brief Timer operation status codes
 */
enum class TimerStatus : uint8_t {
    SUCCESS = 0,
    ERROR_INIT_FAILED,
    ERROR_INVALID_TIMER,
    ERROR_INVALID_PARAM,
    ERROR_TIMER_RUNNING,
    ERROR_TIMER_NOT_RUNNING,
    ERROR_TIMEOUT
};

/**
 * @brief Timer types
 */
enum class TimerType : uint8_t {
    ONE_SHOT = 0,    // Single execution
    PERIODIC         // Repeated execution
};

/**
 * @brief Hardware timer IDs
 */
enum class HWTimerID : uint8_t {
    TIMER_0 = 0,
    TIMER_1,
    TIMER_2,
    TIMER_3,
    MAX_TIMERS
};

/**
 * @brief Timer configuration structure
 */
struct TimerConfig {
    const char* name;           // Timer name for debugging
    TimerType type;            // Timer type (one-shot or periodic)
    uint64_t period_us;        // Timer period in microseconds
    bool auto_reload;          // Auto-reload for periodic timers
    bool skip_unhandled_events; // Skip missed events if handler is slow
    uint32_t dispatch_method;   // ESP timer dispatch method
};

/**
 * @brief Default timer configuration
 */
constexpr TimerConfig DEFAULT_TIMER_CONFIG = {
    .name = "cmx_timer",
    .type = TimerType::ONE_SHOT,
    .period_us = 1000,
    .auto_reload = true,
    .skip_unhandled_events = false,
    .dispatch_method = 0  // ESP_TIMER_TASK
};

/**
 * @brief Timer statistics
 */
struct TimerStats {
    uint64_t total_triggers;     // Total number of timer triggers
    uint64_t missed_triggers;    // Number of missed triggers
    uint32_t min_callback_time_us; // Minimum callback execution time
    uint32_t max_callback_time_us; // Maximum callback execution time
    uint32_t avg_callback_time_us; // Average callback execution time
    uint64_t total_runtime_us;   // Total time timer has been running
    uint32_t callback_overruns;  // Number of callback overruns
};

/**
 * @brief Profiling timer result
 */
struct ProfileResult {
    uint64_t start_time_us;      // Start timestamp
    uint64_t end_time_us;        // End timestamp
    uint64_t duration_us;        // Duration in microseconds
    uint32_t cpu_cycles;         // CPU cycles elapsed
    float duration_ms;           // Duration in milliseconds
};

/**
 * @brief Timer callback function type
 */
using TimerCallback = std::function<void(void* user_data)>;

/**
 * @brief ESP32 Timer Manager
 */
class ESP32TimerManager {
public:
    /**
     * @brief Initialize timer subsystem
     * @return TimerStatus::SUCCESS on success
     */
    static TimerStatus initialize();
    
    /**
     * @brief Shutdown timer subsystem
     * @return TimerStatus::SUCCESS on success
     */
    static TimerStatus shutdown();
    
    /**
     * @brief Check if timer subsystem is initialized
     * @return true if initialized
     */
    static bool is_initialized();
    
    /**
     * @brief Get system tick count in microseconds
     * @return Current tick count
     */
    static uint64_t get_tick_us();
    
    /**
     * @brief Get system tick count in milliseconds
     * @return Current tick count
     */
    static uint64_t get_tick_ms();
    
    /**
     * @brief Convert microseconds to system ticks
     * @param us Microseconds
     * @return System ticks
     */
    static uint64_t us_to_ticks(uint64_t us);
    
    /**
     * @brief Convert system ticks to microseconds
     * @param ticks System ticks
     * @return Microseconds
     */
    static uint64_t ticks_to_us(uint64_t ticks);

private:
    static bool initialized_;
};

/**
 * @brief Software Timer using ESP timer
 */
class SoftwareTimer {
public:
    /**
     * @brief Create software timer
     * @param config Timer configuration
     */
    explicit SoftwareTimer(const TimerConfig& config = DEFAULT_TIMER_CONFIG);
    
    /**
     * @brief Destructor - stops and deletes timer
     */
    ~SoftwareTimer();
    
    // Disable copy operations
    SoftwareTimer(const SoftwareTimer&) = delete;
    SoftwareTimer& operator=(const SoftwareTimer&) = delete;
    
    // Enable move operations
    SoftwareTimer(SoftwareTimer&& other) noexcept;
    SoftwareTimer& operator=(SoftwareTimer&& other) noexcept;
    
    /**
     * @brief Check if timer is valid
     * @return true if timer is ready to use
     */
    bool is_valid() const;
    
    /**
     * @brief Set timer callback function
     * @param callback Callback function
     * @param user_data User data passed to callback
     */
    void set_callback(TimerCallback callback, void* user_data = nullptr);
    
    /**
     * @brief Start the timer
     * @param delay_us Initial delay in microseconds (0 = use configured period)
     * @return TimerStatus::SUCCESS on success
     */
    TimerStatus start(uint64_t delay_us = 0);
    
    /**
     * @brief Stop the timer
     * @return TimerStatus::SUCCESS on success
     */
    TimerStatus stop();
    
    /**
     * @brief Restart the timer with new period
     * @param new_period_us New period in microseconds
     * @return TimerStatus::SUCCESS on success
     */
    TimerStatus restart(uint64_t new_period_us);
    
    /**
     * @brief Check if timer is running
     * @return true if timer is active
     */
    bool is_running() const;
    
    /**
     * @brief Get timer configuration
     * @return Current timer configuration
     */
    const TimerConfig& get_config() const;
    
    /**
     * @brief Get timer statistics
     * @return Timer statistics
     */
    TimerStats get_stats() const;
    
    /**
     * @brief Reset timer statistics
     */
    void reset_stats();
    
    /**
     * @brief Get time until next trigger
     * @return Microseconds until next trigger (0 if not running)
     */
    uint64_t get_time_until_trigger() const;

private:
    TimerConfig config_;
    esp_timer_handle_t timer_handle_;
    TimerCallback callback_;
    void* user_data_;
    bool initialized_;
    mutable TimerStats stats_;
    uint64_t last_trigger_time_;
    
    void handle_timer_callback();
    static void esp_timer_callback(void* arg);
    void update_callback_stats(uint32_t callback_time_us);
};

/**
 * @brief Hardware Timer for precise timing
 */
class HardwareTimer {
public:
    /**
     * @brief Create hardware timer
     * @param timer_id Hardware timer ID
     * @param resolution_us Timer resolution in microseconds
     */
    explicit HardwareTimer(HWTimerID timer_id, uint32_t resolution_us = 1);
    
    /**
     * @brief Destructor - deinitializes timer
     */
    ~HardwareTimer();
    
    // Disable copy operations
    HardwareTimer(const HardwareTimer&) = delete;
    HardwareTimer& operator=(const HardwareTimer&) = delete;
    
    /**
     * @brief Check if timer is valid
     * @return true if timer is ready to use
     */
    bool is_valid() const;
    
    /**
     * @brief Set timer callback for interrupt mode
     * @param callback Callback function
     * @param user_data User data passed to callback
     */
    void set_callback(TimerCallback callback, void* user_data = nullptr);
    
    /**
     * @brief Start timer in interrupt mode
     * @param period_us Timer period in microseconds
     * @param auto_reload Auto-reload timer
     * @return TimerStatus::SUCCESS on success
     */
    TimerStatus start_interrupt(uint64_t period_us, bool auto_reload = false);
    
    /**
     * @brief Start timer in polling mode
     * @param period_us Timer period in microseconds
     * @return TimerStatus::SUCCESS on success
     */
    TimerStatus start_polling(uint64_t period_us);
    
    /**
     * @brief Stop the timer
     * @return TimerStatus::SUCCESS on success
     */
    TimerStatus stop();
    
    /**
     * @brief Check if timer has triggered (polling mode)
     * @return true if timer has triggered
     */
    bool has_triggered();
    
    /**
     * @brief Clear timer trigger flag
     */
    void clear_trigger();
    
    /**
     * @brief Get current timer value
     * @return Current timer count
     */
    uint64_t get_counter_value() const;
    
    /**
     * @brief Set timer counter value
     * @param value Counter value to set
     */
    void set_counter_value(uint64_t value);
    
    /**
     * @brief Check if timer is running
     * @return true if timer is active
     */
    bool is_running() const;

private:
    HWTimerID timer_id_;
    uint32_t resolution_us_;
    void* timer_handle_;  // Platform-specific timer handle
    TimerCallback callback_;
    void* user_data_;
    bool initialized_;
    bool interrupt_mode_;
    
    bool configure_hardware_timer();
    static void IRAM_ATTR hardware_timer_isr(void* arg);
};

/**
 * @brief Profiling utilities for performance measurement
 */
class Profiler {
public:
    /**
     * @brief Start profiling measurement
     * @return Profile handle for end measurement
     */
    static uint32_t start_profile();
    
    /**
     * @brief End profiling measurement
     * @param handle Profile handle from start_profile
     * @return Profile result with timing information
     */
    static ProfileResult end_profile(uint32_t handle);
    
    /**
     * @brief Measure execution time of a function
     * @param func Function to measure
     * @return Profile result
     */
    template<typename Func>
    static ProfileResult measure_function(Func&& func);
    
    /**
     * @brief Get current high-resolution timestamp
     * @return Timestamp in microseconds
     */
    static uint64_t get_timestamp_us();
    
    /**
     * @brief Get current CPU cycle count
     * @return CPU cycle count
     */
    static uint32_t get_cpu_cycles();
    
    /**
     * @brief Convert CPU cycles to microseconds
     * @param cycles CPU cycles
     * @return Microseconds
     */
    static float cycles_to_us(uint32_t cycles);
    
    /**
     * @brief Delay for precise number of microseconds
     * @param us Microseconds to delay
     */
    static void precise_delay_us(uint32_t us);
    
    /**
     * @brief Delay for precise number of nanoseconds (busy wait)
     * @param ns Nanoseconds to delay
     */
    static void precise_delay_ns(uint32_t ns);

private:
    struct ProfileHandle {
        uint64_t start_time_us;
        uint32_t start_cycles;
        bool active;
    };
    
    static constexpr size_t MAX_CONCURRENT_PROFILES = 16;
    static ProfileHandle profile_handles_[MAX_CONCURRENT_PROFILES];
    static uint32_t next_handle_id_;
    
    static uint32_t allocate_handle();
    static void free_handle(uint32_t handle);
};

/**
 * @brief Watchdog timer management
 */
namespace watchdog {
    /**
     * @brief Watchdog timer types
     */
    enum class WatchdogType : uint8_t {
        TASK_WATCHDOG = 0,    // Task watchdog timer
        INTERRUPT_WATCHDOG    // Interrupt watchdog timer
    };
    
    /**
     * @brief Initialize watchdog timer
     * @param type Watchdog type
     * @param timeout_ms Timeout in milliseconds
     * @return TimerStatus::SUCCESS on success
     */
    TimerStatus init_watchdog(WatchdogType type, uint32_t timeout_ms);
    
    /**
     * @brief Feed (reset) watchdog timer
     * @param type Watchdog type
     * @return TimerStatus::SUCCESS on success
     */
    TimerStatus feed_watchdog(WatchdogType type);
    
    /**
     * @brief Add current task to watchdog monitoring
     * @return TimerStatus::SUCCESS on success
     */
    TimerStatus add_current_task();
    
    /**
     * @brief Remove current task from watchdog monitoring
     * @return TimerStatus::SUCCESS on success
     */
    TimerStatus remove_current_task();
    
    /**
     * @brief Enable/disable watchdog timer
     * @param type Watchdog type
     * @param enable Enable or disable
     * @return TimerStatus::SUCCESS on success
     */
    TimerStatus enable_watchdog(WatchdogType type, bool enable);
    
    /**
     * @brief Get watchdog timeout value
     * @param type Watchdog type
     * @return Timeout in milliseconds
     */
    uint32_t get_timeout(WatchdogType type);
}

/**
 * @brief Delay and timing utilities
 */
namespace timing_utils {
    /**
     * @brief Accurate microsecond delay
     * @param us Microseconds to delay
     */
    void delay_us(uint32_t us);
    
    /**
     * @brief Accurate millisecond delay
     * @param ms Milliseconds to delay
     */
    void delay_ms(uint32_t ms);
    
    /**
     * @brief Non-blocking delay check
     * @param start_time_us Start time in microseconds
     * @param delay_us Delay duration in microseconds
     * @return true if delay period has elapsed
     */
    bool delay_elapsed(uint64_t start_time_us, uint32_t delay_us);
    
    /**
     * @brief Create timestamp string for logging
     * @param timestamp_us Timestamp in microseconds
     * @param buffer Buffer to write string to
     * @param buffer_size Size of buffer
     * @return Length of string written
     */
    size_t format_timestamp(uint64_t timestamp_us, char* buffer, size_t buffer_size);
    
    /**
     * @brief Get time difference in microseconds
     * @param start_time_us Start time
     * @param end_time_us End time
     * @return Time difference in microseconds
     */
    uint64_t time_diff_us(uint64_t start_time_us, uint64_t end_time_us);
    
    /**
     * @brief Calculate timer frequency from period
     * @param period_us Period in microseconds
     * @return Frequency in Hz
     */
    float period_to_frequency(uint64_t period_us);
    
    /**
     * @brief Calculate period from frequency
     * @param frequency_hz Frequency in Hz
     * @return Period in microseconds
     */
    uint64_t frequency_to_period(float frequency_hz);
}

// Template implementation
template<typename Func>
ProfileResult Profiler::measure_function(Func&& func) {
    uint32_t handle = start_profile();
    func();
    return end_profile(handle);
}

} // namespace esp32
} // namespace platform
} // namespace cmx