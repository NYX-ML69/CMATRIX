#include "cmx_xtensa_timer.hpp"
#include "cmx_xtensa_port.hpp"
#include <xtensa/config/core.h>
#include <xtensa/hal.h>
#include <xtensa/tie/xt_timer.h>
#include <algorithm>
#include <cstring>

namespace cmx::platform::xtensa {

// Timer system constants
static constexpr uint8_t MAX_TIMERS = 16;
static constexpr uint32_t TIMER_RESOLUTION_US = 1; // 1 microsecond resolution

// Global timer state
static TimerHandle g_timers[MAX_TIMERS];
static TimerStats g_timer_stats = {};
static TimerConfig g_timer_config = {
    .max_timers = MAX_TIMERS,
    .enable_statistics = true,
    .enable_profiling = false,
    .timer_resolution_us = TIMER_RESOLUTION_US
};
static bool g_timer_initialized = false;
static uint32_t g_cpu_freq_hz = 0;
static uint64_t g_timer_base_cycles = 0;

void timer_init() {
    if (g_timer_initialized) {
        return;
    }
    
    // Get CPU frequency
    g_cpu_freq_hz = cmx_get_cpu_freq_hz();
    
    // Record initial cycle count
    g_timer_base_cycles = xthal_get_ccount();
    
    // Initialize timer handles
    for (uint8_t i = 0; i < MAX_TIMERS; i++) {
        g_timers[i].timer_id = i;
        g_timers[i].active = false;
        g_timers[i].periodic = false;
        g_timers[i].interval_us = 0;
        g_timers[i].next_trigger_us = 0;
        g_timers[i].callback = nullptr;
        g_timers[i].user_data = nullptr;
    }
    
    // Clear statistics
    std::memset(&g_timer_stats, 0, sizeof(g_timer_stats));
    
    g_timer_initialized = true;
    cmx_log("TIMER: Initialized timer subsystem");
}

uint64_t cmx_now_us() {
    uint32_t current_cycles = xthal_get_ccount();
    uint64_t elapsed_cycles = current_cycles - g_timer_base_cycles;
    
    // Convert cycles to microseconds
    return (elapsed_cycles * 1000000ULL) / g_cpu_freq_hz;
}

uint64_t cmx_now_ms() {
    return cmx_now_us() / 1000;
}

void cmx_delay_ms(uint32_t ms) {
    cmx_delay_us(ms * 1000);
}

void cmx_delay_us(uint32_t us) {
    if (us == 0) {
        return;
    }
    
    uint64_t start_time = cmx_now_us();
    uint64_t end_time = start_time + us;
    
    // For very short delays, use busy wait
    if (us < 100) {
        while (cmx_now_us() < end_time) {
            asm volatile ("nop");
        }
        return;
    }
    
    // For longer delays, yield CPU periodically
    while (cmx_now_us() < end_time) {
        cmx_yield();
    }
}

bool cmx_delay_elapsed(uint64_t start_time_us, uint32_t delay_us) {
    return (cmx_now_us() - start_time_us) >= delay_us;
}

static TimerHandle* find_free_timer() {
    for (uint8_t i = 0; i < MAX_TIMERS; i++) {
        if (!g_timers[i].active) {
            return &g_timers[i];
        }
    }
    return nullptr;
}

TimerHandle* cmx_timer_create_periodic(uint64_t interval_us, 
                                      TimerCallback callback, 
                                      void* user_data) {
    if (!g_timer_initialized) {
        timer_init();
    }
    
    if (!callback || interval_us == 0) {
        return nullptr;
    }
    
    TimerHandle* timer = find_free_timer();
    if (!timer) {
        return nullptr;
    }
    
    timer->active = false;
    timer->periodic = true;
    timer->interval_us = interval_us;
    timer->next_trigger_us = 0;
    timer->callback = callback;
    timer->user_data = user_data;
    
    if (g_timer_config.enable_statistics) {
        g_timer_stats.total_timers_created++;
    }
    
    return timer;
}

TimerHandle* cmx_timer_create_oneshot(uint64_t delay_us,
                                     TimerCallback callback,
                                     void* user_data) {
    if (!g_timer_initialized) {
        timer_init();
    }
    
    if (!callback || delay_us == 0) {
        return nullptr;
    }
    
    TimerHandle* timer = find_free_timer();
    if (!timer) {
        return nullptr;
    }
    
    timer->active = false;
    timer->periodic = false;
    timer->interval_us = delay_us;
    timer->next_trigger_us = 0;
    timer->callback = callback;
    timer->user_data = user_data;
    
    if (g_timer_config.enable_statistics) {
        g_timer_stats.total_timers_created++;
    }
    
    return timer;
}

bool cmx_timer_start(TimerHandle* handle) {
    if (!handle || handle->timer_id >= MAX_TIMERS) {
        return false;
    }
    
    uint64_t now = cmx_now_us();
    handle->next_trigger_us = now + handle->interval_us;
    handle->active = true;
    
    if (g_timer_config.enable_statistics) {
        g_timer_stats.active_timers++;
    }
    
    return true;
}

void cmx_timer_stop(TimerHandle* handle) {
    if (!handle || handle->timer_id >= MAX_TIMERS) {
        return;
    }
    
    if (handle->active && g_timer_config.enable_statistics) {
        g_timer_stats.active_timers--;
    }
    
    handle->active = false;
}

void cmx_timer_delete(TimerHandle* handle) {
    if (!handle || handle->timer_id >= MAX_TIMERS) {
        return;
    }
    
    cmx_timer_stop(handle);
    handle->callback = nullptr;
    handle->user_data = nullptr;
}

bool cmx_timer_set_interval(TimerHandle* handle, uint64_t new_interval_us) {
    if (!handle || handle->timer_id >= MAX_TIMERS || new_interval_us == 0) {
        return false;
    }
    
    handle->interval_us = new_interval_us;
    
    // If active, update next trigger time
    if (handle->active) {
        uint64_t now = cmx_now_us();
        handle->next_trigger_us = now + new_interval_us;
    }
    
    return true;
}

bool cmx_timer_is_active(const TimerHandle* handle) {
    if (!handle || handle->timer_id >= MAX_TIMERS) {
        return false;
    }
    
    return handle->active;
}

uint64_t cmx_timer_time_remaining(const TimerHandle* handle) {
    if (!handle || !handle->active || handle->timer_id >= MAX_TIMERS) {
        return 0;
    }
    
    uint64_t now = cmx_now_us();
    if (now >= handle->next_trigger_us) {
        return 0;
    }
    
    return handle->next_trigger_us - now;
}

void cmx_timer_process() {
    if (!g_timer_initialized) {
        return;
    }
    
    uint64_t now = cmx_now_us();
    
    for (uint8_t i = 0; i < MAX_TIMERS; i++) {
        TimerHandle* timer = &g_timers[i];
        
        if (!timer->active || !timer->callback) {
            continue;
        }
        
        if (now >= timer->next_trigger_us) {
            uint64_t callback_start = 0;
            if (g_timer_config.enable_statistics) {
                callback_start = cmx_now_us();
            }
            
            // Execute callback
            timer->callback(timer->user_data);
            
            if (g_timer_config.enable_statistics) {
                uint64_t callback_time = cmx_now_us() - callback_start;
                g_timer_stats.timer_callbacks_executed++;
                g_timer_stats.total_callback_time_us += callback_time;
                
                if (callback_time > g_timer_stats.max_callback_time_us) {
                    g_timer_stats.max_callback_time_us = static_cast<uint32_t>(callback_time);
                }
            }
            
            if (timer->periodic) {
                // Schedule next execution
                timer->next_trigger_us = now + timer->interval_us;
                
                // Check for overruns
                if (cmx_now_us() > timer->next_trigger_us && g_timer_config.enable_statistics) {
                    g_timer_stats.timer_overruns++;
                }
            } else {
                // One-shot timer, deactivate
                timer->active = false;
                if (g_timer_config.enable_statistics) {
                    g_timer_stats.active_timers--;
                }
            }
        }
    }
}

// BenchmarkTimer Implementation
BenchmarkTimer::BenchmarkTimer() : start_time_us_(0), elapsed_us_(0), running_(false) {}

void BenchmarkTimer::start() {
    start_time_us_ = cmx_now_us();
    running_ = true;
}

void BenchmarkTimer::stop() {
    if (running_) {
        elapsed_us_ += cmx_now_us() - start_time_us_;
        running_ = false;
    }
}

void BenchmarkTimer::reset() {
    elapsed_us_ = 0;
    running_ = false;
}

uint64_t BenchmarkTimer::get_elapsed_us() const {
    uint64_t total = elapsed_us_;
    if (running_) {
        total += cmx_now_us() - start_time_us_;
    }
    return total;
}

uint64_t BenchmarkTimer::get_elapsed_ns() const {
    return get_elapsed_us() * 1000;
}

double BenchmarkTimer::get_elapsed_ms() const {
    return static_cast<double>(get_elapsed_us()) / 1000.0;
}

bool BenchmarkTimer::is_running() const {
    return running_;
}

// WatchdogTimer Implementation
WatchdogTimer::WatchdogTimer(uint32_t timeout_ms, TimerCallback callback)
    : timeout_ms_(timeout_ms), callback_(callback), enabled_(false) {
    last_feed_time_us_ = cmx_now_us();
}

void WatchdogTimer::feed() {
    last_feed_time_us_ = cmx_now_us();
}

void WatchdogTimer::enable() {
    enabled_ = true;
    feed();
}

void WatchdogTimer::disable() {
    enabled_ = false;
}

bool WatchdogTimer::is_enabled() const {
    return enabled_;
}

uint32_t WatchdogTimer::get_time_since_feed() const {
    return static_cast<uint32_t>((cmx_now_us() - last_feed_time_us_) / 1000);
}

// RateLimiter Implementation
RateLimiter::RateLimiter(uint32_t max_rate_hz) 
    : last_action_time_us_(0) {
    set_rate(max_rate_hz);
}

bool RateLimiter::allow() {
    uint64_t now = cmx_now_us();
    if (now - last_action_time_us_ >= interval_us_) {
        last_action_time_us_ = now;
        return true;
    }
    return false;
}

void RateLimiter::reset() {
    last_action_time_us_ = 0;
}

void RateLimiter::set_rate(uint32_t max_rate_hz) {
    if (max_rate_hz > 0) {
        interval_us_ = 1000000ULL / max_rate_hz;
    } else {
        interval_us_ = 0;
    }
}

// ProfileTimer Implementation
ProfileTimer::ProfileTimer(const char* name) 
    : name_(name), start_time_us_(0), total_time_us_(0), 
      min_time_us_(UINT64_MAX), max_time_us_(0), 
      call_count_(0), active_(false) {}

ProfileTimer::~ProfileTimer() {
    if (active_) {
        end();
    }
}

void ProfileTimer::begin() {
    start_time_us_ = cmx_now_us();
    active_ = true;
}

void ProfileTimer::end() {
    if (active_) {
        uint64_t elapsed = cmx_now_us() - start_time_us_;
        total_time_us_ += elapsed;
        call_count_++;
        
        if (elapsed < min_time_us_) {
            min_time_us_ = elapsed;
        }
        if (elapsed > max_time_us_) {
            max_time_us_ = elapsed;
        }
        
        active_ = false;
    }
}

uint64_t ProfileTimer::get_total_time_us() const {
    return total_time_us_;
}

uint32_t ProfileTimer::get_call_count() const {
    return call_count_;
}

uint64_t ProfileTimer::get_average_time_us() const {
    return call_count_ > 0 ? total_time_us_ / call_count_ : 0;
}

uint64_t ProfileTimer::get_min_time_us() const {
    return min_time_us_ != UINT64_MAX ? min_time_us_ : 0;
}

uint64_t ProfileTimer::get_max_time_us() const {
    return max_time_us_;
}

void ProfileTimer::reset() {
    total_time_us_ = 0;
    min_time_us_ = UINT64_MAX;
    max_time_us_ = 0;
    call_count_ = 0;
}

// Timeout Implementation
Timeout::Timeout(uint64_t timeout_us) 
    : timeout_us_(timeout_us) {
    reset();
}

bool Timeout::expired() const {
    return (cmx_now_us() - start_time_us_) >= timeout_us_;
}

void Timeout::reset() {
    start_time_us_ = cmx_now_us();
}

uint64_t Timeout::remaining() const {
    uint64_t elapsed = cmx_now_us() - start_time_us_;
    return elapsed >= timeout_us_ ? 0 : (timeout_us_ - elapsed);
}

const TimerStats& cmx_timer_get_stats() {
    return g_timer_stats;
}

void cmx_timer_reset_stats() {
    CMX_CRITICAL_SECTION();
    std::memset(&g_timer_stats, 0, sizeof(g_timer_stats));
}

bool cmx_timer_configure(const TimerConfig& config) {
    if (g_timer_initialized) {
        return false; // Cannot reconfigure after initialization
    }
    
    g_timer_config = config;
    return true;
}

uint32_t cmx_timer_get_resolution() {
    return g_timer_config.timer_resolution_us;
}

double cmx_timer_calibrate(uint32_t reference_freq_hz) {
    // Simple calibration - measure our timer against reference
    uint64_t start_time = cmx_now_us();
    
    // Wait for reference period (1 second)
    cmx_delay_ms(1000);
    
    uint64_t measured_time = cmx_now_us() - start_time;
    double expected_time = 1000000.0; // 1 second in microseconds
    
    return expected_time / static_cast<double>(measured_time);
}

} // namespace cmx::platform::xtensa