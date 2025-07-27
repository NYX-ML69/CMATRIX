#include "cmx_zephyr_timer.hpp"
#include <zephyr/kernel.h>
#include <zephyr/logging/log.h>
#include <zephyr/sys/time_units.h>

LOG_MODULE_REGISTER(cmx_timer, LOG_LEVEL_INF);

namespace cmx::platform::zephyr {

static bool timer_initialized = false;
static uint32_t tick_frequency = 0;

void cmx_timer_init() {
    if (timer_initialized) {
        LOG_WRN("Timer already initialized");
        return;
    }

    tick_frequency = CONFIG_SYS_CLOCK_TICKS_PER_SEC;
    timer_initialized = true;
    
    LOG_INF("CMX timer subsystem initialized (freq: %d Hz)", tick_frequency);
}

uint64_t cmx_now_us() {
    if (!timer_initialized) {
        LOG_ERR("Timer not initialized");
        return 0;
    }
    
    return k_uptime_get() * 1000ULL;  // Convert ms to us
}

uint64_t cmx_now_ms() {
    if (!timer_initialized) {
        LOG_ERR("Timer not initialized");
        return 0;
    }
    
    return k_uptime_get();
}

void cmx_delay_ms(uint32_t ms) {
    if (!timer_initialized) {
        LOG_ERR("Timer not initialized");
        return;
    }
    
    k_msleep(ms);
}

void cmx_delay_us(uint32_t us) {
    if (!timer_initialized) {
        LOG_ERR("Timer not initialized");
        return;
    }
    
    // Use busy wait for microsecond precision
    k_busy_wait(us);
}

uint32_t cmx_get_tick_frequency() {
    return tick_frequency;
}

uint64_t cmx_ticks_to_us(uint64_t ticks) {
    if (tick_frequency == 0) {
        return 0;
    }
    
    return (ticks * 1000000ULL) / tick_frequency;
}

uint64_t cmx_us_to_ticks(uint64_t us) {
    if (tick_frequency == 0) {
        return 0;
    }
    
    return (us * tick_frequency) / 1000000ULL;
}

// CmxTimer implementation
CmxTimer::CmxTimer() : start_time_(0), stop_time_(0), running_(false) {}

void CmxTimer::start() {
    start_time_ = cmx_now_us();
    running_ = true;
}

uint64_t CmxTimer::stop() {
    if (!running_) {
        LOG_WRN("Timer not running");
        return 0;
    }
    
    stop_time_ = cmx_now_us();
    running_ = false;
    
    return stop_time_ - start_time_;
}

uint64_t CmxTimer::elapsed() const {
    if (!running_) {
        return stop_time_ - start_time_;
    }
    
    return cmx_now_us() - start_time_;
}

void CmxTimer::reset() {
    start_time_ = 0;
    stop_time_ = 0;
    running_ = false;
}

bool CmxTimer::is_running() const {
    return running_;
}

// CmxProfiler implementation
CmxProfiler::CmxProfiler(const char* name) 
    : name_(name), total_time_(0), min_time_(UINT64_MAX), max_time_(0), count_(0) {}

void CmxProfiler::start() {
    timer_.start();
}

void CmxProfiler::stop() {
    if (!timer_.is_running()) {
        LOG_WRN("Profiler %s: timer not running", name_);
        return;
    }
    
    uint64_t elapsed = timer_.stop();
    
    total_time_ += elapsed;
    count_++;
    
    if (elapsed < min_time_) {
        min_time_ = elapsed;
    }
    
    if (elapsed > max_time_) {
        max_time_ = elapsed;
    }
}

uint64_t CmxProfiler::get_average_us() const {
    if (count_ == 0) {
        return 0;
    }
    
    return total_time_ / count_;
}

uint64_t CmxProfiler::get_min_us() const {
    return (count_ > 0) ? min_time_ : 0;
}

uint64_t CmxProfiler::get_max_us() const {
    return max_time_;
}

uint32_t CmxProfiler::get_count() const {
    return count_;
}

void CmxProfiler::reset() {
    total_time_ = 0;
    min_time_ = UINT64_MAX;
    max_time_ = 0;
    count_ = 0;
    timer_.reset();
}

void CmxProfiler::print_stats() const {
    if (count_ == 0) {
        LOG_INF("Profiler %s: No measurements", name_);
        return;
    }
    
    LOG_INF("Profiler %s Statistics:", name_);
    LOG_INF("  Count: %u", count_);
    LOG_INF("  Average: %llu us", get_average_us());
    LOG_INF("  Min: %llu us", min_time_);
    LOG_INF("  Max: %llu us", max_time_);
    LOG_INF("  Total: %llu us", total_time_);
}

// CmxScopedTimer implementation
CmxScopedTimer::CmxScopedTimer(CmxProfiler* profiler) : profiler_(profiler) {
    timer_.start();
    if (profiler_) {
        profiler_->start();
    }
}

CmxScopedTimer::~CmxScopedTimer() {
    timer_.stop();
    if (profiler_) {
        profiler_->stop();
    }
}

uint64_t CmxScopedTimer::elapsed() const {
    return timer_.elapsed();
}

// CmxWatchdog implementation
CmxWatchdog::CmxWatchdog(uint32_t timeout_ms) 
    : timeout_ms_(timeout_ms), last_feed_time_(0), active_(false) {}

void CmxWatchdog::start() {
    last_feed_time_ = cmx_now_ms();
    active_ = true;
}

void CmxWatchdog::feed() {
    if (active_) {
        last_feed_time_ = cmx_now_ms();
    }
}

bool CmxWatchdog::has_timeout() {
    if (!active_) {
        return false;
    }
    
    uint64_t current_time = cmx_now_ms();
    return (current_time - last_feed_time_) > timeout_ms_;
}

void CmxWatchdog::stop() {
    active_ = false;
}

} // namespace cmx::platform::zephyr