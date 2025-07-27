#include "cmx_esp32_timer.hpp"

// ESP-IDF includes
#include "esp_timer.h"
#include "esp_task_wdt.h"
#include "esp_log.h"
#include "driver/gptimer.h"
#include "hal/cpu_hal.h"
#include "esp_cpu.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_rom_sys.h"

#include <algorithm>
#include <cstring>

namespace cmx {
namespace platform {
namespace esp32 {

static const char* TAG = "CMX_ESP32_TIMER";

// Static member definitions
bool ESP32TimerManager::initialized_ = false;

// ESP32TimerManager Implementation
TimerStatus ESP32TimerManager::initialize() {
    if (initialized_) {
        ESP_LOGW(TAG, "Timer manager already initialized");
        return TimerStatus::SUCCESS;
    }
    
    ESP_LOGI(TAG, "Initializing timer subsystem");
    
    // ESP timer is automatically initialized by ESP-IDF
    // Just verify it's working
    uint64_t current_time = esp_timer_get_time();
    if (current_time == 0) {
        ESP_LOGE(TAG, "ESP timer not functioning");
        return TimerStatus::ERROR_INIT_FAILED;
    }
    
    initialized_ = true;
    ESP_LOGI(TAG, "Timer subsystem initialized successfully");
    return TimerStatus::SUCCESS;
}

TimerStatus ESP32TimerManager::shutdown() {
    if (!initialized_) {
        return TimerStatus::SUCCESS;
    }
    
    ESP_LOGI(TAG, "Shutting down timer subsystem");
    initialized_ = false;
    return TimerStatus::SUCCESS;
}

bool ESP32TimerManager::is_initialized() {
    return initialized_;
}

uint64_t ESP32TimerManager::get_tick_us() {
    return esp_timer_get_time();
}

uint64_t ESP32TimerManager::get_tick_ms() {
    return esp_timer_get_time() / 1000ULL;
}

uint64_t ESP32TimerManager::us_to_ticks(uint64_t us) {
    return us; // ESP timer already in microseconds
}

uint64_t ESP32TimerManager::ticks_to_us(uint64_t ticks) {
    return ticks; // ESP timer already in microseconds
}

// SoftwareTimer Implementation
SoftwareTimer::SoftwareTimer(const TimerConfig& config)
    : config_(config)
    , timer_handle_(nullptr)
    , callback_(nullptr)
    , user_data_(nullptr)
    , initialized_(false)
    , stats_{0}
    , last_trigger_time_(0) {
    
    if (!ESP32TimerManager::is_initialized()) {
        ESP_LOGE(TAG, "Timer manager not initialized");
        return;
    }
    
    // Create ESP timer
    esp_timer_create_args_t timer_args = {
        .callback = esp_timer_callback,
        .arg = this,
        .dispatch_method = static_cast<esp_timer_dispatch_t>(config_.dispatch_method),
        .name = config_.name,
        .skip_unhandled_events = config_.skip_unhandled_events
    };
    
    esp_err_t err = esp_timer_create(&timer_args, &timer_handle_);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Failed to create timer: %s", esp_err_to_name(err));
        return;
    }
    
    initialized_ = true;
    ESP_LOGD(TAG, "Software timer '%s' created", config_.name);
}

SoftwareTimer::~SoftwareTimer() {
    if (initialized_ && timer_handle_) {
        if (is_running()) {
            stop();
        }
        esp_timer_delete(timer_handle_);
        ESP_LOGD(TAG, "Software timer '%s' destroyed", config_.name);
    }
}

SoftwareTimer::SoftwareTimer(SoftwareTimer&& other) noexcept
    : config_(other.config_)
    , timer_handle_(other.timer_handle_)
    , callback_(std::move(other.callback_))
    , user_data_(other.user_data_)
    , initialized_(other.initialized_)
    , stats_(other.stats_)
    , last_trigger_time_(other.last_trigger_time_) {
    
    other.timer_handle_ = nullptr;
    other.initialized_ = false;
    other.user_data_ = nullptr;
}

SoftwareTimer& SoftwareTimer::operator=(SoftwareTimer&& other) noexcept {
    if (this != &other) {
        // Clean up current resources
        if (initialized_ && timer_handle_) {
            if (is_running()) {
                stop();
            }
            esp_timer_delete(timer_handle_);
        }
        
        // Move data
        config_ = other.config_;
        timer_handle_ = other.timer_handle_;
        callback_ = std::move(other.callback_);
        user_data_ = other.user_data_;
        initialized_ = other.initialized_;
        stats_ = other.stats_;
        last_trigger_time_ = other.last_trigger_time_;
        
        // Reset other
        other.timer_handle_ = nullptr;
        other.initialized_ = false;
        other.user_data_ = nullptr;
    }
    return *this;
}

bool SoftwareTimer::is_valid() const {
    return initialized_ && timer_handle_ != nullptr;
}

void SoftwareTimer::set_callback(TimerCallback callback, void* user_data) {
    callback_ = callback;
    user_data_ = user_data;
}

TimerStatus SoftwareTimer::start(uint64_t delay_us) {
    if (!is_valid()) {
        return TimerStatus::ERROR_INVALID_TIMER;
    }
    
    if (is_running()) {
        ESP_LOGW(TAG, "Timer already running, stopping first");
        stop();
    }
    
    uint64_t period = (delay_us > 0) ? delay_us : config_.period_us;
    
    esp_err_t err;
    if (config_.type == TimerType::PERIODIC) {
        err = esp_timer_start_periodic(timer_handle_, period);
    } else {
        err = esp_timer_start_once(timer_handle_, period);
    }
    
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Failed to start timer: %s", esp_err_to_name(err));
        return TimerStatus::ERROR_INVALID_PARAM;
    }
    
    last_trigger_time_ = ESP32TimerManager::get_tick_us();
    ESP_LOGD(TAG, "Timer '%s' started with period %llu us", config_.name, period);
    return TimerStatus::SUCCESS;
}

TimerStatus SoftwareTimer::stop() {
    if (!is_valid()) {
        return TimerStatus::ERROR_INVALID_TIMER;
    }
    
    if (!is_running()) {
        return TimerStatus::SUCCESS;
    }
    
    esp_err_t err = esp_timer_stop(timer_handle_);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Failed to stop timer: %s", esp_err_to_name(err));
        return TimerStatus::ERROR_INVALID_TIMER;
    }
    
    ESP_LOGD(TAG, "Timer '%s' stopped", config_.name);
    return TimerStatus::SUCCESS;
}

TimerStatus SoftwareTimer::restart(uint64_t new_period_us) {
    if (!is_valid()) {
        return TimerStatus::ERROR_INVALID_TIMER;
    }
    
    // Update configuration
    config_.period_us = new_period_us;
    
    // Stop and restart
    TimerStatus status = stop();
    if (status != TimerStatus::SUCCESS) {
        return status;
    }
    
    return start(new_period_us);
}

bool SoftwareTimer::is_running() const {
    if (!is_valid()) {
        return false;
    }
    
    return esp_timer_is_active(timer_handle_);
}

const TimerConfig& SoftwareTimer::get_config() const {
    return config_;
}

TimerStats SoftwareTimer::get_stats() const {
    return stats_;
}

void SoftwareTimer::reset_stats() {
    stats_ = {0};
}

uint64_t SoftwareTimer::get_time_until_trigger() const {
    if (!is_running()) {
        return 0;
    }
    
    uint64_t current_time = ESP32TimerManager::get_tick_us();
    uint64_t elapsed = current_time - last_trigger_time_;
    
    if (elapsed >= config_.period_us) {
        return 0; // Should trigger soon/already triggered
    }
    
    return config_.period_us - elapsed;
}

void SoftwareTimer::handle_timer_callback() {
    uint64_t callback_start = ESP32TimerManager::get_tick_us();
    last_trigger_time_ = callback_start;
    stats_.total_triggers++;
    
    if (callback_) {
        callback_(user_data_);
    }
    
    uint64_t callback_end = ESP32TimerManager::get_tick_us();
    uint32_t callback_time = (uint32_t)(callback_end - callback_start);
    update_callback_stats(callback_time);
}

void SoftwareTimer::esp_timer_callback(void* arg) {
    SoftwareTimer* timer = static_cast<SoftwareTimer*>(arg);
    if (timer) {
        timer->handle_timer_callback();
    }
}

void SoftwareTimer::update_callback_stats(uint32_t callback_time_us) {
    if (stats_.total_triggers == 1) {
        stats_.min_callback_time_us = callback_time_us;
        stats_.max_callback_time_us = callback_time_us;
        stats_.avg_callback_time_us = callback_time_us;
    } else {
        stats_.min_callback_time_us = std::min(stats_.min_callback_time_us, callback_time_us);
        stats_.max_callback_time_us = std::max(stats_.max_callback_time_us, callback_time_us);
        
        // Update rolling average
        stats_.avg_callback_time_us = 
            (stats_.avg_callback_time_us * (stats_.total_triggers - 1) + callback_time_us) / 
            stats_.total_triggers;
    }
    
    // Check for callback overruns (callback took longer than period)
    if (config_.type == TimerType::PERIODIC && callback_time_us > config_.period_us) {
        stats_.callback_overruns++;
        ESP_LOGW(TAG, "Callback overrun detected: %u us > %llu us", 
                 callback_time_us, config_.period_us);
    }
}

// HardwareTimer Implementation
HardwareTimer::HardwareTimer(HWTimerID timer_id, uint32_t resolution_us)
    : timer_id_(timer_id)
    , resolution_us_(resolution_us)
    , timer_handle_(nullptr)
    , callback_(nullptr)
    , user_data_(nullptr)
    , initialized_(false)
    , interrupt_mode_(false) {
    
    if (timer_id >= HWTimerID::MAX_TIMERS) {
        ESP_LOGE(TAG, "Invalid timer ID: %d", static_cast<int>(timer_id));
        return;
    }
    
    if (configure_hardware_timer()) {
        initialized_ = true;
        ESP_LOGI(TAG, "Hardware timer %d initialized", static_cast<int>(timer_id_));
    }
}

HardwareTimer::~HardwareTimer() {
    if (initialized_ && timer_handle_) {
        stop();
        gptimer_del_timer(static_cast<gptimer_handle_t>(timer_handle_));
        ESP_LOGI(TAG, "Hardware timer %d destroyed", static_cast<int>(timer_id_));
    }
}

bool HardwareTimer::is_valid() const {
    return initialized_ && timer_handle_ != nullptr;
}

void HardwareTimer::set_callback(TimerCallback callback, void* user_data) {
    callback_ = callback;
    user_data_ = user_data;
}

TimerStatus HardwareTimer::start_interrupt(uint64_t period_us, bool auto_reload) {
    if (!is_valid() || !callback_) {
        return TimerStatus::ERROR_INVALID_TIMER;
    }
    
    gptimer_handle_t timer = static_cast<gptimer_handle_t>(timer_handle_);
    
    // Configure alarm
    gptimer_alarm_config_t alarm_config = {
        .alarm_count = period_us,
        .reload_count = auto_reload ? 0 : period_us,
        .flags = {
            .auto_reload_on_alarm = auto_reload
        }
    };
    
    esp_err_t err = gptimer_set_alarm_action(timer, &alarm_config);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Failed to set alarm: %s", esp_err_to_name(err));
        return TimerStatus::ERROR_INVALID_PARAM;
    }
    
    // Register callback
    gptimer_event_callbacks_t callbacks = {
        .on_alarm = hardware_timer_isr
    };
    
    err = gptimer_register_event_callbacks(timer, &callbacks, this);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Failed to register callback: %s", esp_err_to_name(err));
        return TimerStatus::ERROR_INVALID_PARAM;
    }
    
    // Enable and start timer
    err = gptimer_enable(timer);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Failed to enable timer: %s", esp_err_to_name(err));
        return TimerStatus::ERROR_INVALID_PARAM;
    }
    
    err = gptimer_start(timer);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Failed to start timer: %s", esp_err_to_name(err));
        gptimer_disable(timer);
        return TimerStatus::ERROR_INVALID_PARAM;
    }
    
    interrupt_mode_ = true;
    ESP_LOGI(TAG, "Hardware timer %d started in interrupt mode", static_cast<int>(timer_id_));
    return TimerStatus::SUCCESS;
}

TimerStatus HardwareTimer::start_polling(uint64_t period_us) {
    if (!is_valid()) {
        return TimerStatus::ERROR_INVALID_TIMER;
    }
    
    gptimer_handle_t timer = static_cast<gptimer_handle_t>(timer_handle_);
    
    // Configure alarm for polling
    gptimer_alarm_config_t alarm_config = {
        .alarm_count = period_us,
        .reload_count = 0,
        .flags = {
            .auto_reload_on_alarm = true
        }
    };
    
    esp_err_t err = gptimer_set_alarm_action(timer, &alarm_config);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Failed to set alarm: %s", esp_err_to_name(err));
        return TimerStatus::ERROR_INVALID_PARAM;
    }
    
    // Enable and start timer
    err = gptimer_enable(timer);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Failed to enable timer: %s", esp_err_to_name(err));
        return TimerStatus::ERROR_INVALID_PARAM;
    }
    
    err = gptimer_start(timer);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Failed to start timer: %s", esp_err_to_name(err));
        gptimer_disable(timer);
        return TimerStatus::ERROR_INVALID_PARAM;
    }
    
    interrupt_mode_ = false;
    ESP_LOGI(TAG, "Hardware timer %d started in polling mode", static_cast<int>(timer_id_));
    return TimerStatus::SUCCESS;
}

TimerStatus HardwareTimer::stop() {
    if (!is_valid()) {
        return TimerStatus::ERROR_INVALID_TIMER;
    }
    
    gptimer_handle_t timer = static_cast<gptimer_handle_t>(timer_handle_);
    
    esp_err_t err = gptimer_stop(timer);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Failed to stop timer: %s", esp_err_to_name(err));
        return TimerStatus::ERROR_INVALID_TIMER;
    }
    
    err = gptimer_disable(timer);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Failed to disable timer: %s", esp_err_to_name(err));
    }
    
    ESP_LOGI(TAG, "Hardware timer %d stopped", static_cast<int>(timer_id_));
    return TimerStatus::SUCCESS;
}

bool HardwareTimer::has_triggered() {
    if (!is_valid() || interrupt_mode_) {
        return false;
    }
    
    // For polling mode, check alarm flag
    // This is a simplified implementation - actual implementation would check hardware flags
    return false; // Placeholder
}

void HardwareTimer::clear_trigger() {
    // Clear hardware trigger flag - implementation depends on specific timer
}

uint64_t HardwareTimer::get_counter_value() const {
    if (!is_valid()) {
        return 0;
    }
    
    uint64_t count_value;
    gptimer_handle_t timer = static_cast<gptimer_handle_t>(timer_handle_);
    
    esp_err_t err = gptimer_get_raw_count(timer, &count_value);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Failed to get counter value: %s", esp_err_to_name(err));
        return 0;
    }
    
    return count_value;
}

void HardwareTimer::set_counter_value(uint64_t value) {
    if (!is_valid()) {
        return;
    }
    
    gptimer_handle_t timer = static_cast<gptimer_handle_t>(timer_handle_);
    esp_err_t err = gptimer_set_raw_count(timer, value);
    
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Failed to set counter value: %s", esp_err_to_name(err));
    }
}

bool HardwareTimer::is_running() const {
    // Implementation would check timer state
    return initialized_ && timer_handle_ != nullptr;
}

bool HardwareTimer::configure_hardware_timer() {
    gptimer_config_t timer_config = {
        .clk_src = GPTIMER_CLK_SRC_DEFAULT,
        .direction = GPTIMER_COUNT_UP,
        .resolution_hz = 1000000 / resolution_us_, // Convert to Hz
        .flags = {
            .intr_shared = false
        }
    };
    
    gptimer_handle_t timer_handle;
    esp_err_t err = gptimer_new_timer(&timer_config, &timer_handle);
    
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Failed to create hardware timer: %s", esp_err_to_name(err));
        return false;
    }
    
    timer_handle_ = timer_handle;
    return true;
}

bool IRAM_ATTR HardwareTimer::hardware_timer_isr(gptimer_handle_t timer, 
                                                const gptimer_alarm_event_data_t* edata, 
                                                void* user_ctx) {
    HardwareTimer* hw_timer = static_cast<HardwareTimer*>(user_ctx);
    
    if (hw_timer && hw_timer->callback_) {
        hw_timer->callback_(hw_timer->user_data_);
    }
    
    return false; // Don't yield from ISR
}

// Profiler Implementation
Profiler::ProfileHandle Profiler::profile_handles_[MAX_CONCURRENT_PROFILES] = {0};
uint32_t Profiler::next_handle_id_ = 1;

uint32_t Profiler::start_profile() {
    uint32_t handle = allocate_handle();
    if (handle == 0) {
        return 0; // No available handles
    }
    
    ProfileHandle& prof = profile_handles_[handle - 1];
    prof.start_time_us = get_timestamp_us();
    prof.start_cycles = get_cpu_cycles();
    prof.active = true;
    
    return handle;
}

ProfileResult Profiler::end_profile(uint32_t handle) {
    if (handle == 0 || handle > MAX_CONCURRENT_PROFILES) {
        return {0, 0, 0, 0, 0.0f};
    }
    
    uint64_t end_time_us = get_timestamp_us();
    uint32_t end_cycles = get_cpu_cycles();
    
    ProfileHandle& prof = profile_handles_[handle - 1];
    if (!prof.active) {
        return {0, 0, 0, 0, 0.0f};
    }
    
    ProfileResult result = {
        .start_time_us = prof.start_time_us,
        .end_time_us = end_time_us,
        .duration_us = end_time_us - prof.start_time_us,
        .cpu_cycles = end_cycles - prof.start_cycles,
        .duration_ms = (end_time_us - prof.start_time_us) / 1000.0f
    };
    
    free_handle(handle);
    return result;
}

uint64_t Profiler::get_timestamp_us() {
    return esp_timer_get_time();
}

uint32_t Profiler::get_cpu_cycles() {
    return cpu_hal_get_cycle_count();
}

float Profiler::cycles_to_us(uint32_t cycles) {
    uint32_t cpu_freq_hz = esp_clk_cpu_freq();
    return (float)cycles / (float)cpu_freq_hz * 1000000.0f;
}

void Profiler::precise_delay_us(uint32_t us) {
    if (us < 16) {
        // Use ROM delay for very short delays
        esp_rom_delay_us(us);
    } else {
        // Use esp_timer for longer delays
        uint64_t start = esp_timer_get_time();
        while ((esp_timer_get_time() - start) < us) {
            // Busy wait
        }
    }
}

void Profiler::precise_delay_ns(uint32_t ns) {
    if (ns < 1000) {
        // Very short delay - use CPU cycles
        uint32_t cpu_freq_hz = esp_clk_cpu_freq();
        uint32_t cycles = (ns * cpu_freq_hz) / 1000000000UL;
        
        uint32_t start_cycle = cpu_hal_get_cycle_count();
        while ((cpu_hal_get_cycle_count() - start_cycle) < cycles) {
            // Busy wait
        }
    } else {
        precise_delay_us(ns / 1000);
    }
}

uint32_t Profiler::allocate_handle() {
    for (size_t i = 0; i < MAX_CONCURRENT_PROFILES; i++) {
        if (!profile_handles_[i].active) {
            profile_handles_[i].active = true;
            return i + 1;
        }
    }
    return 0; // No available handles
}

void Profiler::free_handle(uint32_t handle) {
    if (handle > 0 && handle <= MAX_CONCURRENT_PROFILES) {
        profile_handles_[handle - 1].active = false;
    }
}

// Watchdog utilities implementation
namespace watchdog {

TimerStatus init_watchdog(WatchdogType type, uint32_t timeout_ms) {
    esp_err_t err = ESP_OK;
    
    switch (type) {
        case WatchdogType::TASK_WATCHDOG: {
            esp_task_wdt_config_t wdt_config = {
                .timeout_ms = timeout_ms,
                .idle_core_mask = 0,
                .trigger_panic = true
            };
            err = esp_task_wdt_init(&wdt_config);
            break;
        }
        case WatchdogType::INTERRUPT_WATCHDOG:
            // Interrupt watchdog is typically configured via menuconfig
            ESP_LOGI(TAG, "Interrupt watchdog configured via menuconfig");
            break;
    }
    
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Failed to initialize watchdog: %s", esp_err_to_name(err));
        return TimerStatus::ERROR_INIT_FAILED;
    }
    
    return TimerStatus::SUCCESS;
}

TimerStatus feed_watchdog(WatchdogType type) {
    esp_err_t err = ESP_OK;
    
    switch (type) {
        case WatchdogType::TASK_WATCHDOG:
            err = esp_task_wdt_reset();
            break;
        case WatchdogType::INTERRUPT_WATCHDOG:
            // Interrupt watchdog is fed automatically
            break;
    }
    
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Failed to feed watchdog: %s", esp_err_to_name(err));
        return TimerStatus::ERROR_INVALID_PARAM;
    }
    
    return TimerStatus::SUCCESS;
}

TimerStatus add_current_task() {
    esp_err_t err = esp_task_wdt_add(xTaskGetCurrentTaskHandle());
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Failed to add task to watchdog: %s", esp_err_to_name(err));
        return TimerStatus::ERROR_INVALID_PARAM;
    }
    return TimerStatus::SUCCESS;
}

TimerStatus remove_current_task() {
    esp_err_t err = esp_task_wdt_delete(xTaskGetCurrentTaskHandle());
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Failed to remove task from watchdog: %s", esp_err_to_name(err));
        return TimerStatus::ERROR_INVALID_PARAM;
    }
    return TimerStatus::SUCCESS;
}

TimerStatus enable_watchdog(WatchdogType type, bool enable) {
    // Implementation depends on specific watchdog type
    ESP_LOGI(TAG, "Watchdog %s %s", 
             (type == WatchdogType::TASK_WATCHDOG) ? "task" : "interrupt",
             enable ? "enabled" : "disabled");
    return TimerStatus::SUCCESS;
}

uint32_t get_timeout(WatchdogType type) {
    // Return configured timeout - implementation specific
    return 30000; // Default 30 seconds
}

} // namespace watchdog

// Timing utilities implementation
namespace timing_utils {

void delay_us(uint32_t us) {
    Profiler::precise_delay_us(us);
}

void delay_ms(uint32_t ms) {
    if (ms < 10) {
        delay_us(ms * 1000);
    } else {
        vTaskDelay(pdMS_TO_TICKS(ms));
    }
}

bool delay_elapsed(uint64_t start_time_us, uint32_t delay_us) {
    uint64_t current_time = ESP32TimerManager::get_tick_us();
    return (current_time - start_time_us) >= delay_us;
}

size_t format_timestamp(uint64_t timestamp_us, char* buffer, size_t buffer_size) {
    if (!buffer || buffer_size < 16) {
        return 0;
    }
    
    uint64_t seconds = timestamp_us / 1000000ULL;
    uint32_t microseconds = timestamp_us % 1000000ULL;
    
    return snprintf(buffer, buffer_size, "%llu.%06u", seconds, microseconds);
}

uint64_t time_diff_us(uint64_t start_time_us, uint64_t end_time_us) {
    return (end_time_us >= start_time_us) ? (end_time_us - start_time_us) : 0;
}

float period_to_frequency(uint64_t period_us) {
    if (period_us == 0) {
        return 0.0f;
    }
    return 1000000.0f / (float)period_us;
}

uint64_t frequency_to_period(float frequency_hz) {
    if (frequency_hz <= 0.0f) {
        return 0;
    }
    return (uint64_t)(1000000.0f / frequency_hz);
}

} // namespace timing_utils

} // namespace esp32
} // namespace platform
} // namespace cmx