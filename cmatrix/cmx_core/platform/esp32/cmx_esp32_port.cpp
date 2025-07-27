#include "cmx_esp32_port.hpp"

// ESP-IDF includes
#include "esp_system.h"
#include "esp_timer.h"
#include "esp_sleep.h"
#include "esp_cpu.h"
#include "esp_heap_caps.h"
#include "esp_pm.h"
#include "esp_clk_tree.h"
#include "driver/gpio.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "soc/rtc_cntl_reg.h"
#include "hal/cpu_hal.h"

#include <esp_log.h>

namespace cmx {
namespace platform {
namespace esp32 {

static const char* TAG = "CMX_ESP32_PORT";

// Static member definitions
bool ESP32Platform::initialized_ = false;
PlatformConfig ESP32Platform::config_ = DEFAULT_CONFIG;

// Platform Management Implementation
PlatformStatus ESP32Platform::init_platform(const PlatformConfig& config) {
    if (initialized_) {
        ESP_LOGW(TAG, "Platform already initialized");
        return PlatformStatus::SUCCESS;
    }
    
    config_ = config;
    
    // Configure CPU frequency
    if (config_.cpu_freq_mhz > 0) {
        esp_pm_config_esp32_t pm_config = {
            .max_freq_mhz = config_.cpu_freq_mhz,
            .min_freq_mhz = config_.cpu_freq_mhz,
            .light_sleep_enable = false
        };
        
        if (esp_pm_configure(&pm_config) != ESP_OK) {
            ESP_LOGE(TAG, "Failed to configure power management");
            return PlatformStatus::ERROR_INIT_FAILED;
        }
    }
    
    // Initialize high-resolution timer
    if (esp_timer_early_init() != ESP_OK) {
        ESP_LOGE(TAG, "Failed to initialize esp_timer");
        return PlatformStatus::ERROR_TIMER_FAILED;
    }
    
    ESP_LOGI(TAG, "Platform initialized successfully");
    ESP_LOGI(TAG, "CPU Freq: %u MHz, PSRAM: %s, Dual Core: %s", 
             config_.cpu_freq_mhz,
             config_.enable_psram ? "enabled" : "disabled",
             config_.enable_dual_core ? "enabled" : "disabled");
    
    initialized_ = true;
    return PlatformStatus::SUCCESS;
}

PlatformStatus ESP32Platform::shutdown_platform() {
    if (!initialized_) {
        return PlatformStatus::SUCCESS;
    }
    
    ESP_LOGI(TAG, "Shutting down platform");
    initialized_ = false;
    return PlatformStatus::SUCCESS;
}

bool ESP32Platform::is_initialized() {
    return initialized_;
}

const PlatformConfig& ESP32Platform::get_config() {
    return config_;
}

// Timing Implementation
namespace timing {

uint64_t get_time_ms() {
    return esp_timer_get_time() / 1000ULL;
}

uint64_t get_time_us() {
    return esp_timer_get_time();
}

void sleep_us(uint32_t us) {
    if (us < 1000) {
        esp_rom_delay_us(us);
    } else {
        vTaskDelay(pdMS_TO_TICKS(us / 1000));
    }
}

void sleep_ms(uint32_t ms) {
    vTaskDelay(pdMS_TO_TICKS(ms));
}

uint32_t get_cpu_cycles() {
    return cpu_hal_get_cycle_count();
}

float cycles_to_us(uint32_t cycles) {
    uint32_t cpu_freq_hz = esp_clk_cpu_freq();
    return (float)cycles / (float)cpu_freq_hz * 1000000.0f;
}

} // namespace timing

// System Implementation
namespace system {

size_t get_free_heap() {
    return esp_get_free_heap_size();
}

size_t get_minimum_free_heap() {
    return esp_get_minimum_free_heap_size();
}

size_t get_free_internal_ram() {
    return heap_caps_get_free_size(MALLOC_CAP_INTERNAL);
}

size_t get_free_psram() {
    return heap_caps_get_free_size(MALLOC_CAP_SPIRAM);
}

uint32_t get_cpu_freq_hz() {
    return esp_clk_cpu_freq();
}

float get_chip_temperature() {
    // ESP32 doesn't have built-in temperature sensor
    // This would need external sensor or approximation
    return -1.0f;
}

void reset_system() {
    esp_restart();
}

void feed_watchdog() {
    esp_task_wdt_reset();
}

} // namespace system

// Power Management Implementation
namespace power {

void light_sleep(uint64_t sleep_time_us) {
    esp_sleep_enable_timer_wakeup(sleep_time_us);
    esp_light_sleep_start();
}

void deep_sleep(uint64_t sleep_time_us) {
    if (sleep_time_us > 0) {
        esp_sleep_enable_timer_wakeup(sleep_time_us);
    }
    esp_deep_sleep_start();
}

bool set_cpu_frequency(uint32_t freq_mhz) {
    esp_pm_config_esp32_t pm_config = {
        .max_freq_mhz = freq_mhz,
        .min_freq_mhz = freq_mhz,
        .light_sleep_enable = false
    };
    
    return esp_pm_configure(&pm_config) == ESP_OK;
}

void enable_auto_light_sleep(bool enable) {
    esp_pm_config_esp32_t pm_config = {
        .max_freq_mhz = ESP32Platform::get_config().cpu_freq_mhz,
        .min_freq_mhz = 10, // Minimum frequency for light sleep
        .light_sleep_enable = enable
    };
    
    esp_pm_configure(&pm_config);
}

} // namespace power

// GPIO Implementation
namespace gpio {

bool set_output(uint8_t pin) {
    gpio_config_t io_conf = {
        .pin_bit_mask = (1ULL << pin),
        .mode = GPIO_MODE_OUTPUT,
        .pull_up_en = GPIO_PULLUP_DISABLE,
        .pull_down_en = GPIO_PULLDOWN_DISABLE,
        .intr_type = GPIO_INTR_DISABLE
    };
    
    return gpio_config(&io_conf) == ESP_OK;
}

bool set_input_pullup(uint8_t pin) {
    gpio_config_t io_conf = {
        .pin_bit_mask = (1ULL << pin),
        .mode = GPIO_MODE_INPUT,
        .pull_up_en = GPIO_PULLUP_ENABLE,
        .pull_down_en = GPIO_PULLDOWN_DISABLE,
        .intr_type = GPIO_INTR_DISABLE
    };
    
    return gpio_config(&io_conf) == ESP_OK;
}

void set_high(uint8_t pin) {
    gpio_set_level(static_cast<gpio_num_t>(pin), 1);
}

void set_low(uint8_t pin) {
    gpio_set_level(static_cast<gpio_num_t>(pin), 0);
}

bool read_pin(uint8_t pin) {
    return gpio_get_level(static_cast<gpio_num_t>(pin)) == 1;
}

void toggle_pin(uint8_t pin) {
    gpio_num_t gpio_pin = static_cast<gpio_num_t>(pin);
    int current_level = gpio_get_level(gpio_pin);
    gpio_set_level(gpio_pin, !current_level);
}

} // namespace gpio

} // namespace esp32
} // namespace platform
} // namespace cmx