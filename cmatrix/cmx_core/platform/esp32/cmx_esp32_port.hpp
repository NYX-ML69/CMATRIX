#pragma once

#include <cstdint>
#include <cstddef>

// Forward declarations to minimize ESP-IDF dependencies
struct esp_timer_handle_t;
typedef struct esp_timer_handle_t* esp_timer_handle_t;

namespace cmx {
namespace platform {
namespace esp32 {

/**
 * @brief ESP32 Platform Status Codes
 */
enum class PlatformStatus : uint8_t {
    SUCCESS = 0,
    ERROR_INIT_FAILED,
    ERROR_TIMER_FAILED,
    ERROR_DMA_FAILED,
    ERROR_MEMORY_FAILED,
    ERROR_INVALID_PARAM,
    ERROR_TIMEOUT
};

/**
 * @brief Platform configuration structure
 */
struct PlatformConfig {
    // Core configuration
    uint32_t cpu_freq_mhz;           // Target CPU frequency
    bool enable_psram;               // Enable external PSRAM if available
    bool enable_dual_core;           // Use both cores (ESP32-S3)
    
    // Memory configuration
    size_t tensor_arena_size;        // Size of tensor working memory
    size_t model_buffer_size;        // Size for model weights
    bool use_internal_sram;          // Prefer internal SRAM over PSRAM
    
    // Timer configuration
    uint32_t timer_resolution_us;    // Timer resolution in microseconds
    bool enable_profiling;           // Enable performance profiling
    
    // DMA configuration
    bool enable_dma;                 // Enable DMA transfers
    uint8_t dma_channel;            // DMA channel to use (0-7)
};

/**
 * @brief Default platform configuration for ESP32
 */
constexpr PlatformConfig DEFAULT_CONFIG = {
    .cpu_freq_mhz = 240,
    .enable_psram = true,
    .enable_dual_core = false,
    .tensor_arena_size = 64 * 1024,  // 64KB
    .model_buffer_size = 128 * 1024, // 128KB
    .use_internal_sram = false,
    .timer_resolution_us = 1,
    .enable_profiling = true,
    .enable_dma = true,
    .dma_channel = 0
};

/**
 * @brief Platform initialization and management
 */
class ESP32Platform {
public:
    /**
     * @brief Initialize ESP32 platform with given configuration
     * @param config Platform configuration
     * @return PlatformStatus::SUCCESS on success
     */
    static PlatformStatus init_platform(const PlatformConfig& config = DEFAULT_CONFIG);
    
    /**
     * @brief Shutdown platform and cleanup resources
     * @return PlatformStatus::SUCCESS on success
     */
    static PlatformStatus shutdown_platform();
    
    /**
     * @brief Check if platform is initialized
     * @return true if initialized
     */
    static bool is_initialized();
    
    /**
     * @brief Get current platform configuration
     * @return Current configuration
     */
    static const PlatformConfig& get_config();

private:
    static bool initialized_;
    static PlatformConfig config_;
};

/**
 * @brief High-resolution timing functions
 */
namespace timing {
    /**
     * @brief Get current time in milliseconds
     * @return Current time in ms since boot
     */
    uint64_t get_time_ms();
    
    /**
     * @brief Get current time in microseconds
     * @return Current time in us since boot
     */
    uint64_t get_time_us();
    
    /**
     * @brief Sleep for specified microseconds (blocking)
     * @param us Microseconds to sleep
     */
    void sleep_us(uint32_t us);
    
    /**
     * @brief Sleep for specified milliseconds (blocking)
     * @param ms Milliseconds to sleep
     */
    void sleep_ms(uint32_t ms);
    
    /**
     * @brief Get CPU cycle count (for ultra-precise timing)
     * @return Current CPU cycle count
     */
    uint32_t get_cpu_cycles();
    
    /**
     * @brief Convert CPU cycles to microseconds
     * @param cycles CPU cycles
     * @return Microseconds
     */
    float cycles_to_us(uint32_t cycles);
}

/**
 * @brief System utilities and diagnostics
 */
namespace system {
    /**
     * @brief Get free heap memory in bytes
     * @return Free heap size
     */
    size_t get_free_heap();
    
    /**
     * @brief Get minimum free heap ever recorded
     * @return Minimum free heap size
     */
    size_t get_minimum_free_heap();
    
    /**
     * @brief Get free internal RAM (fast access)
     * @return Free internal RAM size
     */
    size_t get_free_internal_ram();
    
    /**
     * @brief Get free PSRAM (if available)
     * @return Free PSRAM size
     */
    size_t get_free_psram();
    
    /**
     * @brief Get CPU frequency in Hz
     * @return CPU frequency
     */
    uint32_t get_cpu_freq_hz();
    
    /**
     * @brief Get chip temperature in Celsius (if available)
     * @return Temperature or -1 if not supported
     */
    float get_chip_temperature();
    
    /**
     * @brief Trigger watchdog reset
     */
    void reset_system();
    
    /**
     * @brief Feed watchdog timer
     */
    void feed_watchdog();
}

/**
 * @brief Power management utilities
 */
namespace power {
    /**
     * @brief Enter light sleep mode for specified duration
     * @param sleep_time_us Sleep duration in microseconds
     */
    void light_sleep(uint64_t sleep_time_us);
    
    /**
     * @brief Enter deep sleep mode (will reset on wake)
     * @param sleep_time_us Sleep duration in microseconds (0 = indefinite)
     */
    void deep_sleep(uint64_t sleep_time_us = 0);
    
    /**
     * @brief Set CPU frequency dynamically
     * @param freq_mhz Target frequency in MHz
     * @return true if successful
     */
    bool set_cpu_frequency(uint32_t freq_mhz);
    
    /**
     * @brief Enable/disable automatic light sleep
     * @param enable Enable automatic sleep
     */
    void enable_auto_light_sleep(bool enable);
}

/**
 * @brief GPIO and peripheral utilities
 */
namespace gpio {
    /**
     * @brief Configure GPIO pin as output
     * @param pin GPIO pin number
     * @return true if successful
     */
    bool set_output(uint8_t pin);
    
    /**
     * @brief Configure GPIO pin as input with pullup
     * @param pin GPIO pin number
     * @return true if successful
     */
    bool set_input_pullup(uint8_t pin);
    
    /**
     * @brief Set GPIO pin high
     * @param pin GPIO pin number
     */
    void set_high(uint8_t pin);
    
    /**
     * @brief Set GPIO pin low
     * @param pin GPIO pin number
     */
    void set_low(uint8_t pin);
    
    /**
     * @brief Read GPIO pin state
     * @param pin GPIO pin number
     * @return Pin state (true = high, false = low)
     */
    bool read_pin(uint8_t pin);
    
    /**
     * @brief Toggle GPIO pin
     * @param pin GPIO pin number
     */
    void toggle_pin(uint8_t pin);
}

} // namespace esp32
} // namespace platform
} // namespace cmx