#include <cstdint>
#include <cstring>

#ifdef ESP_PLATFORM
#include "esp_system.h"
#include "esp_log.h"
#include "esp_err.h"
#include "esp_heap_caps.h"
#include "esp_psram.h"
#include "esp_timer.h"
#include "esp_cpu.h"
#include "esp_chip_info.h"
#include "driver/gpio.h"
#include "driver/uart.h"
#include "nvs_flash.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/semphr.h"
#include "soc/rtc.h"
#include "soc/soc_caps.h"
#else
#include <cstdio>
#include <thread>
#include <chrono>
#endif

namespace cmx {
namespace platform {
namespace esp32 {

#ifdef ESP_PLATFORM
static const char* TAG = "CMX_ESP32_STARTUP";

// CMX Runtime configuration
static constexpr uint32_t CMX_TASK_STACK_SIZE = 8192;
static constexpr uint32_t CMX_TASK_PRIORITY = 5;
static constexpr uint32_t CMX_UART_BAUD_RATE = 115200;
static constexpr gpio_num_t CMX_STATUS_LED_PIN = GPIO_NUM_2;

// Task handles and synchronization
static TaskHandle_t cmx_main_task_handle = nullptr;
static SemaphoreHandle_t cmx_init_complete_sem = nullptr;
static bool cmx_system_initialized = false;

// Forward declarations for CMX runtime integration
extern "C" {
    // These functions should be implemented by the CMX runtime core
    int cmx_runtime_init(void);
    int cmx_runtime_main(void);
    void cmx_runtime_cleanup(void);
}
#endif

/**
 * @brief Initialize ESP32 hardware peripherals for CMX runtime
 */
static void initialize_hardware() {
#ifdef ESP_PLATFORM
    ESP_LOGI(TAG, "Initializing ESP32 hardware for CMX runtime");

    // Configure status LED (built-in LED on most ESP32 boards)
    gpio_config_t led_config = {};
    led_config.intr_type = GPIO_INTR_DISABLE;
    led_config.mode = GPIO_MODE_OUTPUT;
    led_config.pin_bit_mask = (1ULL << CMX_STATUS_LED_PIN);
    led_config.pull_down_en = GPIO_PULLDOWN_DISABLE;
    led_config.pull_up_en = GPIO_PULLUP_DISABLE;
    
    esp_err_t ret = gpio_config(&led_config);
    if (ret == ESP_OK) {
        gpio_set_level(CMX_STATUS_LED_PIN, 0); // LED off initially
        ESP_LOGI(TAG, "Status LED configured on GPIO %d", CMX_STATUS_LED_PIN);
    } else {
        ESP_LOGW(TAG, "Failed to configure status LED: %s", esp_err_to_name(ret));
    }

    // Configure UART for debug output (usually already done by ESP-IDF)
    uart_config_t uart_config = {};
    uart_config.baud_rate = CMX_UART_BAUD_RATE;
    uart_config.data_bits = UART_DATA_8_BITS;
    uart_config.parity = UART_PARITY_DISABLE;
    uart_config.stop_bits = UART_STOP_BITS_1;
    uart_config.flow_ctrl = UART_HW_FLOWCTRL_DISABLE;
    uart_config.source_clk = UART_SCLK_DEFAULT;

    ret = uart_driver_install(UART_NUM_0, 1024, 1024, 0, NULL, 0);
    if (ret == ESP_OK) {
        uart_param_config(UART_NUM_0, &uart_config);
        ESP_LOGI(TAG, "UART configured at %d baud", CMX_UART_BAUD_RATE);
    }

    // Print chip information
    esp_chip_info_t chip_info;
    esp_chip_info(&chip_info);
    
    ESP_LOGI(TAG, "ESP32 Chip Info:");
    ESP_LOGI(TAG, "  Model: %s", (chip_info.model == CHIP_ESP32) ? "ESP32" : 
             (chip_info.model == CHIP_ESP32S2) ? "ESP32-S2" :
             (chip_info.model == CHIP_ESP32S3) ? "ESP32-S3" :
             (chip_info.model == CHIP_ESP32C3) ? "ESP32-C3" : "Unknown");
    ESP_LOGI(TAG, "  Cores: %d", chip_info.cores);
    ESP_LOGI(TAG, "  Features: %s%s%s%s",
             (chip_info.features & CHIP_FEATURE_WIFI_BGN) ? "WiFi " : "",
             (chip_info.features & CHIP_FEATURE_BT) ? "BT " : "",
             (chip_info.features & CHIP_FEATURE_BLE) ? "BLE " : "",
             (chip_info.features & CHIP_FEATURE_EMB_FLASH) ? "Embedded-Flash " : "");
    ESP_LOGI(TAG, "  Silicon revision: %d", chip_info.revision);

    // Check and initialize PSRAM if available
    if (esp_psram_is_initialized()) {
        size_t psram_size = esp_psram_get_size();
        ESP_LOGI(TAG, "PSRAM initialized: %zu bytes", psram_size);
    } else {
        ESP_LOGI(TAG, "PSRAM not available");
    }

    ESP_LOGI(TAG, "Hardware initialization complete");
#else
    printf("CMX ESP32 Startup: Hardware initialization (non-ESP32 build)\n");
#endif
}

/**
 * @brief Configure system clocks and power management
 */
static void configure_system_clocks() {
#ifdef ESP_PLATFORM
    ESP_LOGI(TAG, "Configuring system clocks for optimal performance");

    // Set CPU frequency to maximum for best ML inference performance
    rtc_cpu_freq_config_t freq_config;
    rtc_clk_cpu_freq_get_config(&freq_config);
    
    ESP_LOGI(TAG, "Current CPU frequency: %d MHz", freq_config.freq_mhz);

    // Try to set maximum CPU frequency
    if (rtc_clk_cpu_freq_mhz_to_config(240, &freq_config)) {
        rtc_clk_cpu_freq_set_config(&freq_config);
        ESP_LOGI(TAG, "CPU frequency set to 240 MHz");
    } else if (rtc_clk_cpu_freq_mhz_to_config(160, &freq_config)) {
        rtc_clk_cpu_freq_set_config(&freq_config);
        ESP_LOGI(TAG, "CPU frequency set to 160 MHz");
    } else {
        ESP_LOGW(TAG, "Could not set optimal CPU frequency");
    }

    // Configure high-resolution timer
    esp_timer_early_init();
    ESP_LOGI(TAG, "High-resolution timer initialized");

    ESP_LOGI(TAG, "System clock configuration complete");
#else
    printf("CMX ESP32 Startup: System clock configuration (non-ESP32 build)\n");
#endif
}

/**
 * @brief Initialize NVS (Non-Volatile Storage) for configuration
 */
static void initialize_nvs() {
#ifdef ESP_PLATFORM
    ESP_LOGI(TAG, "Initializing NVS");

    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        // NVS partition was truncated and needs to be erased
        ESP_LOGW(TAG, "NVS partition needs to be erased");
        ESP_ERROR_CHECK(nvs_flash_erase());
        ret = nvs_flash_init();
    }
    
    if (ret == ESP_OK) {
        ESP_LOGI(TAG, "NVS initialized successfully");
    } else {
        ESP_LOGE(TAG, "Failed to initialize NVS: %s", esp_err_to_name(ret));
    }
#else
    printf("CMX ESP32 Startup: NVS initialization (non-ESP32 build)\n");
#endif
}

/**
 * @brief Print system information and memory status
 */
static void print_system_info() {
#ifdef ESP_PLATFORM
    ESP_LOGI(TAG, "=== CMX ESP32 System Information ===");
    
    // Memory information
    size_t internal_free = heap_caps_get_free_size(MALLOC_CAP_INTERNAL);
    size_t internal_total = heap_caps_get_total_size(MALLOC_CAP_INTERNAL);
    ESP_LOGI(TAG, "Internal RAM: %zu / %zu bytes free", internal_free, internal_total);

    if (esp_psram_is_initialized()) {
        size_t psram_free = heap_caps_get_free_size(MALLOC_CAP_SPIRAM);
        size_t psram_total = heap_caps_get_total_size(MALLOC_CAP_SPIRAM);
        ESP_LOGI(TAG, "PSRAM: %zu / %zu bytes free", psram_free, psram_total);
    }

    // DMA capable memory
    size_t dma_free = heap_caps_get_free_size(MALLOC_CAP_DMA);
    ESP_LOGI(TAG, "DMA capable memory: %zu bytes free", dma_free);

    // Task information
    ESP_LOGI(TAG, "FreeRTOS tick rate: %d Hz", configTICK_RATE_HZ);
    ESP_LOGI(TAG, "Available stack for current task: %d bytes", 
             uxTaskGetStackHighWaterMark(NULL));

    ESP_LOGI(TAG, "=====================================");
#else
    printf("CMX ESP32 Startup: System information (non-ESP32 build)\n");
#endif
}

/**
 * @brief CMX main task function
 */
#ifdef ESP_PLATFORM
static void cmx_main_task(void* parameters) {
    ESP_LOGI(TAG, "CMX main task started");

    // Signal that initialization is complete
    if (cmx_init_complete_sem) {
        xSemaphoreGive(cmx_init_complete_sem);
    }

    // Turn on status LED to indicate system is running
    gpio_set_level(CMX_STATUS_LED_PIN, 1);

    // Initialize CMX runtime
    int ret = cmx_runtime_init();
    if (ret != 0) {
        ESP_LOGE(TAG, "CMX runtime initialization failed: %d", ret);
        gpio_set_level(CMX_STATUS_LED_PIN, 0); // Turn off LED on error
        vTaskDelete(NULL);
        return;
    }

    ESP_LOGI(TAG, "CMX runtime initialized successfully");

    // Run the main CMX runtime loop
    ret = cmx_runtime_main();
    if (ret != 0) {
        ESP_LOGE(TAG, "CMX runtime main loop failed: %d", ret);
    }

    // Cleanup
    cmx_runtime_cleanup();
    ESP_LOGI(TAG, "CMX runtime cleanup complete");

    // Turn off status LED
    gpio_set_level(CMX_STATUS_LED_PIN, 0);

    // Delete this task
    vTaskDelete(NULL);
}
#endif

/**
 * @brief Setup FreeRTOS tasks for CMX runtime
 */
static void setup_cmx_tasks() {
#ifdef ESP_PLATFORM
    ESP_LOGI(TAG, "Setting up CMX runtime tasks");

    // Create semaphore for initialization synchronization
    cmx_init_complete_sem = xSemaphoreCreateBinary();
    if (!cmx_init_complete_sem) {
        ESP_LOGE(TAG, "Failed to create initialization semaphore");
        return;
    }

    // Create main CMX task
    BaseType_t task_created = xTaskCreatePinnedToCore(
        cmx_main_task,           // Task function
        "cmx_main",              // Task name
        CMX_TASK_STACK_SIZE,     // Stack size
        NULL,                    // Parameters
        CMX_TASK_PRIORITY,       // Priority
        &cmx_main_task_handle,   // Task handle
        1                        // Pin to core 1 (core 0 is used by WiFi/BT)
    );

    if (task_created != pdPASS) {
        ESP_LOGE(TAG, "Failed to create CMX main task");
        vSemaphoreDelete(cmx_init_complete_sem);
        cmx_init_complete_sem = nullptr;
        return;
    }

    // Wait for task initialization to complete
    if (xSemaphoreTake(cmx_init_complete_sem, pdMS_TO_TICKS(5000)) == pdTRUE) {
        ESP_LOGI(TAG, "CMX task initialization complete");
    } else {
        ESP_LOGE(TAG, "CMX task initialization timeout");
    }

    ESP_LOGI(TAG, "CMX runtime tasks setup complete");
#else
    printf("CMX ESP32 Startup: Task setup (non-ESP32 build)\n");
#endif
}

/**
 * @brief Main ESP32 application entry point
 */
extern "C" void app_main(void) {
#ifdef ESP_PLATFORM
    ESP_LOGI(TAG, "CMX ESP32 Startup - Version 1.0");
    ESP_LOGI(TAG, "Initializing CMX runtime on ESP32");

    // Step 1: Initialize NVS
    initialize_nvs();

    // Step 2: Configure system clocks for optimal performance
    configure_system_clocks();

    // Step 3: Initialize hardware peripherals
    initialize_hardware();

    // Step 4: Print system information
    print_system_info();

    // Step 5: Setup CMX runtime tasks
    setup_cmx_tasks();

    // Mark system as initialized
    cmx_system_initialized = true;
    ESP_LOGI(TAG, "CMX ESP32 system initialization complete");

    // Main thread can now handle other system tasks or go idle
    // The CMX runtime will run in its dedicated task
    while (cmx_system_initialized) {
        // Periodic system health checks
        vTaskDelay(pdMS_TO_TICKS(10000)); // 10 second intervals
        
        // Monitor task health
        if (cmx_main_task_handle) {
            eTaskState task_state = eTaskGetState(cmx_main_task_handle);
            if (task_state == eDeleted) {
                ESP_LOGW(TAG, "CMX main task has been deleted");
                cmx_main_task_handle = nullptr;
            }
        }

        // Optional: Print memory statistics periodically
        static int stats_counter = 0;
        if (++stats_counter >= 6) { // Every minute
            ESP_LOGI(TAG, "Free heap: %zu bytes", esp_get_free_heap_size());
            stats_counter = 0;
        }
    }
#else
    // Non-ESP32 build - simplified startup for testing
    printf("CMX ESP32 Startup: Non-ESP32 build\n");
    printf("Simulating ESP32 startup sequence...\n");
    
    initialize_hardware();
    configure_system_clocks();
    initialize_nvs();
    print_system_info();
    setup_cmx_tasks();
    
    printf("CMX ESP32 startup complete (simulation)\n");
    
    // Simulate running
    while (true) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
#endif
}

/**
 * @brief Emergency shutdown function
 */
void cmx_emergency_shutdown() {
#ifdef ESP_PLATFORM
    ESP_LOGE(TAG, "Emergency shutdown initiated");
    
    // Stop main CMX task
    if (cmx_main_task_handle) {
        vTaskDelete(cmx_main_task_handle);
        cmx_main_task_handle = nullptr;
    }
    
    // Turn off status LED
    gpio_set_level(CMX_STATUS_LED_PIN, 0);
    
    // Clean up resources
    if (cmx_init_complete_sem) {
        vSemaphoreDelete(cmx_init_complete_sem);
        cmx_init_complete_sem = nullptr;
    }
    
    cmx_system_initialized = false;
    
    ESP_LOGE(TAG, "Emergency shutdown complete");
    
    // Restart system
    esp_restart();
#else
    printf("CMX ESP32 Startup: Emergency shutdown (non-ESP32 build)\n");
    exit(1);
#endif
}

/**
 * @brief Get system initialization status
 */
bool cmx_is_system_initialized() {
#ifdef ESP_PLATFORM
    return cmx_system_initialized;
#else
    return true; // Always initialized in non-ESP32 builds
#endif
}

} // namespace esp32
} // namespace platform
} // namespace cmx

// C-style interface for integration with CMX runtime
extern "C" {
    void cmx_esp32_emergency_shutdown() {
        cmx::platform::esp32::cmx_emergency_shutdown();
    }
    
    bool cmx_esp32_is_initialized() {
        return cmx::platform::esp32::cmx_is_system_initialized();
    }
}