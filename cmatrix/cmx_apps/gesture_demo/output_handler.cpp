#include "output_handler.hpp"
#include "cmx_config.hpp"

// Platform-specific includes
#ifdef STM32
    #include "stm32f4xx_hal.h"
#elif defined(ESP32)
    #include "driver/gpio.h"
    #include "freertos/FreeRTOS.h"
    #include "freertos/task.h"
    #include "esp_log.h"
#else
    #include <iostream>
    #include <chrono>
#endif

namespace GestureDemo {

// Static configuration and state variables
static OutputConfig g_config;
static bool g_initialized = false;

// Timing variables for non-blocking operations
static uint32_t g_led_blink_start = 0;
static uint32_t g_led_blink_duration = 0;
static uint32_t g_led_blink_pin = 0;
static bool g_led_blinking = false;

static uint32_t g_buzzer_start = 0;
static uint32_t g_buzzer_duration = 0;
static uint32_t g_buzzer_pin = 0;
static bool g_buzzer_active = false;

// Confidence threshold for gesture actions
static constexpr float CONFIDENCE_THRESHOLD = 0.7f;

bool init_output_handler(const OutputConfig& config) {
    g_config = config;
    
    // Initialize GPIO pins
#ifdef STM32
    // STM32 HAL GPIO initialization
    GPIO_InitTypeDef GPIO_InitStruct = {0};
    
    // Configure LED pin
    GPIO_InitStruct.Pin = g_config.led_pin;
    GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
    GPIO_InitStruct.Pull = GPIO_NOPULL;
    GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
    HAL_GPIO_Init(GPIOB, &GPIO_InitStruct);
    
    // Configure buzzer pin
    GPIO_InitStruct.Pin = g_config.buzzer_pin;
    HAL_GPIO_Init(GPIOC, &GPIO_InitStruct);
    
#elif defined(ESP32)
    // ESP32 GPIO initialization
    gpio_config_t io_conf = {};
    io_conf.intr_type = GPIO_INTR_DISABLE;
    io_conf.mode = GPIO_MODE_OUTPUT;
    io_conf.pin_bit_mask = (1ULL << g_config.led_pin) | (1ULL << g_config.buzzer_pin);
    io_conf.pull_down_en = GPIO_PULLDOWN_DISABLE;
    io_conf.pull_up_en = GPIO_PULLUP_DISABLE;
    gpio_config(&io_conf);
    
    // Set initial states to LOW
    gpio_set_level((gpio_num_t)g_config.led_pin, 0);
    gpio_set_level((gpio_num_t)g_config.buzzer_pin, 0);
    
#else
    // Simulation mode - just print initialization
    std::cout << "Output Handler: Initialized (simulation mode)" << std::endl;
#endif
    
    g_initialized = true;
    return true;
}

void handle_gesture(uint8_t gesture_id, float confidence) {
    handle_gesture(static_cast<GestureClass>(gesture_id), confidence);
}

void handle_gesture(GestureClass gesture, float confidence) {
    if (!g_initialized) {
        return;
    }
    
    // Skip action if confidence is below threshold
    if (confidence < CONFIDENCE_THRESHOLD) {
        return;
    }
    
    // Send serial output if enabled
    if (g_config.serial_enabled) {
        send_serial_output(gesture, confidence);
    }
    
    // Handle specific gesture actions
    switch (gesture) {
        case GestureClass::LEFT:
            // Left gesture: Blink LED once
            blink_led(g_config.led_pin, g_config.blink_duration_ms);
            break;
            
        case GestureClass::RIGHT:
            // Right gesture: Double blink LED
            blink_led(g_config.led_pin, g_config.blink_duration_ms / 2);
            // Schedule second blink (simplified implementation)
            break;
            
        case GestureClass::UP:
            // Up gesture: Solid LED for longer duration
            set_led(g_config.led_pin, true);
            blink_led(g_config.led_pin, g_config.blink_duration_ms * 2);
            break;
            
        case GestureClass::DOWN:
            // Down gesture: Activate buzzer
            activate_buzzer(g_config.buzzer_pin, g_config.buzzer_duration_ms);
            break;
            
        case GestureClass::CIRCLE:
            // Circle gesture: Both LED and buzzer
            blink_led(g_config.led_pin, g_config.blink_duration_ms);
            activate_buzzer(g_config.buzzer_pin, g_config.buzzer_duration_ms / 2);
            break;
            
        case GestureClass::NONE:
        case GestureClass::UNKNOWN:
        default:
            // No action for unknown or no gesture
            break;
    }
}

void update_output_handler() {
    if (!g_initialized) {
        return;
    }
    
    uint32_t current_time = get_system_time_ms();
    
    // Handle LED blinking timeout
    if (g_led_blinking && (current_time - g_led_blink_start >= g_led_blink_duration)) {
        set_led(g_led_blink_pin, false);
        g_led_blinking = false;
    }
    
    // Handle buzzer timeout
    if (g_buzzer_active && (current_time - g_buzzer_start >= g_buzzer_duration)) {
        set_led(g_buzzer_pin, false); // Turn off buzzer
        g_buzzer_active = false;
    }
}

void set_led(uint32_t pin, bool state) {
#ifdef STM32
    HAL_GPIO_WritePin(GPIOB, pin, state ? GPIO_PIN_SET : GPIO_PIN_RESET);
#elif defined(ESP32)
    gpio_set_level((gpio_num_t)pin, state ? 1 : 0);
#else
    std::cout << "LED Pin " << pin << ": " << (state ? "ON" : "OFF") << std::endl;
#endif
}

void blink_led(uint32_t pin, uint32_t duration_ms) {
    set_led(pin, true);
    g_led_blink_start = get_system_time_ms();
    g_led_blink_duration = duration_ms;
    g_led_blink_pin = pin;
    g_led_blinking = true;
}

void activate_buzzer(uint32_t pin, uint32_t duration_ms) {
    set_led(pin, true); // Reuse LED function for simplicity
    g_buzzer_start = get_system_time_ms();
    g_buzzer_duration = duration_ms;
    g_buzzer_pin = pin;
    g_buzzer_active = true;
}

void send_serial_output(GestureClass gesture, float confidence) {
#ifdef STM32
    // STM32 UART output (simplified)
    char buffer[64];
    snprintf(buffer, sizeof(buffer), "Gesture: %s, Confidence: %.2f\r\n", 
             gesture_to_string(gesture), confidence);
    // HAL_UART_Transmit(&huart2, (uint8_t*)buffer, strlen(buffer), HAL_MAX_DELAY);
    
#elif defined(ESP32)
    ESP_LOGI("GESTURE", "Detected: %s (%.2f)", gesture_to_string(gesture), confidence);
    
#else
    std::cout << "Gesture: " << gesture_to_string(gesture) 
              << ", Confidence: " << confidence << std::endl;
#endif
}

const char* gesture_to_string(GestureClass gesture) {
    switch (gesture) {
        case GestureClass::NONE:    return "None";
        case GestureClass::LEFT:    return "Left";
        case GestureClass::RIGHT:   return "Right";
        case GestureClass::UP:      return "Up";
        case GestureClass::DOWN:    return "Down";
        case GestureClass::CIRCLE:  return "Circle";
        case GestureClass::UNKNOWN: return "Unknown";
        default:                    return "Invalid";
    }
}

uint32_t get_system_time_ms() {
#ifdef STM32
    return HAL_GetTick();
#elif defined(ESP32)
    return xTaskGetTickCount() * portTICK_PERIOD_MS;
#else
    // Simulation using C++ chrono
    auto now = std::chrono::steady_clock::now();
    auto duration = now.time_since_epoch();
    return std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();
#endif
}

} // namespace GestureDemo