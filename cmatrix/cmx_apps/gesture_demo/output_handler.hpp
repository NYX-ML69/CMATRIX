#ifndef OUTPUT_HANDLER_HPP
#define OUTPUT_HANDLER_HPP

#include <stdint.h>

/**
 * @file output_handler.hpp
 * @brief Output handler for gesture recognition demo
 * 
 * This module maps gesture predictions to physical output actions
 * such as LED control, buzzer activation, or display updates.
 */

namespace GestureDemo {

/**
 * @brief Gesture class IDs matching the trained model output
 */
enum class GestureClass : uint8_t {
    NONE = 0,       // No gesture detected
    LEFT = 1,       // Left swipe gesture
    RIGHT = 2,      // Right swipe gesture  
    UP = 3,         // Up swipe gesture
    DOWN = 4,       // Down swipe gesture
    CIRCLE = 5,     // Circular gesture
    UNKNOWN = 255   // Unknown/invalid gesture
};

/**
 * @brief Output action types
 */
enum class OutputAction : uint8_t {
    LED_BLINK,
    LED_SOLID,
    BUZZER_BEEP,
    DISPLAY_MESSAGE,
    SERIAL_OUTPUT
};

/**
 * @brief Configuration structure for output actions
 */
struct OutputConfig {
    uint32_t led_pin;
    uint32_t buzzer_pin;
    uint32_t blink_duration_ms;
    uint32_t buzzer_duration_ms;
    bool serial_enabled;
};

/**
 * @brief Initialize the output handler system
 * @param config Output configuration structure
 * @return true if initialization successful, false otherwise
 */
bool init_output_handler(const OutputConfig& config);

/**
 * @brief Handle a detected gesture by triggering appropriate output
 * @param gesture_id The detected gesture class ID (0-5)
 * @param confidence Confidence score (0.0-1.0) of the prediction
 */
void handle_gesture(uint8_t gesture_id, float confidence = 1.0f);

/**
 * @brief Handle a detected gesture using enum type
 * @param gesture The detected gesture class
 * @param confidence Confidence score (0.0-1.0) of the prediction
 */
void handle_gesture(GestureClass gesture, float confidence = 1.0f);

/**
 * @brief Update output handler (call periodically from main loop)
 * This handles timing for blinking LEDs, buzzer duration, etc.
 */
void update_output_handler();

/**
 * @brief Set LED state
 * @param pin GPIO pin number
 * @param state true for ON, false for OFF
 */
void set_led(uint32_t pin, bool state);

/**
 * @brief Blink LED for specified duration
 * @param pin GPIO pin number
 * @param duration_ms Duration in milliseconds
 */
void blink_led(uint32_t pin, uint32_t duration_ms);

/**
 * @brief Activate buzzer for specified duration
 * @param pin GPIO pin number
 * @param duration_ms Duration in milliseconds
 */
void activate_buzzer(uint32_t pin, uint32_t duration_ms);

/**
 * @brief Send gesture info via serial/UART
 * @param gesture The detected gesture
 * @param confidence Confidence score
 */
void send_serial_output(GestureClass gesture, float confidence);

/**
 * @brief Convert gesture class to string representation
 * @param gesture The gesture class
 * @return Pointer to null-terminated string
 */
const char* gesture_to_string(GestureClass gesture);

/**
 * @brief Get current system time in milliseconds
 * @return Current time in ms (platform-specific implementation)
 */
uint32_t get_system_time_ms();

} // namespace GestureDemo

#endif // OUTPUT_HANDLER_HPP