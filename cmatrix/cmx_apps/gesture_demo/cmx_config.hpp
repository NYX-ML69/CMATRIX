#ifndef CMX_CONFIG_HPP
#define CMX_CONFIG_HPP

#include <cstdint>

// =============================================================================
// CMatrix Runtime Configuration
// =============================================================================

namespace GestureDemo {

// Model Input Configuration
constexpr uint32_t INPUT_FEATURES = 6;      // 3x accel + 3x gyro
constexpr uint32_t TIME_STEPS = 32;         // Sliding window size
constexpr uint32_t INPUT_SIZE = INPUT_FEATURES * TIME_STEPS;
constexpr uint32_t OUTPUT_CLASSES = 5;      // Number of gesture classes

// Model Input Tensor Shape: [1, TIME_STEPS, INPUT_FEATURES]
constexpr uint32_t TENSOR_BATCH_SIZE = 1;
constexpr uint32_t TENSOR_HEIGHT = TIME_STEPS;
constexpr uint32_t TENSOR_WIDTH = INPUT_FEATURES;

// Gesture Class Labels
enum class GestureClass : uint8_t {
    LEFT = 0,
    RIGHT = 1,
    UP = 2,
    DOWN = 3,
    CIRCLE = 4,
    UNKNOWN = 255
};

// Gesture Labels for Display/Debug
constexpr const char* GESTURE_LABELS[] = {
    "LEFT",
    "RIGHT", 
    "UP",
    "DOWN",
    "CIRCLE"
};

// Inference Configuration
constexpr float CONFIDENCE_THRESHOLD = 0.7f;    // Minimum confidence for gesture detection
constexpr float NOISE_THRESHOLD = 0.05f;        // IMU noise filtering threshold
constexpr uint32_t INFERENCE_INTERVAL_MS = 100; // Run inference every 100ms
constexpr uint32_t SAMPLE_RATE_HZ = 50;         // IMU sampling rate

// Data Processing
constexpr float ACCEL_SCALE = 16.0f;            // ±16g range
constexpr float GYRO_SCALE = 2000.0f;           // ±2000 dps range
constexpr bool NORMALIZE_INPUT = true;          // Apply input normalization

// =============================================================================
// Hardware Pin Definitions
// =============================================================================

// I2C Configuration for IMU
constexpr uint8_t I2C_SDA_PIN = 21;             // GPIO21 (ESP32) / PB9 (STM32)
constexpr uint8_t I2C_SCL_PIN = 22;             // GPIO22 (ESP32) / PB8 (STM32)
constexpr uint32_t I2C_FREQUENCY = 400000;      // 400kHz I2C clock

// IMU Configuration
constexpr uint8_t IMU_I2C_ADDR = 0x68;          // MPU6050 default address
constexpr uint8_t IMU_INT_PIN = 19;             // Data ready interrupt pin

// Output Hardware Pins
constexpr uint8_t LED_LEFT_PIN = 12;            // Left gesture LED
constexpr uint8_t LED_RIGHT_PIN = 13;           // Right gesture LED  
constexpr uint8_t LED_UP_PIN = 14;              // Up gesture LED
constexpr uint8_t LED_DOWN_PIN = 15;            // Down gesture LED
constexpr uint8_t LED_CIRCLE_PIN = 16;          // Circle gesture LED

constexpr uint8_t BUZZER_PIN = 17;              // Buzzer/Speaker output
constexpr uint8_t STATUS_LED_PIN = 2;           // System status LED

// UART Configuration
constexpr uint32_t UART_BAUDRATE = 115200;     // Serial debug output
constexpr uint8_t UART_TX_PIN = 1;             // UART TX pin
constexpr uint8_t UART_RX_PIN = 3;             // UART RX pin

// =============================================================================
// Memory Configuration
// =============================================================================

// Buffer Sizes
constexpr uint32_t IMU_BUFFER_SIZE = INPUT_SIZE * 2;        // Double buffer for IMU data
constexpr uint32_t MODEL_WORKSPACE_SIZE = 8192;             // CMatrix workspace memory
constexpr uint32_t INFERENCE_BUFFER_SIZE = OUTPUT_CLASSES * sizeof(float);

// Stack Sizes (FreeRTOS tasks)
constexpr uint32_t MAIN_TASK_STACK_SIZE = 4096;
constexpr uint32_t IMU_TASK_STACK_SIZE = 2048;
constexpr uint32_t OUTPUT_TASK_STACK_SIZE = 1024;

// =============================================================================
// CMatrix Runtime Types
// =============================================================================

// Data types compatible with CMatrix
using cmx_float_t = float;
using cmx_int_t = int32_t;
using cmx_uint_t = uint32_t;
using cmx_tensor_shape_t = uint32_t[4];  // [N, H, W, C] format

// CMatrix tensor configuration
struct CmxTensorConfig {
    cmx_tensor_shape_t input_shape = {TENSOR_BATCH_SIZE, TENSOR_HEIGHT, TENSOR_WIDTH, 1};
    cmx_tensor_shape_t output_shape = {TENSOR_BATCH_SIZE, OUTPUT_CLASSES, 1, 1};
    size_t input_size_bytes = INPUT_SIZE * sizeof(cmx_float_t);
    size_t output_size_bytes = OUTPUT_CLASSES * sizeof(cmx_float_t);
};

// =============================================================================
// Debug Configuration
// =============================================================================

#define ENABLE_DEBUG_OUTPUT 1
#define ENABLE_PERFORMANCE_TIMING 1
#define ENABLE_IMU_RAW_OUTPUT 0
#define ENABLE_INFERENCE_LOGGING 1

// Debug print macros
#if ENABLE_DEBUG_OUTPUT
    #define DEBUG_PRINT(fmt, ...) printf("[DEBUG] " fmt "\n", ##__VA_ARGS__)
    #define INFO_PRINT(fmt, ...) printf("[INFO] " fmt "\n", ##__VA_ARGS__)
    #define ERROR_PRINT(fmt, ...) printf("[ERROR] " fmt "\n", ##__VA_ARGS__)
#else
    #define DEBUG_PRINT(fmt, ...)
    #define INFO_PRINT(fmt, ...)
    #define ERROR_PRINT(fmt, ...)
#endif

// Performance timing
#if ENABLE_PERFORMANCE_TIMING
    #define PERF_START() uint32_t perf_start = millis()
    #define PERF_END(msg) DEBUG_PRINT(msg " took %lu ms", millis() - perf_start)
#else
    #define PERF_START()
    #define PERF_END(msg)
#endif

} // namespace GestureDemo

#endif // CMX_CONFIG_HPP