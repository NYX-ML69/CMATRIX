#include <Arduino.h>
#include <freertos/FreeRTOS.h>
#include <freertos/task.h>
#include <freertos/queue.h>
#include <cmatrix/runtime.h>
#include <cstring>

#include "cmx_config.hpp"
#include "input_interface.hpp"
#include "output_handler.hpp"

using namespace GestureDemo;

// =============================================================================
// Global Variables
// =============================================================================

// CMatrix Runtime Objects
static cmx_runtime_t* g_runtime = nullptr;
static cmx_model_t* g_model = nullptr;
static cmx_tensor_t* g_input_tensor = nullptr;
static cmx_tensor_t* g_output_tensor = nullptr;

// Data Buffers
static float g_imu_buffer[INPUT_SIZE] __attribute__((aligned(16)));
static float g_inference_output[OUTPUT_CLASSES] __attribute__((aligned(16)));
static uint8_t g_model_workspace[MODEL_WORKSPACE_SIZE] __attribute__((aligned(16)));

// Task Handles
static TaskHandle_t g_main_task_handle = nullptr;
static TaskHandle_t g_imu_task_handle = nullptr;
static TaskHandle_t g_output_task_handle = nullptr;

// Inter-task Communication
static QueueHandle_t g_gesture_queue = nullptr;

// System State
static volatile bool g_system_initialized = false;
static volatile uint32_t g_inference_count = 0;
static volatile uint32_t g_last_inference_time = 0;

// =============================================================================
// Forward Declarations
// =============================================================================

bool initialize_cmatrix_runtime();
bool load_gesture_model();
void cleanup_cmatrix_runtime();
GestureClass run_inference();
void main_task(void* parameter);
void imu_task(void* parameter);
void output_task(void* parameter);
void system_status_blink();

// =============================================================================
// CMatrix Runtime Initialization
// =============================================================================

bool initialize_cmatrix_runtime() {
    INFO_PRINT("Initializing CMatrix runtime...");
    
    // Initialize runtime configuration
    cmx_runtime_config_t config = {
        .workspace_memory = g_model_workspace,
        .workspace_size = MODEL_WORKSPACE_SIZE,
        .enable_profiling = ENABLE_PERFORMANCE_TIMING,
        .memory_alignment = 16
    };
    
    // Create runtime instance
    cmx_status_t status = cmx_runtime_create(&config, &g_runtime);
    if (status != CMX_SUCCESS) {
        ERROR_PRINT("Failed to create CMatrix runtime: %d", status);
        return false;
    }
    
    // Create input tensor
    CmxTensorConfig tensor_config;
    status = cmx_tensor_create(g_runtime, 
                              tensor_config.input_shape, 
                              CMX_DTYPE_FLOAT32,
                              &g_input_tensor);
    if (status != CMX_SUCCESS) {
        ERROR_PRINT("Failed to create input tensor: %d", status);
        return false;
    }
    
    // Create output tensor
    status = cmx_tensor_create(g_runtime,
                              tensor_config.output_shape,
                              CMX_DTYPE_FLOAT32, 
                              &g_output_tensor);
    if (status != CMX_SUCCESS) {
        ERROR_PRINT("Failed to create output tensor: %d", status);
        return false;
    }
    
    INFO_PRINT("CMatrix runtime initialized successfully");
    return true;
}

bool load_gesture_model() {
    INFO_PRINT("Loading gesture recognition model...");
    
    // In a real implementation, model would be loaded from flash/SPIFFS
    // For this demo, we assume model is embedded in firmware
    extern const uint8_t gesture_model_data[];
    extern const size_t gesture_model_size;
    
    cmx_status_t status = cmx_model_load_from_buffer(g_runtime,
                                                    gesture_model_data,
                                                    gesture_model_size,
                                                    &g_model);
    if (status != CMX_SUCCESS) {
        ERROR_PRINT("Failed to load model: %d", status);
        return false;
    }
    
    // Verify model input/output shapes
    cmx_tensor_shape_t model_input_shape, model_output_shape;
    cmx_model_get_input_shape(g_model, 0, model_input_shape);
    cmx_model_get_output_shape(g_model, 0, model_output_shape);
    
    INFO_PRINT("Model loaded - Input: [%d,%d,%d,%d], Output: [%d,%d,%d,%d]",
               model_input_shape[0], model_input_shape[1], 
               model_input_shape[2], model_input_shape[3],
               model_output_shape[0], model_output_shape[1],
               model_output_shape[2], model_output_shape[3]);
    
    return true;
}

void cleanup_cmatrix_runtime() {
    if (g_output_tensor) {
        cmx_tensor_destroy(g_output_tensor);
        g_output_tensor = nullptr;
    }
    
    if (g_input_tensor) {
        cmx_tensor_destroy(g_input_tensor);
        g_input_tensor = nullptr;
    }
    
    if (g_model) {
        cmx_model_destroy(g_model);
        g_model = nullptr;
    }
    
    if (g_runtime) {
        cmx_runtime_destroy(g_runtime);
        g_runtime = nullptr;
    }
}

// =============================================================================
// Inference Engine
// =============================================================================

GestureClass run_inference() {
    PERF_START();
    
    // Copy IMU data to input tensor
    cmx_status_t status = cmx_tensor_set_data(g_input_tensor, 
                                             g_imu_buffer, 
                                             INPUT_SIZE * sizeof(float));
    if (status != CMX_SUCCESS) {
        ERROR_PRINT("Failed to set input tensor data: %d", status);
        return GestureClass::UNKNOWN;
    }
    
    // Run inference
    status = cmx_model_inference(g_model, &g_input_tensor, 1, &g_output_tensor, 1);
    if (status != CMX_SUCCESS) {
        ERROR_PRINT("Inference failed: %d", status);
        return GestureClass::UNKNOWN;
    }
    
    // Get output data
    status = cmx_tensor_get_data(g_output_tensor, 
                                g_inference_output, 
                                OUTPUT_CLASSES * sizeof(float));
    if (status != CMX_SUCCESS) {
        ERROR_PRINT("Failed to get output tensor data: %d", status);
        return GestureClass::UNKNOWN;
    }
    
    PERF_END("Inference");
    
    // Find class with highest confidence
    uint8_t predicted_class = 0;
    float max_confidence = g_inference_output[0];
    
    for (uint8_t i = 1; i < OUTPUT_CLASSES; i++) {
        if (g_inference_output[i] > max_confidence) {
            max_confidence = g_inference_output[i];
            predicted_class = i;
        }
    }
    
    g_inference_count++;
    g_last_inference_time = millis();
    
    #if ENABLE_INFERENCE_LOGGING
    DEBUG_PRINT("Inference #%lu: Class=%d, Confidence=%.3f", 
                g_inference_count, predicted_class, max_confidence);
    #endif
    
    // Apply confidence threshold
    if (max_confidence < CONFIDENCE_THRESHOLD) {
        return GestureClass::UNKNOWN;
    }
    
    return static_cast<GestureClass>(predicted_class);
}

// =============================================================================
// FreeRTOS Tasks
// =============================================================================

void main_task(void* parameter) {
    TickType_t last_wake_time = xTaskGetTickCount();
    const TickType_t inference_period = pdMS_TO_TICKS(INFERENCE_INTERVAL_MS);
    
    INFO_PRINT("Main inference task started");
    
    while (true) {
        if (g_system_initialized) {
            // Check if new IMU data is available
            if (is_imu_data_ready()) {
                // Read IMU window
                if (read_imu_window(g_imu_buffer)) {
                    // Run gesture inference
                    GestureClass detected_gesture = run_inference();
                    
                    // Send gesture to output handler if valid
                    if (detected_gesture != GestureClass::UNKNOWN) {
                        BaseType_t queue_result = xQueueSend(g_gesture_queue, 
                                                            &detected_gesture, 
                                                            pdMS_TO_TICKS(10));
                        if (queue_result != pdTRUE) {
                            DEBUG_PRINT("Failed to queue gesture");
                        }
                    }
                } else {
                    DEBUG_PRINT("Failed to read IMU window");
                }
            }
        }
        
        // Wait for next inference cycle
        vTaskDelayUntil(&last_wake_time, inference_period);
    }
}

void imu_task(void* parameter) {
    INFO_PRINT("IMU task started");
    
    while (true) {
        if (g_system_initialized) {
            // Update IMU data collection
            update_imu_data();
        }
        
        // Run at IMU sample rate
        vTaskDelay(pdMS_TO_TICKS(1000 / SAMPLE_RATE_HZ));
    }
}

void output_task(void* parameter) {
    GestureClass received_gesture;
    
    INFO_PRINT("Output task started");
    
    while (true) {
        // Wait for gesture from main task
        if (xQueueReceive(g_gesture_queue, &received_gesture, portMAX_DELAY) == pdTRUE) {
            // Handle the detected gesture
            handle_gesture(static_cast<int>(received_gesture));
            
            INFO_PRINT("Processed gesture: %s", 
                      GESTURE_LABELS[static_cast<uint8_t>(received_gesture)]);
        }
    }
}

// =============================================================================
// System Status and Monitoring
// =============================================================================

void system_status_blink() {
    static uint32_t last_blink = 0;
    static bool led_state = false;
    
    uint32_t current_time = millis();
    
    // Blink rate indicates system status
    uint32_t blink_interval = g_system_initialized ? 1000 : 200;  // Slow if OK, fast if initializing
    
    if (current_time - last_blink > blink_interval) {
        led_state = !led_state;
        digitalWrite(STATUS_LED_PIN, led_state);
        last_blink = current_time;
    }
}

// =============================================================================
// Arduino Setup and Main Loop
// =============================================================================

void setup() {
    // Initialize serial communication
    Serial.begin(UART_BAUDRATE);
    delay(1000);  // Wait for serial to stabilize
    
    INFO_PRINT("=== Gesture Recognition Demo Starting ===");
    INFO_PRINT("CMatrix Runtime Version: %s", cmx_get_version());
    
    // Initialize GPIO pins
    pinMode(STATUS_LED_PIN, OUTPUT);
    digitalWrite(STATUS_LED_PIN, LOW);
    
    // Initialize hardware modules
    INFO_PRINT("Initializing hardware interfaces...");
    
    if (!initialize_imu()) {
        ERROR_PRINT("Failed to initialize IMU");
        while (true) {
            system_status_blink();
            delay(100);
        }
    }
    
    if (!initialize_output_handler()) {
        ERROR_PRINT("Failed to initialize output handler");
        while (true) {
            system_status_blink();
            delay(100);
        }
    }
    
    // Initialize CMatrix runtime
    if (!initialize_cmatrix_runtime()) {
        ERROR_PRINT("Failed to initialize CMatrix runtime");
        while (true) {
            system_status_blink();
            delay(100);
        }
    }
    
    // Load gesture recognition model
    if (!load_gesture_model()) {
        ERROR_PRINT("Failed to load gesture model");
        cleanup_cmatrix_runtime();
        while (true) {
            system_status_blink();
            delay(100);
        }
    }
    
    // Create inter-task communication queue
    g_gesture_queue = xQueueCreate(10, sizeof(GestureClass));
    if (g_gesture_queue == nullptr) {
        ERROR_PRINT("Failed to create gesture queue");
        while (true) {
            system_status_blink();
            delay(100);
        }
    }
    
    // Create FreeRTOS tasks
    BaseType_t task_result;
    
    task_result = xTaskCreate(main_task, "MainTask", MAIN_TASK_STACK_SIZE, 
                             nullptr, 2, &g_main_task_handle);
    if (task_result != pdPASS) {
        ERROR_PRINT("Failed to create main task");
        while (true) delay(1000);
    }
    
    task_result = xTaskCreate(imu_task, "IMUTask", IMU_TASK_STACK_SIZE, 
                             nullptr, 3, &g_imu_task_handle);
    if (task_result != pdPASS) {
        ERROR_PRINT("Failed to create IMU task");
        while (true) delay(1000);
    }
    
    task_result = xTaskCreate(output_task, "OutputTask", OUTPUT_TASK_STACK_SIZE, 
                             nullptr, 1, &g_output_task_handle);
    if (task_result != pdPASS) {
        ERROR_PRINT("Failed to create output task");
        while (true) delay(1000);
    }
    
    // System initialization complete
    g_system_initialized = true;
    digitalWrite(STATUS_LED_PIN, HIGH);
    
    INFO_PRINT("=== System Initialized Successfully ===");
    INFO_PRINT("Ready for gesture recognition!");
    
    // Start FreeRTOS scheduler (won't return)
    vTaskStartScheduler();
}

void loop() {
    // FreeRTOS handles task scheduling
    // This loop should never be reached
    system_status_blink();
    delay(1000);
}