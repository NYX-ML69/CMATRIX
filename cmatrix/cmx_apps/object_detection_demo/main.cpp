#include "cmx_config.hpp"
#include "camera_input.hpp"
#include "bounding_box_renderer.hpp"
#include <cmx_model_loader.h>
#include <cmx_runtime_api.h>
#include <cstring>
#include <cstdlib>

// ============================================================================
// Global Variables (Static Memory Allocation)
// ============================================================================

static uint8_t g_model_buffer[CMX_MODEL_BUFFER_SIZE] __attribute__((aligned(32)));
static uint8_t g_inference_buffer[CMX_INFERENCE_BUFFER_SIZE] __attribute__((aligned(32)));
static uint8_t g_camera_buffer[CMX_CAMERA_BUFFER_SIZE] __attribute__((aligned(32)));
static CMXDetection g_detections[CMX_MAX_DETECTIONS];
static CMXModelInfo g_model_info;

// CMatrix runtime handles
static cmx_model_handle_t g_model_handle = nullptr;
static cmx_runtime_handle_t g_runtime_handle = nullptr;

// Performance tracking
#if CMX_ENABLE_PROFILING
static uint32_t g_frame_count = 0;
static uint32_t g_total_inference_time = 0;
static uint32_t g_last_fps_update = 0;
#endif

// ============================================================================
// Forward Declarations
// ============================================================================

static CMXErrorCode initialize_cmatrix_runtime();
static CMXErrorCode load_object_detection_model();
static CMXErrorCode process_frame();
static CMXErrorCode run_inference(const uint8_t* input_data, float* output_data);
static uint16_t postprocess_detections(const float* output_data, CMXDetection* detections);
static void cleanup_resources();
static uint32_t get_system_time_ms();

// ============================================================================
// Main Application Entry Point
// ============================================================================

int main() {
    CMXErrorCode result;
    
    CMX_LOG_INFO("Starting CMX Object Detection Demo");
    CMX_LOG_INFO("Input: %dx%dx%d, Classes: %d", CMX_INPUT_WIDTH, CMX_INPUT_HEIGHT, 
                 CMX_INPUT_CHANNELS, CMX_NUM_CLASSES);
    
    // Initialize camera module
    result = static_cast<CMXErrorCode>(camera_init());
    if (result != CMX_SUCCESS) {
        CMX_LOG_ERROR("Failed to initialize camera: %d", result);
        return result;
    }
    CMX_LOG_INFO("Camera initialized successfully");
    
    // Initialize display module
    result = static_cast<CMXErrorCode>(display_init());
    if (result != CMX_SUCCESS) {
        CMX_LOG_ERROR("Failed to initialize display: %d", result);
        camera_deinit();
        return result;
    }
    CMX_LOG_INFO("Display initialized successfully");
    
    // Initialize CMatrix runtime
    result = initialize_cmatrix_runtime();
    if (result != CMX_SUCCESS) {
        CMX_LOG_ERROR("Failed to initialize CMatrix runtime: %d", result);
        cleanup_resources();
        return result;
    }
    CMX_LOG_INFO("CMatrix runtime initialized");
    
    // Load object detection model
    result = load_object_detection_model();
    if (result != CMX_SUCCESS) {
        CMX_LOG_ERROR("Failed to load model: %d", result);
        cleanup_resources();
        return result;
    }
    CMX_LOG_INFO("Model loaded successfully");
    
    // Main processing loop
    CMX_LOG_INFO("Starting main processing loop (Target FPS: %d)", CMX_TARGET_FPS);
    uint32_t last_frame_time = get_system_time_ms();
    
    while (1) {
        uint32_t frame_start = get_system_time_ms();
        
        // Process current frame
        result = process_frame();
        if (result != CMX_SUCCESS) {
            CMX_LOG_WARN("Frame processing failed: %d", result);
            continue;
        }
        
        // Frame rate control
        uint32_t frame_time = get_system_time_ms() - frame_start;
        if (frame_time < CMX_FRAME_TIME_MS) {
            // Simple delay - replace with proper RTOS delay if available
            uint32_t delay_time = CMX_FRAME_TIME_MS - frame_time;
            for (volatile uint32_t i = 0; i < delay_time * 1000; i++);
        }
        
        #if CMX_ENABLE_PROFILING
        g_frame_count++;
        if (frame_start - g_last_fps_update >= 1000) {
            float fps = g_frame_count * 1000.0f / (frame_start - g_last_fps_update);
            float avg_inference = g_total_inference_time / (float)g_frame_count;
            CMX_LOG_INFO("FPS: %.1f, Avg Inference: %.1fms", fps, avg_inference);
            g_frame_count = 0;
            g_total_inference_time = 0;
            g_last_fps_update = frame_start;
        }
        #endif
    }
    
    // Cleanup (unreachable in embedded loop)
    cleanup_resources();
    return CMX_SUCCESS;
}

// ============================================================================
// CMatrix Runtime Initialization
// ============================================================================

static CMXErrorCode initialize_cmatrix_runtime() {
    // Initialize CMatrix runtime configuration
    cmx_runtime_config_t config = {};
    config.inference_buffer = g_inference_buffer;
    config.inference_buffer_size = CMX_INFERENCE_BUFFER_SIZE;
    config.enable_profiling = CMX_ENABLE_PROFILING;
    config.max_threads = 1;  // Single-threaded for microcontrollers
    
    // Create runtime instance
    cmx_status_t status = cmx_runtime_create(&config, &g_runtime_handle);
    if (status != CMX_STATUS_SUCCESS) {
        CMX_LOG_ERROR("Failed to create CMatrix runtime: %d", status);
        return CMX_ERROR_INIT;
    }
    
    return CMX_SUCCESS;
}

// ============================================================================
// Model Loading
// ============================================================================

static CMXErrorCode load_object_detection_model() {
    // In a real implementation, this would load from flash/SD card
    // For now, assume model is embedded in firmware
    extern const uint8_t object_detection_model_data[];
    extern const uint32_t object_detection_model_size;
    
    // Copy model to aligned buffer
    if (object_detection_model_size > CMX_MODEL_BUFFER_SIZE) {
        CMX_LOG_ERROR("Model size (%u) exceeds buffer size (%u)", 
                      object_detection_model_size, CMX_MODEL_BUFFER_SIZE);
        return CMX_ERROR_MODEL_LOAD;
    }
    
    memcpy(g_model_buffer, object_detection_model_data, object_detection_model_size);
    
    // Load model into CMatrix runtime
    cmx_model_config_t model_config = {};
    model_config.model_data = g_model_buffer;
    model_config.model_size = object_detection_model_size;
    model_config.runtime_handle = g_runtime_handle;
    
    cmx_status_t status = cmx_model_load(&model_config, &g_model_handle);
    if (status != CMX_STATUS_SUCCESS) {
        CMX_LOG_ERROR("Failed to load model: %d", status);
        return CMX_ERROR_MODEL_LOAD;
    }
    
    // Get model information
    cmx_model_get_info(g_model_handle, &g_model_info.input_size, 
                       &g_model_info.output_size, &g_model_info.input_width,
                       &g_model_info.input_height, &g_model_info.input_channels);
    
    CMX_LOG_INFO("Model info - Input: %ux%ux%u, Output size: %u",
                 g_model_info.input_width, g_model_info.input_height,
                 g_model_info.input_channels, g_model_info.output_size);
    
    return CMX_SUCCESS;
}

// ============================================================================
// Frame Processing Pipeline
// ============================================================================

static CMXErrorCode process_frame() {
    // Capture frame from camera
    uint8_t* frame_data = get_camera_frame();
    if (!frame_data) {
        return CMX_ERROR_CAMERA;
    }
    
    // Run inference
    float* output_data = reinterpret_cast<float*>(g_inference_buffer);
    CMXErrorCode result = run_inference(frame_data, output_data);
    if (result != CMX_SUCCESS) {
        return result;
    }
    
    // Post-process detections
    uint16_t num_detections = postprocess_detections(output_data, g_detections);
    
    // Clear display and render frame
    display_clear();
    
    // Display camera frame (if display supports it)
    display_draw_image(0, 0, CMX_INPUT_WIDTH, CMX_INPUT_HEIGHT, frame_data);
    
    // Render bounding boxes
    for (uint16_t i = 0; i < num_detections; i++) {
        const CMXDetection& det = g_detections[i];
        
        // Convert normalized coordinates to pixel coordinates
        uint16_t x = static_cast<uint16_t>(det.x * CMX_DISPLAY_WIDTH);
        uint16_t y = static_cast<uint16_t>(det.y * CMX_DISPLAY_HEIGHT);
        uint16_t w = static_cast<uint16_t>(det.w * CMX_DISPLAY_WIDTH);
        uint16_t h = static_cast<uint16_t>(det.h * CMX_DISPLAY_HEIGHT);
        
        // Draw bounding box
        draw_bounding_box(x, y, w, h, det.class_name, det.confidence);
    }
    
    // Update display
    display_update();
    
    return CMX_SUCCESS;
}

// ============================================================================
// Inference Execution
// ============================================================================

static CMXErrorCode run_inference(const uint8_t* input_data, float* output_data) {
    #if CMX_ENABLE_PROFILING
    uint32_t start_time = get_system_time_ms();
    #endif
    
    // Prepare input tensor
    cmx_tensor_t input_tensor = {};
    input_tensor.data = const_cast<uint8_t*>(input_data);
    input_tensor.size = CMX_INPUT_SIZE;
    input_tensor.dtype = CMX_DTYPE_UINT8;
    input_tensor.shape[0] = 1;  // batch size
    input_tensor.shape[1] = CMX_INPUT_HEIGHT;
    input_tensor.shape[2] = CMX_INPUT_WIDTH;
    input_tensor.shape[3] = CMX_INPUT_CHANNELS;
    
    // Prepare output tensor
    cmx_tensor_t output_tensor = {};
    output_tensor.data = reinterpret_cast<uint8_t*>(output_data);
    output_tensor.size = g_model_info.output_size * sizeof(float);
    output_tensor.dtype = CMX_DTYPE_FLOAT32;
    
    // Run inference
    cmx_status_t status = cmx_model_predict(g_model_handle, &input_tensor, &output_tensor);
    if (status != CMX_STATUS_SUCCESS) {
        CMX_LOG_ERROR("Inference failed: %d", status);
        return CMX_ERROR_INFERENCE;
    }
    
    #if CMX_ENABLE_PROFILING
    uint32_t inference_time = get_system_time_ms() - start_time;
    g_total_inference_time += inference_time;
    CMX_LOG_DEBUG("Inference time: %ums", inference_time);
    #endif
    
    return CMX_SUCCESS;
}

// ============================================================================
// Post-processing
// ============================================================================

static uint16_t postprocess_detections(const float* output_data, CMXDetection* detections) {
    uint16_t detection_count = 0;
    
    // Assume output format: [batch, num_detections, 6] where 6 = [x, y, w, h, confidence, class_id]
    const uint16_t detection_size = 6;
    const uint16_t max_raw_detections = g_model_info.output_size / detection_size;
    
    for (uint16_t i = 0; i < max_raw_detections && detection_count < CMX_MAX_DETECTIONS; i++) {
        const float* detection = &output_data[i * detection_size];
        
        float confidence = detection[4];
        if (confidence < CMX_CONFIDENCE_THRESHOLD) {
            continue;
        }
        
        uint16_t class_id = static_cast<uint16_t>(detection[5]);
        if (class_id >= CMX_NUM_CLASSES) {
            continue;
        }
        
        // Extract bounding box (assuming normalized coordinates)
        CMXDetection& det = detections[detection_count];
        det.x = CMX_CLAMP(detection[0], 0.0f, 1.0f);
        det.y = CMX_CLAMP(detection[1], 0.0f, 1.0f);
        det.w = CMX_CLAMP(detection[2], 0.0f, 1.0f - det.x);
        det.h = CMX_CLAMP(detection[3], 0.0f, 1.0f - det.y);
        det.confidence = confidence;
        det.class_id = class_id;
        det.class_name = CMX_CLASS_LABELS[class_id];
        
        detection_count++;
    }
    
    CMX_LOG_DEBUG("Found %u detections", detection_count);
    return detection_count;
}

// ============================================================================
// Utility Functions
// ============================================================================

static void cleanup_resources() {
    if (g_model_handle) {
        cmx_model_unload(g_model_handle);
        g_model_handle = nullptr;
    }
    
    if (g_runtime_handle) {
        cmx_runtime_destroy(g_runtime_handle);
        g_runtime_handle = nullptr;
    }
    
    camera_deinit();
    display_deinit();
    
    CMX_LOG_INFO("Resources cleaned up");
}

static uint32_t get_system_time_ms() {
    // Platform-specific implementation
    #ifdef STM32
        return HAL_GetTick();
    #elif defined(ESP32)
        return esp_timer_get_time() / 1000;
    #else
        // Fallback - implement based on your platform
        static uint32_t counter = 0;
        return counter++;
    #endif
}