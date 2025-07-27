#include <iostream>
#include <vector>
#include <chrono>
#include <thread>
#include <cstring>
#include <signal.h>

#include "cmx_config.hpp"
#include "cmatrix/cmatrix.h"

// External function declarations
extern bool mic_init();
extern bool mic_read_frame(int16_t* buffer, size_t len);
extern void handle_wake_detected();

// Global variables
static bool g_running = true;
static cmx_context_t* g_cmx_context = nullptr;
static cmx_model_t* g_wake_model = nullptr;

// Signal handler for graceful shutdown
void signal_handler(int signal) {
    std::cout << "\nReceived signal " << signal << ", shutting down..." << std::endl;
    g_running = false;
}

// Initialize CMatrix runtime and load model
bool init_cmatrix() {
    // Initialize CMatrix runtime
    cmx_status_t status = cmx_init(&g_cmx_context);
    if (status != CMX_SUCCESS) {
        std::cerr << "Failed to initialize CMatrix runtime: " << status << std::endl;
        return false;
    }
    
    // Load wake word model
    status = cmx_load_model(g_cmx_context, WAKE_MODEL_PATH, &g_wake_model);
    if (status != CMX_SUCCESS) {
        std::cerr << "Failed to load model from " << WAKE_MODEL_PATH << ": " << status << std::endl;
        return false;
    }
    
    std::cout << "CMatrix runtime initialized successfully" << std::endl;
    std::cout << "Loaded wake word model: " << WAKE_MODEL_PATH << std::endl;
    return true;
}

// Clean up CMatrix resources
void cleanup_cmatrix() {
    if (g_wake_model) {
        cmx_unload_model(g_wake_model);
        g_wake_model = nullptr;
    }
    
    if (g_cmx_context) {
        cmx_shutdown(g_cmx_context);
        g_cmx_context = nullptr;
    }
    
    std::cout << "CMatrix resources cleaned up" << std::endl;
}

// Process audio frame and run inference
bool process_audio_frame(const int16_t* audio_data, size_t frame_size) {
    // Prepare input tensor
    std::vector<float> input_data(MODEL_INPUT_SIZE);
    
    // Convert int16 audio to float and normalize
    for (size_t i = 0; i < std::min(frame_size, (size_t)MODEL_INPUT_SIZE); i++) {
        input_data[i] = static_cast<float>(audio_data[i]) / 32768.0f;
    }
    
    // Pad with zeros if needed
    if (frame_size < MODEL_INPUT_SIZE) {
        for (size_t i = frame_size; i < MODEL_INPUT_SIZE; i++) {
            input_data[i] = 0.0f;
        }
    }
    
    // Run inference
    cmx_tensor_t input_tensor;
    input_tensor.data = input_data.data();
    input_tensor.size = MODEL_INPUT_SIZE * sizeof(float);
    input_tensor.shape = INPUT_TENSOR_SHAPE;
    input_tensor.dtype = CMX_FLOAT32;
    
    cmx_tensor_t output_tensor;
    float output_data[MODEL_OUTPUT_SIZE];
    output_tensor.data = output_data;
    output_tensor.size = MODEL_OUTPUT_SIZE * sizeof(float);
    output_tensor.shape = OUTPUT_TENSOR_SHAPE;
    output_tensor.dtype = CMX_FLOAT32;
    
    cmx_status_t status = cmx_run_inference(g_wake_model, &input_tensor, &output_tensor);
    if (status != CMX_SUCCESS) {
        std::cerr << "Inference failed: " << status << std::endl;
        return false;
    }
    
    // Check wake word confidence
    float confidence = output_data[0];
    if (confidence > CONFIDENCE_THRESHOLD) {
        std::cout << "Wake word detected! Confidence: " << confidence << std::endl;
        handle_wake_detected();
        
        // Debounce - sleep to avoid repeated triggers
        std::this_thread::sleep_for(std::chrono::milliseconds(DEBOUNCE_TIME_MS));
    }
    
    return true;
}

// Main processing loop
void run_detection_loop() {
    std::vector<int16_t> audio_buffer(FRAME_SIZE);
    
    std::cout << "Starting wake word detection..." << std::endl;
    std::cout << "Listening for wake words (Ctrl+C to exit)..." << std::endl;
    
    while (g_running) {
        // Read audio frame from microphone
        if (!mic_read_frame(audio_buffer.data(), FRAME_SIZE)) {
            std::cerr << "Failed to read audio frame" << std::endl;
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }
        
        // Process the audio frame
        if (!process_audio_frame(audio_buffer.data(), FRAME_SIZE)) {
            std::cerr << "Failed to process audio frame" << std::endl;
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }
        
        // Small delay to prevent excessive CPU usage
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

int main() {
    std::cout << APP_NAME << " v" << APP_VERSION << std::endl;
    std::cout << "Initializing voice wake word detection..." << std::endl;
    
    // Set up signal handler for graceful shutdown
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    
    // Initialize microphone
    if (!mic_init()) {
        std::cerr << "Failed to initialize microphone" << std::endl;
        return -1;
    }
    std::cout << "Microphone initialized" << std::endl;
    
    // Initialize CMatrix runtime and load model
    if (!init_cmatrix()) {
        std::cerr << "Failed to initialize CMatrix" << std::endl;
        return -1;
    }
    
    // Run the main detection loop
    try {
        run_detection_loop();
    } catch (const std::exception& e) {
        std::cerr << "Exception in detection loop: " << e.what() << std::endl;
    }
    
    // Clean up resources
    cleanup_cmatrix();
    
    std::cout << "Voice wake demo terminated" << std::endl;
    return 0;
}