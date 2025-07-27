#pragma once

// Audio Configuration
#define SAMPLE_RATE 16000
#define FRAME_SIZE 1024
#define CHANNELS 1
#define BITS_PER_SAMPLE 16

// Model Configuration
#define WAKE_MODEL_PATH "wake_model.cmx"
#define MODEL_INPUT_SIZE 16000  // 1 second of audio at 16kHz
#define MODEL_OUTPUT_SIZE 1     // Single confidence score

// Wake Detection Parameters
#define CONFIDENCE_THRESHOLD 0.8f
#define DEBOUNCE_TIME_MS 2000   // 2 seconds between detections

// Buffer Configuration
#define AUDIO_BUFFER_SIZE 4096
#define MAX_AUDIO_FRAMES 10

// Model Tensor Shapes
#define INPUT_TENSOR_SHAPE {1, MODEL_INPUT_SIZE}
#define OUTPUT_TENSOR_SHAPE {1, MODEL_OUTPUT_SIZE}

// Application Settings
#define APP_NAME "Voice Wake Demo"
#define APP_VERSION "1.0.0"