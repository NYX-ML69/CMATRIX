#ifndef CMX_CONFIG_HPP
#define CMX_CONFIG_HPP

#include <cstdint>

// ============================================================================
// CMX Object Detection Configuration
// ============================================================================

// Model Input Configuration
#define CMX_INPUT_WIDTH         320
#define CMX_INPUT_HEIGHT        240
#define CMX_INPUT_CHANNELS      3
#define CMX_INPUT_SIZE          (CMX_INPUT_WIDTH * CMX_INPUT_HEIGHT * CMX_INPUT_CHANNELS)

// Model Output Configuration
#define CMX_MAX_DETECTIONS      20
#define CMX_NUM_CLASSES         80  // COCO dataset classes
#define CMX_BBOX_COORDS         4   // x, y, w, h

// Detection Thresholds
#define CMX_CONFIDENCE_THRESHOLD    0.5f
#define CMX_NMS_THRESHOLD          0.4f
#define CMX_SCORE_THRESHOLD        0.3f

// Memory Configuration
#define CMX_MODEL_BUFFER_SIZE      (512 * 1024)  // 512KB for model weights
#define CMX_INFERENCE_BUFFER_SIZE  (64 * 1024)   // 64KB for inference
#define CMX_CAMERA_BUFFER_SIZE     CMX_INPUT_SIZE

// Platform-specific definitions
#ifdef STM32
    #define CMX_HEAP_SIZE          (128 * 1024)
    #define CMX_STACK_SIZE         (16 * 1024)
    #define CMX_USE_DMA            1
#elif defined(ESP32)
    #define CMX_HEAP_SIZE          (256 * 1024)
    #define CMX_STACK_SIZE         (32 * 1024)
    #define CMX_USE_PSRAM          1
#else
    #define CMX_HEAP_SIZE          (64 * 1024)
    #define CMX_STACK_SIZE         (8 * 1024)
#endif

// Display Configuration
#define CMX_DISPLAY_WIDTH       320
#define CMX_DISPLAY_HEIGHT      240
#define CMX_BOX_COLOR           0x07E0  // Green in RGB565
#define CMX_TEXT_COLOR          0xFFFF  // White in RGB565
#define CMX_BOX_THICKNESS       2

// COCO Class Labels (subset for embedded systems)
static const char* CMX_CLASS_LABELS[CMX_NUM_CLASSES] = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
};

// Performance Configuration
#define CMX_TARGET_FPS             10
#define CMX_FRAME_TIME_MS          (1000 / CMX_TARGET_FPS)
#define CMX_INFERENCE_TIMEOUT_MS   500

// Debug Configuration
#ifdef DEBUG
    #define CMX_ENABLE_PROFILING   1
    #define CMX_ENABLE_LOGGING     1
    #define CMX_LOG_LEVEL          2  // 0=ERROR, 1=WARN, 2=INFO, 3=DEBUG
#else
    #define CMX_ENABLE_PROFILING   0
    #define CMX_ENABLE_LOGGING     0
    #define CMX_LOG_LEVEL          0
#endif

// Detection Result Structure
struct CMXDetection {
    float x, y, w, h;           // Bounding box coordinates (normalized 0-1)
    float confidence;           // Detection confidence
    uint16_t class_id;          // Class identifier
    const char* class_name;     // Class name string
};

// Model Information Structure
struct CMXModelInfo {
    uint32_t input_size;
    uint32_t output_size;
    uint16_t input_width;
    uint16_t input_height;
    uint16_t input_channels;
    uint16_t num_classes;
    const char* model_name;
    const char* model_version;
};

// Error Codes
enum CMXErrorCode {
    CMX_SUCCESS = 0,
    CMX_ERROR_INIT = -1,
    CMX_ERROR_MODEL_LOAD = -2,
    CMX_ERROR_INFERENCE = -3,
    CMX_ERROR_CAMERA = -4,
    CMX_ERROR_DISPLAY = -5,
    CMX_ERROR_MEMORY = -6,
    CMX_ERROR_TIMEOUT = -7,
    CMX_ERROR_INVALID_PARAM = -8
};

// Utility Macros
#define CMX_CLAMP(x, min, max) ((x) < (min) ? (min) : ((x) > (max) ? (max) : (x)))
#define CMX_MIN(a, b) ((a) < (b) ? (a) : (b))
#define CMX_MAX(a, b) ((a) > (b) ? (a) : (b))

#if CMX_ENABLE_LOGGING
    #define CMX_LOG_ERROR(msg, ...) printf("[ERROR] " msg "\n", ##__VA_ARGS__)
    #define CMX_LOG_WARN(msg, ...)  printf("[WARN]  " msg "\n", ##__VA_ARGS__)
    #if CMX_LOG_LEVEL >= 2
        #define CMX_LOG_INFO(msg, ...)  printf("[INFO]  " msg "\n", ##__VA_ARGS__)
    #else
        #define CMX_LOG_INFO(msg, ...)
    #endif
    #if CMX_LOG_LEVEL >= 3
        #define CMX_LOG_DEBUG(msg, ...) printf("[DEBUG] " msg "\n", ##__VA_ARGS__)
    #else
        #define CMX_LOG_DEBUG(msg, ...)
    #endif
#else
    #define CMX_LOG_ERROR(msg, ...)
    #define CMX_LOG_WARN(msg, ...)
    #define CMX_LOG_INFO(msg, ...)
    #define CMX_LOG_DEBUG(msg, ...)
#endif

#endif // CMX_CONFIG_HPP