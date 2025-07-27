#ifndef CAMERA_INPUT_HPP
#define CAMERA_INPUT_HPP

#include <stdint.h>
#include <stdbool.h>

/**
 * @file camera_input.hpp
 * @brief Camera input interface for object detection demo
 * 
 * This module provides a unified interface for embedded camera modules
 * such as OV7670, OV2640, and other common sensors used in MCU projects.
 */

namespace ObjectDetection {

/**
 * @brief Supported camera module types
 */
enum class CameraType : uint8_t {
    OV7670 = 0,     // OV7670 VGA camera
    OV2640 = 1,     // OV2640 2MP camera
    OV5640 = 2,     // OV5640 5MP camera
    GENERIC = 255   // Generic camera interface
};

/**
 * @brief Camera image formats
 */
enum class ImageFormat : uint8_t {
    GRAYSCALE_8BIT = 0,     // 8-bit grayscale
    RGB565 = 1,             // 16-bit RGB565
    RGB888 = 2,             // 24-bit RGB888
    YUV422 = 3,             // YUV422 format
    JPEG = 4                // JPEG compressed
};

/**
 * @brief Camera resolution settings
 */
enum class CameraResolution : uint8_t {
    QQVGA_160x120 = 0,      // 160x120 pixels
    QVGA_320x240 = 1,       // 320x240 pixels
    VGA_640x480 = 2,        // 640x480 pixels
    SVGA_800x600 = 3,       // 800x600 pixels
    CUSTOM = 255            // Custom resolution
};

/**
 * @brief Camera configuration structure
 */
struct CameraConfig {
    CameraType type;
    CameraResolution resolution;
    ImageFormat format;
    uint16_t custom_width;
    uint16_t custom_height;
    uint8_t frame_rate;         // Target FPS
    uint8_t brightness;         // 0-255
    uint8_t contrast;           // 0-255
    uint8_t saturation;         // 0-255 (for color formats)
    bool auto_exposure;
    bool auto_white_balance;
    uint32_t i2c_frequency;     // I2C clock frequency in Hz
};

/**
 * @brief Camera frame buffer structure
 */
struct FrameBuffer {
    uint8_t* data;              // Pointer to image data
    uint32_t size;              // Buffer size in bytes
    uint16_t width;             // Image width in pixels
    uint16_t height;            // Image height in pixels
    ImageFormat format;         // Pixel format
    uint32_t timestamp_ms;      // Frame timestamp
    bool is_valid;              // Frame validity flag
};

/**
 * @brief Camera status information
 */
struct CameraStatus {
    bool is_initialized;
    bool is_streaming;
    uint32_t frames_captured;
    uint32_t frames_dropped;
    uint32_t last_error_code;
    float current_fps;
};

/**
 * @brief Initialize the camera module
 * @param config Camera configuration parameters
 * @return true if initialization successful, false otherwise
 */
bool init_camera(const CameraConfig& config);

/**
 * @brief Deinitialize the camera module and free resources
 * @return true if deinitialization successful
 */
bool deinit_camera();

/**
 * @brief Start camera streaming
 * @return true if streaming started successfully
 */
bool start_camera_stream();

/**
 * @brief Stop camera streaming
 * @return true if streaming stopped successfully
 */
bool stop_camera_stream();

/**
 * @brief Capture a single frame (blocking call)
 * @param buffer Pointer to frame buffer structure to fill
 * @param timeout_ms Maximum time to wait for frame (0 = no timeout)
 * @return true if frame captured successfully
 */
bool capture_frame(FrameBuffer* buffer, uint32_t timeout_ms = 1000);

/**
 * @brief Get pointer to the latest captured frame (non-blocking)
 * @return Pointer to frame data, nullptr if no frame available
 */
uint8_t* get_camera_frame();

/**
 * @brief Get the latest frame buffer with metadata
 * @return Pointer to FrameBuffer structure, nullptr if no frame available
 */
const FrameBuffer* get_frame_buffer();

/**
 * @brief Check if a new frame is available
 * @return true if new frame is ready for processing
 */
bool is_frame_available();

/**
 * @brief Release the current frame buffer (mark as processed)
 * This allows the camera to reuse the buffer for next capture
 */
void release_frame();

/**
 * @brief Convert RGB565 to grayscale
 * @param rgb565_data Source RGB565 data
 * @param gray_data Destination grayscale buffer
 * @param pixel_count Number of pixels to convert
 */
void rgb565_to_grayscale(const uint16_t* rgb565_data, uint8_t* gray_data, uint32_t pixel_count);

/**
 * @brief Convert RGB888 to grayscale
 * @param rgb888_data Source RGB888 data
 * @param gray_data Destination grayscale buffer
 * @param pixel_count Number of pixels to convert
 */
void rgb888_to_grayscale(const uint8_t* rgb888_data, uint8_t* gray_data, uint32_t pixel_count);

/**
 * @brief Convert YUV422 to RGB888
 * @param yuv_data Source YUV422 data
 * @param rgb_data Destination RGB888 buffer
 * @param pixel_count Number of pixels to convert
 */
void yuv422_to_rgb888(const uint8_t* yuv_data, uint8_t* rgb_data, uint32_t pixel_count);

/**
 * @brief Resize image using nearest neighbor interpolation
 * @param src_data Source image data
 * @param src_width Source image width
 * @param src_height Source image height
 * @param dst_data Destination image buffer
 * @param dst_width Target width
 * @param dst_height Target height
 * @param bytes_per_pixel Number of bytes per pixel (1 for grayscale, 3 for RGB)
 */
void resize_image_nearest(const uint8_t* src_data, uint16_t src_width, uint16_t src_height,
                         uint8_t* dst_data, uint16_t dst_width, uint16_t dst_height,
                         uint8_t bytes_per_pixel);

/**
 * @brief Apply basic image normalization (0-255 to 0.0-1.0)
 * @param image_data Input image data (uint8_t)
 * @param normalized_data Output normalized data (float)
 * @param pixel_count Number of pixels to normalize
 */
void normalize_image_data(const uint8_t* image_data, float* normalized_data, uint32_t pixel_count);

/**
 * @brief Get current camera status
 * @param status Pointer to status structure to fill
 * @return true if status retrieved successfully
 */
bool get_camera_status(CameraStatus* status);

/**
 * @brief Set camera parameter at runtime
 * @param param Parameter type (brightness, contrast, etc.)
 * @param value New parameter value
 * @return true if parameter set successfully
 */
bool set_camera_parameter(const char* param, uint32_t value);

/**
 * @brief Reset camera module (hardware reset if available)
 * @return true if reset successful
 */
bool reset_camera();

/**
 * @brief Get resolution dimensions for predefined settings
 * @param resolution Resolution enum value
 * @param width Pointer to store width value
 * @param height Pointer to store height value
 */
void get_resolution_dimensions(CameraResolution resolution, uint16_t* width, uint16_t* height);

/**
 * @brief Calculate buffer size needed for given format and resolution
 * @param width Image width
 * @param height Image height
 * @param format Image format
 * @return Required buffer size in bytes
 */
uint32_t calculate_buffer_size(uint16_t width, uint16_t height, ImageFormat format);

/**
 * @brief Platform-specific I2C write function for camera registers
 * @param device_addr I2C device address
 * @param reg_addr Register address
 * @param data Data to write
 * @param length Number of bytes to write
 * @return true if write successful
 */
bool camera_i2c_write(uint8_t device_addr, uint16_t reg_addr, const uint8_t* data, uint16_t length);

/**
 * @brief Platform-specific I2C read function for camera registers
 * @param device_addr I2C device address
 * @param reg_addr Register address
 * @param data Buffer to store read data
 * @param length Number of bytes to read
 * @return true if read successful
 */
bool camera_i2c_read(uint8_t device_addr, uint16_t reg_addr, uint8_t* data, uint16_t length);

} // namespace ObjectDetection

#endif // CAMERA_INPUT_HPP