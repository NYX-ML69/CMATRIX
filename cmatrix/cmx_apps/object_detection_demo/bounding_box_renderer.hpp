#ifndef BOUNDING_BOX_RENDERER_HPP
#define BOUNDING_BOX_RENDERER_HPP

#include <cstdint>
#include <cstddef>

// Forward declarations
struct DetectionResult;

namespace BoundingBoxRenderer {

/**
 * @brief Detection result structure for object detection output
 */
struct DetectionResult {
    float x;           // Normalized x coordinate (0.0 - 1.0)
    float y;           // Normalized y coordinate (0.0 - 1.0) 
    float width;       // Normalized width (0.0 - 1.0)
    float height;      // Normalized height (0.0 - 1.0)
    uint8_t class_id;  // Object class ID
    float confidence;  // Detection confidence (0.0 - 1.0)
};

/**
 * @brief Initialize the bounding box renderer
 * @param width Display width in pixels
 * @param height Display height in pixels
 * @return true if initialization successful, false otherwise
 */
bool init_renderer(uint16_t width, uint16_t height);

/**
 * @brief Clear the entire display with specified color
 * @param color 16-bit RGB565 color value
 */
void clear_display(uint16_t color = 0x0000);

/**
 * @brief Draw a single pixel
 * @param x X coordinate
 * @param y Y coordinate
 * @param color 16-bit RGB565 color value
 */
void draw_pixel(uint16_t x, uint16_t y, uint16_t color);

/**
 * @brief Draw a line between two points
 * @param x0 Start X coordinate
 * @param y0 Start Y coordinate
 * @param x1 End X coordinate
 * @param y1 End Y coordinate
 * @param color 16-bit RGB565 color value
 */
void draw_line(uint16_t x0, uint16_t y0, uint16_t x1, uint16_t y1, uint16_t color);

/**
 * @brief Draw a rectangular bounding box
 * @param x Top-left X coordinate
 * @param y Top-left Y coordinate
 * @param w Width of the box
 * @param h Height of the box
 * @param color 16-bit RGB565 color value
 * @param thickness Line thickness in pixels (default: 1)
 */
void draw_box(uint16_t x, uint16_t y, uint16_t w, uint16_t h, uint16_t color, uint8_t thickness = 1);

/**
 * @brief Draw a single character
 * @param x X coordinate
 * @param y Y coordinate
 * @param c Character to draw
 * @param color Text color (RGB565)
 * @param bg_color Background color (RGB565)
 */
void draw_char(uint16_t x, uint16_t y, char c, uint16_t color, uint16_t bg_color = 0x0000);

/**
 * @brief Draw text string
 * @param x X coordinate
 * @param y Y coordinate
 * @param text Null-terminated string to draw
 * @param color Text color (RGB565)
 * @param bg_color Background color (RGB565)
 */
void draw_text(uint16_t x, uint16_t y, const char* text, uint16_t color, uint16_t bg_color = 0x0000);

/**
 * @brief Render all detection results with bounding boxes and labels
 * @param results Array of detection results
 * @param num_detections Number of detections in the array
 */
void render_detection_results(const DetectionResult* results, uint8_t num_detections);

/**
 * @brief Draw FPS counter in top-right corner
 * @param fps Frames per second value
 */
void draw_fps_counter(float fps);

/**
 * @brief Draw status message in bottom-left corner
 * @param message Status message to display
 */
void draw_status_message(const char* message);

/**
 * @brief Convert RGB components to RGB565 format
 * @param r Red component (0-255)
 * @param g Green component (0-255)
 * @param b Blue component (0-255)
 * @return 16-bit RGB565 color value
 */
uint16_t rgb_to_rgb565(uint8_t r, uint8_t g, uint8_t b);

/**
 * @brief Extract RGB components from RGB565 format
 * @param rgb565 16-bit RGB565 color value
 * @param r Pointer to store red component (can be nullptr)
 * @param g Pointer to store green component (can be nullptr)
 * @param b Pointer to store blue component (can be nullptr)
 */
void get_rgb565_components(uint16_t rgb565, uint8_t* r, uint8_t* g, uint8_t* b);

// Common color constants (RGB565 format)
namespace Colors {
    constexpr uint16_t BLACK       = 0x0000;
    constexpr uint16_t WHITE       = 0xFFFF;
    constexpr uint16_t RED         = 0xF800;
    constexpr uint16_t GREEN       = 0x07E0;
    constexpr uint16_t BLUE        = 0x001F;
    constexpr uint16_t YELLOW      = 0xFFE0;
    constexpr uint16_t MAGENTA     = 0xF81F;
    constexpr uint16_t CYAN        = 0x07FF;
    constexpr uint16_t LIGHT_GRAY  = 0x7BEF;
    constexpr uint16_t DARK_GRAY   = 0x39C7;
    constexpr uint16_t ORANGE      = 0xFD20;
    constexpr uint16_t PURPLE      = 0x780F;
}

} // namespace BoundingBoxRenderer

// Hardware abstraction layer - implement these functions for your platform
extern "C" {
    /**
     * @brief Initialize display hardware
     * @param width Display width in pixels
     * @param height Display height in pixels
     */
    void display_init(uint16_t width, uint16_t height);
    
    /**
     * @brief Set a single pixel on the display
     * @param x X coordinate
     * @param y Y coordinate  
     * @param color 16-bit RGB565 color value
     */
    void display_set_pixel(uint16_t x, uint16_t y, uint16_t color);
    
    /**
     * @brief Update a region of the display
     * @param x Start X coordinate
     * @param y Start Y coordinate
     * @param w Width of region
     * @param h Height of region
     */
    void display_update_region(uint16_t x, uint16_t y, uint16_t w, uint16_t h);
    
    /**
     * @brief Clear entire display with specified color
     * @param color 16-bit RGB565 color value
     */
    void display_clear(uint16_t color);
    
    /**
     * @brief Get pointer to display buffer (if available)
     * @return Pointer to display buffer or nullptr if not available
     */
    uint16_t* display_get_buffer();
}

#endif // BOUNDING_BOX_RENDERER_HPP