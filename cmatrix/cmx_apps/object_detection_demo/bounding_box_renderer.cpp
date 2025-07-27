#include "bounding_box_renderer.hpp"
#include "cmx_config.hpp"
#include <cstring>
#include <algorithm>

using namespace BoundingBoxRenderer;

// Simple embedded font data (5x8 bitmap font)
static const uint8_t font_5x8[95][5] = {
    {0x00, 0x00, 0x00, 0x00, 0x00}, // Space
    {0x00, 0x00, 0x5F, 0x00, 0x00}, // !
    {0x00, 0x07, 0x00, 0x07, 0x00}, // "
    {0x14, 0x7F, 0x14, 0x7F, 0x14}, // #
    {0x24, 0x2A, 0x7F, 0x2A, 0x12}, // $
    {0x23, 0x13, 0x08, 0x64, 0x62}, // %
    {0x36, 0x49, 0x55, 0x22, 0x50}, // &
    {0x00, 0x05, 0x03, 0x00, 0x00}, // '
    {0x00, 0x1C, 0x22, 0x41, 0x00}, // (
    {0x00, 0x41, 0x22, 0x1C, 0x00}, // )
    {0x08, 0x2A, 0x1C, 0x2A, 0x08}, // *
    {0x08, 0x08, 0x3E, 0x08, 0x08}, // +
    {0x00, 0x50, 0x30, 0x00, 0x00}, // ,
    {0x08, 0x08, 0x08, 0x08, 0x08}, // -
    {0x00, 0x60, 0x60, 0x00, 0x00}, // .
    {0x20, 0x10, 0x08, 0x04, 0x02}, // /
    {0x3E, 0x51, 0x49, 0x45, 0x3E}, // 0
    {0x00, 0x42, 0x7F, 0x40, 0x00}, // 1
    {0x42, 0x61, 0x51, 0x49, 0x46}, // 2
    {0x21, 0x41, 0x45, 0x4B, 0x31}, // 3
    {0x18, 0x14, 0x12, 0x7F, 0x10}, // 4
    {0x27, 0x45, 0x45, 0x45, 0x39}, // 5
    {0x3C, 0x4A, 0x49, 0x49, 0x30}, // 6
    {0x01, 0x71, 0x09, 0x05, 0x03}, // 7
    {0x36, 0x49, 0x49, 0x49, 0x36}, // 8
    {0x06, 0x49, 0x49, 0x29, 0x1E}, // 9
    // Add more characters as needed...
};

// Color definitions (RGB565 format for 16-bit displays)
static const uint16_t COLORS[] = {
    0xF800, // Red
    0x07E0, // Green
    0x001F, // Blue
    0xFFE0, // Yellow
    0xF81F, // Magenta
    0x07FF, // Cyan
    0xFFFF, // White
    0x7BEF  // Light Gray
};

static const uint8_t NUM_COLORS = sizeof(COLORS) / sizeof(COLORS[0]);

// Display buffer and configuration
static uint16_t* display_buffer = nullptr;
static uint16_t display_width = 0;
static uint16_t display_height = 0;
static bool initialized = false;

// Low-level display interface functions (to be implemented per platform)
extern "C" {
    void display_init(uint16_t width, uint16_t height);
    void display_set_pixel(uint16_t x, uint16_t y, uint16_t color);
    void display_update_region(uint16_t x, uint16_t y, uint16_t w, uint16_t h);
    void display_clear(uint16_t color);
    uint16_t* display_get_buffer();
}

namespace BoundingBoxRenderer {

bool init_renderer(uint16_t width, uint16_t height) {
    if (initialized) {
        return true;
    }
    
    display_width = width;
    display_height = height;
    
    // Initialize display hardware
    display_init(width, height);
    display_buffer = display_get_buffer();
    
    if (!display_buffer) {
        return false;
    }
    
    // Clear display
    clear_display(0x0000); // Black background
    
    initialized = true;
    return true;
}

void clear_display(uint16_t color) {
    if (!initialized || !display_buffer) return;
    
    // Fill buffer with background color
    for (uint32_t i = 0; i < display_width * display_height; i++) {
        display_buffer[i] = color;
    }
    
    display_clear(color);
}

void draw_pixel(uint16_t x, uint16_t y, uint16_t color) {
    if (!initialized || x >= display_width || y >= display_height) return;
    
    if (display_buffer) {
        display_buffer[y * display_width + x] = color;
    }
    display_set_pixel(x, y, color);
}

void draw_line(uint16_t x0, uint16_t y0, uint16_t x1, uint16_t y1, uint16_t color) {
    if (!initialized) return;
    
    // Bresenham's line algorithm
    int16_t dx = abs(x1 - x0);
    int16_t dy = abs(y1 - y0);
    int16_t sx = (x0 < x1) ? 1 : -1;
    int16_t sy = (y0 < y1) ? 1 : -1;
    int16_t err = dx - dy;
    
    while (true) {
        draw_pixel(x0, y0, color);
        
        if (x0 == x1 && y0 == y1) break;
        
        int16_t e2 = 2 * err;
        if (e2 > -dy) {
            err -= dy;
            x0 += sx;
        }
        if (e2 < dx) {
            err += dx;
            y0 += sy;
        }
    }
}

void draw_box(uint16_t x, uint16_t y, uint16_t w, uint16_t h, uint16_t color, uint8_t thickness) {
    if (!initialized || w == 0 || h == 0) return;
    
    // Clamp to display bounds
    if (x >= display_width || y >= display_height) return;
    if (x + w > display_width) w = display_width - x;
    if (y + h > display_height) h = display_height - y;
    
    // Draw rectangle outline with specified thickness
    for (uint8_t t = 0; t < thickness && t < w/2 && t < h/2; t++) {
        // Top and bottom lines
        draw_line(x + t, y + t, x + w - 1 - t, y + t, color);
        draw_line(x + t, y + h - 1 - t, x + w - 1 - t, y + h - 1 - t, color);
        
        // Left and right lines
        draw_line(x + t, y + t, x + t, y + h - 1 - t, color);
        draw_line(x + w - 1 - t, y + t, x + w - 1 - t, y + h - 1 - t, color);
    }
}

void draw_char(uint16_t x, uint16_t y, char c, uint16_t color, uint16_t bg_color) {
    if (!initialized || c < 32 || c > 126) return;
    
    uint8_t char_index = c - 32;
    const uint8_t* char_data = font_5x8[char_index];
    
    for (uint8_t col = 0; col < 5; col++) {
        if (x + col >= display_width) break;
        
        uint8_t column = char_data[col];
        for (uint8_t row = 0; row < 8; row++) {
            if (y + row >= display_height) break;
            
            uint16_t pixel_color = (column & (1 << row)) ? color : bg_color;
            if (pixel_color != bg_color || bg_color != 0x0000) {
                draw_pixel(x + col, y + row, pixel_color);
            }
        }
    }
}

void draw_text(uint16_t x, uint16_t y, const char* text, uint16_t color, uint16_t bg_color) {
    if (!initialized || !text) return;
    
    uint16_t cursor_x = x;
    uint16_t cursor_y = y;
    
    while (*text && cursor_x < display_width) {
        if (*text == '\n') {
            cursor_x = x;
            cursor_y += 9; // 8 pixel font height + 1 pixel spacing
            if (cursor_y >= display_height) break;
        } else {
            draw_char(cursor_x, cursor_y, *text, color, bg_color);
            cursor_x += 6; // 5 pixel font width + 1 pixel spacing
        }
        text++;
    }
}

void render_detection_results(const DetectionResult* results, uint8_t num_detections) {
    if (!initialized || !results) return;
    
    for (uint8_t i = 0; i < num_detections; i++) {
        const DetectionResult& detection = results[i];
        
        // Skip low confidence detections
        if (detection.confidence < DETECTION_CONFIDENCE_THRESHOLD) {
            continue;
        }
        
        // Convert normalized coordinates to pixel coordinates
        uint16_t box_x = static_cast<uint16_t>(detection.x * display_width);
        uint16_t box_y = static_cast<uint16_t>(detection.y * display_height);
        uint16_t box_w = static_cast<uint16_t>(detection.width * display_width);
        uint16_t box_h = static_cast<uint16_t>(detection.height * display_height);
        
        // Select color based on class ID
        uint16_t box_color = COLORS[detection.class_id % NUM_COLORS];
        
        // Draw bounding box
        draw_box(box_x, box_y, box_w, box_h, box_color, 2);
        
        // Prepare label text
        char label_text[32];
        const char* class_name = (detection.class_id < NUM_DETECTION_CLASSES) ? 
                                CLASS_LABELS[detection.class_id] : "Unknown";
        
        // Format: "Class 95%"
        int confidence_percent = static_cast<int>(detection.confidence * 100);
        snprintf(label_text, sizeof(label_text), "%.8s %d%%", class_name, confidence_percent);
        
        // Draw label background
        uint16_t label_x = box_x;
        uint16_t label_y = (box_y >= 10) ? (box_y - 10) : (box_y + box_h + 2);
        uint16_t label_w = strlen(label_text) * 6;
        uint16_t label_h = 9;
        
        // Draw semi-transparent background for label
        for (uint16_t ly = 0; ly < label_h && (label_y + ly) < display_height; ly++) {
            for (uint16_t lx = 0; lx < label_w && (label_x + lx) < display_width; lx++) {
                draw_pixel(label_x + lx, label_y + ly, 0x0000); // Black background
            }
        }
        
        // Draw label text
        draw_text(label_x, label_y, label_text, box_color, 0x0000);
    }
    
    // Update display
    display_update_region(0, 0, display_width, display_height);
}

void draw_fps_counter(float fps) {
    if (!initialized) return;
    
    char fps_text[16];
    snprintf(fps_text, sizeof(fps_text), "FPS: %.1f", fps);
    
    // Draw in top-right corner
    uint16_t text_x = display_width - (strlen(fps_text) * 6);
    uint16_t text_y = 2;
    
    // Clear background
    for (uint8_t i = 0; i < 9; i++) {
        for (uint16_t j = 0; j < strlen(fps_text) * 6; j++) {
            draw_pixel(text_x + j, text_y + i, 0x0000);
        }
    }
    
    // Draw FPS text
    draw_text(text_x, text_y, fps_text, 0xFFFF, 0x0000); // White text
}

void draw_status_message(const char* message) {
    if (!initialized || !message) return;
    
    // Draw in bottom-left corner
    uint16_t text_x = 2;
    uint16_t text_y = display_height - 12;
    
    // Clear background line
    for (uint8_t i = 0; i < 10; i++) {
        for (uint16_t j = 0; j < display_width; j++) {
            draw_pixel(j, text_y + i, 0x0000);
        }
    }
    
    // Draw status message
    draw_text(text_x, text_y, message, 0x07E0, 0x0000); // Green text
}

uint16_t rgb_to_rgb565(uint8_t r, uint8_t g, uint8_t b) {
    return ((r & 0xF8) << 8) | ((g & 0xFC) << 3) | (b >> 3);
}

void get_rgb565_components(uint16_t rgb565, uint8_t* r, uint8_t* g, uint8_t* b) {
    if (r) *r = (rgb565 >> 8) & 0xF8;
    if (g) *g = (rgb565 >> 3) & 0xFC;
    if (b) *b = (rgb565 << 3) & 0xF8;
}

} // namespace BoundingBoxRenderer