#include "camera_input.hpp"
#include "cmx_config.hpp"

// ============================================================================
// Platform-specific includes
// ============================================================================

#ifdef STM32
    #include "stm32h7xx_hal.h"
    #include "stm32h7xx_hal_dcmi.h"
    #include "stm32h7xx_hal_dma.h"
#elif defined(ESP32)
    #include "esp_camera.h"
    #include "driver/gpio.h"
#endif

// ============================================================================
// Static Variables
// ============================================================================

static uint8_t g_frame_buffer[CMX_CAMERA_BUFFER_SIZE] __attribute__((aligned(32)));
static uint8_t g_temp_buffer[CMX_CAMERA_BUFFER_SIZE] __attribute__((aligned(32)));
static volatile bool g_frame_ready = false;
static volatile bool g_camera_initialized = false;

#ifdef STM32
static DCMI_HandleTypeDef g_hdcmi;
static DMA_HandleTypeDef g_hdma_dcmi;
#endif

// ============================================================================
// Forward Declarations
// ============================================================================

static int configure_camera_registers();
static int convert_rgb565_to_rgb888(const uint8_t* src, uint8_t* dst, uint32_t pixel_count);
static int convert_yuv422_to_rgb888(const uint8_t* src, uint8_t* dst, uint32_t pixel_count);
static void rgb_to_grayscale(const uint8_t* rgb, uint8_t* gray, uint32_t pixel_count);

#ifdef STM32
static void dcmi_frame_complete_callback();
static int init_stm32_camera();
#elif defined(ESP32)
static int init_esp32_camera();
#endif

// ============================================================================
// Public Interface Implementation
// ============================================================================

int camera_init() {
    CMX_LOG_INFO("Initializing camera module");
    
    if (g_camera_initialized) {
        CMX_LOG_WARN("Camera already initialized");
        return CMX_SUCCESS;
    }
    
    // Clear buffers
    memset(g_frame_buffer, 0, sizeof(g_frame_buffer));
    memset(g_temp_buffer, 0, sizeof(g_temp_buffer));
    
    int result;
    
    #ifdef STM32
        result = init_stm32_camera();
    #elif defined(ESP32)
        result = init_esp32_camera();
    #else
        // Generic initialization for other platforms
        result = configure_camera_registers();
    #endif
    
    if (result != CMX_SUCCESS) {
        CMX_LOG_ERROR("Platform-specific camera init failed: %d", result);
        return result;
    }
    
    // Configure camera registers
    result = configure_camera_registers();
    if (result != CMX_SUCCESS) {
        CMX_LOG_ERROR("Camera register configuration failed: %d", result);
        return result;
    }
    
    g_camera_initialized = true;
    CMX_LOG_INFO("Camera initialized successfully");
    
    // Capture a few dummy frames to stabilize
    for (int i = 0; i < 3; i++) {
        get_camera_frame();
        // Small delay
        for (volatile int j = 0; j < 100000; j++);
    }
    
    return CMX_SUCCESS;
}

uint8_t* get_camera_frame() {
    if (!g_camera_initialized) {
        CMX_LOG_ERROR("Camera not initialized");
        return nullptr;
    }
    
    #ifdef STM32
        // Start DCMI capture
        g_frame_ready = false;
        if (HAL_DCMI_Start_DMA(&g_hdcmi, DCMI_MODE_CONTINUOUS, 
                              (uint32_t)g_temp_buffer, CMX_CAMERA_BUFFER_SIZE/4) != HAL_OK) {
            CMX_LOG_ERROR("Failed to start DCMI capture");
            return nullptr;
        }
        
        // Wait for frame completion
        uint32_t timeout = HAL_GetTick() + 1000; // 1 second timeout
        while (!g_frame_ready && HAL_GetTick() < timeout) {
            // Yield to other tasks if RTOS is available
        }
        
        HAL_DCMI_Stop(&g_hdcmi);
        
        if (!g_frame_ready) {
            CMX_LOG_ERROR("Camera capture timeout");
            return nullptr;
        }
        
    #elif defined(ESP32)
        camera_fb_t* fb = esp_camera_fb_get();
        if (!fb) {
            CMX_LOG_ERROR("Failed to capture frame");
            return nullptr;
        }
        
        // Copy frame data to our buffer
        size_t copy_size = CMX_MIN(fb->len, CMX_CAMERA_BUFFER_SIZE);
        memcpy(g_temp_buffer, fb->buf, copy_size);
        
        esp_camera_fb_return(fb);
        
    #else
        // Generic frame capture - implement based on your camera interface
        // This is a placeholder that generates a test pattern
        static uint8_t pattern = 0;
        for (uint32_t i = 0; i < CMX_CAMERA_BUFFER_SIZE; i++) {
            g_temp_buffer[i] = (pattern + i) & 0xFF;
        }
        pattern++;
    #endif
    
    // Convert format if necessary
    #ifdef STM32
        // Assume RGB565 input from STM32 camera
        convert_rgb565_to_rgb888(g_temp_buffer, g_frame_buffer, 
                                CMX_INPUT_WIDTH * CMX_INPUT_HEIGHT);
    #elif defined(ESP32)
        // ESP32 camera typically outputs JPEG or RGB565
        // This assumes RGB565 - modify based on your camera configuration
        convert_rgb565_to_rgb888(g_temp_buffer, g_frame_buffer,
                                CMX_INPUT_WIDTH * CMX_INPUT_HEIGHT);
    #else
        // Direct copy for generic case
        memcpy(g_frame_buffer, g_temp_buffer, CMX_CAMERA_BUFFER_SIZE);
    #endif
    
    return g_frame_buffer;
}

void camera_deinit() {
    if (!g_camera_initialized) {
        return;
    }
    
    #ifdef STM32
        HAL_DCMI_Stop(&g_hdcmi);
        HAL_DCMI_DeInit(&g_hdcmi);
        HAL_DMA_DeInit(&g_hdma_dcmi);
    #elif defined(ESP32)
        esp_camera_deinit();
    #endif
    
    g_camera_initialized = false;
    CMX_LOG_INFO("Camera deinitialized");
}

// ============================================================================
// STM32-specific Implementation
// ============================================================================

#ifdef STM32

static int init_stm32_camera() {
    // Initialize DCMI peripheral
    g_hdcmi.Instance = DCMI;
    g_hdcmi.Init.SynchroMode = DCMI_SYNCHRO_HARDWARE;
    g_hdcmi.Init.PCKPolarity = DCMI_PCKPOLARITY_FALLING;
    g_hdcmi.Init.VSPolarity = DCMI_VSPOLARITY_LOW;
    g_hdcmi.Init.HSPolarity = DCMI_HSPOLARITY_LOW;
    g_hdcmi.Init.CaptureRate = DCMI_CR_ALL_FRAME;
    g_hdcmi.Init.ExtendedDataMode = DCMI_EXTEND_DATA_8B;
    g_hdcmi.Init.JPEGMode = DCMI_JPEG_DISABLE;
    g_hdcmi.Init.ByteSelectMode = DCMI_BSM_ALL;
    g_hdcmi.Init.ByteSelectStart = DCMI_OEBS_ODD;
    g_hdcmi.Init.LineSelectMode = DCMI_LSM_ALL;
    g_hdcmi.Init.LineSelectStart = DCMI_OELS_ODD;
    
    if (HAL_DCMI_Init(&g_hdcmi) != HAL_OK) {
        return CMX_ERROR_CAMERA;
    }
    
    // Configure DMA for DCMI
    g_hdma_dcmi.Instance = DMA2_Stream1;
    g_hdma_dcmi.Init.Channel = DMA_CHANNEL_1;
    g_hdma_dcmi.Init.Direction = DMA_PERIPH_TO_MEMORY;
    g_hdma_dcmi.Init.PeriphInc = DMA_PINC_DISABLE;
    g_hdma_dcmi.Init.MemInc = DMA_MINC_ENABLE;
    g_hdma_dcmi.Init.PeriphDataAlignment = DMA_PDATAALIGN_WORD;
    g_hdma_dcmi.Init.MemDataAlignment = DMA_MDATAALIGN_WORD;
    g_hdma_dcmi.Init.Mode = DMA_NORMAL;
    g_hdma_dcmi.Init.Priority = DMA_PRIORITY_HIGH;
    g_hdma_dcmi.Init.FIFOMode = DMA_FIFOMODE_ENABLE;
    g_hdma_dcmi.Init.FIFOThreshold = DMA_FIFO_THRESHOLD_FULL;
    g_hdma_dcmi.Init.MemBurst = DMA_MBURST_SINGLE;
    g_hdma_dcmi.Init.PeriphBurst = DMA_PBURST_SINGLE;
    
    if (HAL_DMA_Init(&g_hdma_dcmi) != HAL_OK) {
        return CMX_ERROR_CAMERA;
    }
    
    __HAL_LINKDMA(&g_hdcmi, DMA_Handle, g_hdma_dcmi);
    
    // Enable DCMI interrupt
    HAL_NVIC_SetPriority(DCMI_IRQn, 5, 0);
    HAL_NVIC_EnableIRQ(DCMI_IRQn);
    
    return CMX_SUCCESS;
}

// DCMI interrupt callback
static void dcmi_frame_complete_callback() {
    g_frame_ready = true;
}

// HAL callback for frame completion
extern "C" void HAL_DCMI_FrameEventCallback(DCMI_HandleTypeDef *hdcmi) {
    dcmi_frame_complete_callback();
}

#endif

// ============================================================================
// ESP32-specific Implementation
// ============================================================================

#ifdef ESP32

static int init_esp32_camera() {
    camera_config_t config;
    config.ledc_channel = LEDC_CHANNEL_0;
    config.ledc_timer = LEDC_TIMER_0;
    config.pin_d0 = 5;
    config.pin_d1 = 18;
    config.pin_d2 = 19;
    config.pin_d3 = 21;
    config.pin_d4 = 36;
    config.pin_d5 = 39;
    config.pin_d6 = 34;
    config.pin_d7 = 35;
    config.pin_xclk = 0;
    config.pin_pclk = 22;
    config.pin_vsync = 25;
    config.pin_href = 23;
    config.pin_sscb_sda = 26;
    config.pin_sscb_scl = 27;
    config.pin_pwdn = 32;
    config.pin_reset = -1;
    config.xclk_freq_hz = 20000000;
    config.pixel_format = PIXFORMAT_RGB565;
    config.frame_size = FRAMESIZE_QVGA; // 320x240
    config.jpeg_quality = 12;
    config.fb_count = 1;
    config.grab_mode = CAMERA_GRAB_WHEN_EMPTY;
    
    esp_err_t err = esp_camera_init(&config);
    if (err != ESP_OK) {
        CMX_LOG_ERROR("ESP32 camera init failed: 0x%x", err);
        return CMX_ERROR_CAMERA;
    }
    
    // Get camera sensor handle for configuration
    sensor_t* s = esp_camera_sensor_get();
    if (s) {
        s->set_brightness(s, 0);     // -2 to 2
        s->set_contrast(s, 0);       // -2 to 2
        s->set_saturation(s, 0);     // -2 to 2
        s->set_special_effect(s, 0); // 0 to 6 (0-No Effect, 1-Negative, 2-Grayscale, 3-Red Tint, 4-Green Tint, 5-Blue Tint, 6-Sepia)
        s->set_whitebal(s, 1);       // 0 = disable , 1 = enable
        s->set_awb_gain(s, 1);       // 0 = disable , 1 = enable
        s->set_wb_mode(s, 0);        // 0 to 4 - if awb_gain enabled (0 - Auto, 1 - Sunny, 2 - Cloudy, 3 - Office, 4 - Home)
        s->set_exposure_ctrl(s, 1);  // 0 = disable , 1 = enable
        s->set_aec2(s, 0);           // 0 = disable , 1 = enable
        s->set_ae_level(s, 0);       // -2 to 2
        s->set_aec_value(s, 300);    // 0 to 1200
        s->set_gain_ctrl(s, 1);      // 0 = disable , 1 = enable
        s->set_agc_gain(s, 0);       // 0 to 30
        s->set_gainceiling(s, (gainceiling_t)0);  // 0 to 6
        s->set_bpc(s, 0);            // 0 = disable , 1 = enable
        s->set_wpc(s, 1);            // 0 = disable , 1 = enable
        s->set_raw_gma(s, 1);        // 0 = disable , 1 = enable
        s->set_lenc(s, 1);           // 0 = disable , 1 = enable
        s->set_hmirror(s, 0);        // 0 = disable , 1 = enable
        s->set_vflip(s, 0);          // 0 = disable , 1 = enable
        s->set_dcw(s, 1);            // 0 = disable , 1 = enable
        s->set_colorbar(s, 0);       // 0 = disable , 1 = enable
    }
    
    return CMX_SUCCESS;
}

#endif

// ============================================================================
// Camera Register Configuration (Generic)
// ============================================================================

static int configure_camera_registers() {
    // This function configures camera registers for OV7670/OV2640
    // In a real implementation, you would use I2C to configure the camera
    
    CMX_LOG_INFO("Configuring camera registers for %dx%d", CMX_INPUT_WIDTH, CMX_INPUT_HEIGHT);
    
    // OV7670 register configuration for QVGA (320x240)
    struct camera_reg {
        uint8_t reg;
        uint8_t val;
    };
    
    static const camera_reg ov7670_qvga_regs[] = {
        {0x12, 0x80}, // Reset
        {0x11, 0x01}, // Clock prescaler
        {0x3a, 0x04}, // TLSB
        {0x12, 0x00}, // VGA
        {0x8c, 0x00}, // RGB 444
        {0x04, 0x00}, // COM1
        {0x40, 0xc0}, // COM15
        {0x3e, 0x00}, // COM14
        {0x70, 0x3a}, // Scaling X
        {0x71, 0x35}, // Scaling Y  
        {0x72, 0x11}, // Scaling dcwctr
        {0x73, 0xf0}, // Scaling pclk_div
        {0xa2, 0x02}, // Scaling pclk_delay
        {0x15, 0x00}, // COM10
        {0x7a, 0x20}, // Gamma curve
        {0x7b, 0x10},
        {0x7c, 0x1e},
        {0x7d, 0x35},
        {0x7e, 0x5a},
        {0x7f, 0x69},
        {0x80, 0x76},
        {0x81, 0x80},
        {0x82, 0x88},
        {0x83, 0x8f},
        {0x84, 0x96},
        {0x85, 0xa3},
        {0x86, 0xaf},
        {0x87, 0xc4},
        {0x88, 0xd7},
        {0x89, 0xe8},
        {0x13, 0xe0}, // COM8
        {0x00, 0x00}, // Gain
        {0x10, 0x00}, // AECH
        {0x0d, 0x40}, // COM4
        {0x14, 0x18}, // COM9
        {0xa5, 0x05}, // BD50MAX
        {0xab, 0x07}, // BD60MAX
        {0x24, 0x95}, // AEW
        {0x25, 0x33}, // AEB
        {0x26, 0xe3}, // VPT
        {0x9f, 0x78}, // HAECC1
        {0xa0, 0x68}, // HAECC2
        {0xa1, 0x03}, // MAGIC
        {0xa6, 0xd8}, // HAECC3
        {0xa7, 0xd8}, // HAECC4
        {0xa8, 0xf0}, // HAECC5
        {0xa9, 0x90}, // HAECC6
        {0xaa, 0x94}, // HAECC7
        {0x13, 0xe5}, // COM8
        {0xff, 0xff}  // End marker
    };
    
    // In a real implementation, write these registers via I2C
    // For now, just log that we're configuring
    for (int i = 0; ov7670_qvga_regs[i].reg != 0xff; i++) {
        // i2c_write_reg(CAMERA_I2C_ADDR, ov7670_qvga_regs[i].reg, ov7670_qvga_regs[i].val);
        CMX_LOG_DEBUG("Config reg 0x%02x = 0x%02x", ov7670_qvga_regs[i].reg, ov7670_qvga_regs[i].val);
    }
    
    return CMX_SUCCESS;
}

// ============================================================================
// Color Space Conversion Functions
// ============================================================================

static int convert_rgb565_to_rgb888(const uint8_t* src, uint8_t* dst, uint32_t pixel_count) {
    const uint16_t* src16 = reinterpret_cast<const uint16_t*>(src);
    
    for (uint32_t i = 0; i < pixel_count; i++) {
        uint16_t pixel = src16[i];
        
        // Extract RGB components from RGB565
        uint8_t r = (pixel >> 11) & 0x1F;
        uint8_t g = (pixel >> 5) & 0x3F;
        uint8_t b = pixel & 0x1F;
        
        // Convert to 8-bit values
        dst[i * 3 + 0] = (r << 3) | (r >> 2); // Red
        dst[i * 3 + 1] = (g << 2) | (g >> 4); // Green
        dst[i * 3 + 2] = (b << 3) | (b >> 2); // Blue
    }
    
    return CMX_SUCCESS;
}

static int convert_yuv422_to_rgb888(const uint8_t* src, uint8_t* dst, uint32_t pixel_count) {
    // YUV422 format: Y0 U0 Y1 V0 (4 bytes for 2 pixels)
    for (uint32_t i = 0; i < pixel_count; i += 2) {
        uint32_t yuv_idx = (i / 2) * 4;
        
        int y0 = src[yuv_idx + 0];
        int u = src[yuv_idx + 1] - 128;
        int y1 = src[yuv_idx + 2];
        int v = src[yuv_idx + 3] - 128;
        
        // Convert first pixel
        int r0 = y0 + ((351 * v) >> 8);
        int g0 = y0 - ((179 * v + 86 * u) >> 8);
        int b0 = y0 + ((443 * u) >> 8);
        
        dst[i * 3 + 0] = CMX_CLAMP(r0, 0, 255);
        dst[i * 3 + 1] = CMX_CLAMP(g0, 0, 255);
        dst[i * 3 + 2] = CMX_CLAMP(b0, 0, 255);
        
        // Convert second pixel (if within bounds)
        if (i + 1 < pixel_count) {
            int r1 = y1 + ((351 * v) >> 8);
            int g1 = y1 - ((179 * v + 86 * u) >> 8);
            int b1 = y1 + ((443 * u) >> 8);
            
            dst[(i + 1) * 3 + 0] = CMX_CLAMP(r1, 0, 255);
            dst[(i + 1) * 3 + 1] = CMX_CLAMP(g1, 0, 255);
            dst[(i + 1) * 3 + 2] = CMX_CLAMP(b1, 0, 255);
        }
    }
    
    return CMX_SUCCESS;
}

static void rgb_to_grayscale(const uint8_t* rgb, uint8_t* gray, uint32_t pixel_count) {
    for (uint32_t i = 0; i < pixel_count; i++) {
        // Standard grayscale conversion: 0.299*R + 0.587*G + 0.114*B
        uint32_t r = rgb[i * 3 + 0];
        uint32_t g = rgb[i * 3 + 1];
        uint32_t b = rgb[i * 3 + 2];
        
        gray[i] = (uint8_t)((299 * r + 587 * g + 114 * b) / 1000);
    }
}