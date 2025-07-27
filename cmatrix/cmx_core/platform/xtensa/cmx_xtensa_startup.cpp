#include "cmx_xtensa_port.hpp"
#include "cmx_xtensa_memory.hpp"
#include "cmx_xtensa_dma.hpp"
#include "cmx_xtensa_timer.hpp"
#include <xtensa/config/core.h>
#include <xtensa/hal.h>

namespace cmx::platform::xtensa {

static bool g_startup_complete = false;

/**
 * @brief Initialize all platform subsystems in proper order
 */
void startup_init() {
    if (g_startup_complete) {
        return;
    }

    // 1. Initialize memory subsystem first
    cmx_memory_init();
    
    // 2. Initialize timer subsystem for time tracking
    timer_init();
    
    // 3. Initialize DMA subsystem
    dma_init();
    
    // 4. Initialize platform port layer
    cmx_platform_init();
    
    g_startup_complete = true;
}

/**
 * @brief Platform diagnostics - verify subsystems
 */
bool startup_diagnostics() {
    if (!g_startup_complete) {
        return false;
    }
    
    // Test memory allocation
    void* test_ptr = cmx_alloc(64);
    if (!test_ptr) {
        cmx_log("STARTUP: Memory allocation test failed");
        return false;
    }
    cmx_free(test_ptr);
    
    // Test timer functionality
    uint64_t start_time = cmx_now_us();
    cmx_delay_us(100);
    uint64_t elapsed = cmx_now_us() - start_time;
    if (elapsed < 90 || elapsed > 200) {
        cmx_log("STARTUP: Timer precision test failed");
        return false;
    }
    
    // Test DMA functionality
    uint8_t src[32] = {0xAA};
    uint8_t dst[32] = {0x00};
    for (int i = 0; i < 32; i++) src[i] = i;
    
    if (!cmx_dma_transfer(dst, src, 32)) {
        cmx_log("STARTUP: DMA test failed, falling back to memcpy");
    }
    
    cmx_log("STARTUP: Platform initialization complete");
    return true;
}

/**
 * @brief Main entry point called from system boot
 */
extern "C" void cmx_xtensa_main() {
    // Disable interrupts during initialization
    uint32_t int_level = XTOS_SET_INTLEVEL(XCHAL_EXCM_LEVEL);
    
    startup_init();
    
    // Re-enable interrupts
    XTOS_RESTORE_INTLEVEL(int_level);
    
    // Run diagnostics
    if (!startup_diagnostics()) {
        cmx_log("STARTUP: Diagnostics failed, continuing anyway");
    }
    
    // Enter main application loop
    while (true) {
        // Yield control to other tasks/interrupts
        cmx_yield();
        
        // Main application logic would go here
        // For now, just a heartbeat
        static uint64_t last_heartbeat = 0;
        uint64_t now = cmx_now_us();
        if (now - last_heartbeat > 1000000) { // 1 second
            cmx_log("HEARTBEAT: System running");
            last_heartbeat = now;
        }
    }
}

/**
 * @brief Get startup status
 */
bool is_startup_complete() {
    return g_startup_complete;
}

} // namespace cmx::platform::xtensa

// Hook into system main if needed
extern "C" void app_main() {
    cmx::platform::xtensa::cmx_xtensa_main();
}