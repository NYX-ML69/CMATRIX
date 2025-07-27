/**
 * @file cmx_zephyr_startup.cpp
 * @brief Boot-time integration for cmatrix runtime with Zephyr RTOS
 * 
 * Initializes the cmatrix runtime components after Zephyr kernel starts.
 * Sets up memory pools, DMA channels, timers, and logging subsystems.
 */

#include <zephyr/kernel.h>
#include <zephyr/logging/log.h>
#include <zephyr/sys/printk.h>
#include <zephyr/init.h>

#include "cmx_zephyr_memory.hpp"
#include "cmx_zephyr_dma.hpp"
#include "cmx_zephyr_timer.hpp"
#include "cmx_zephyr_port.hpp"

LOG_MODULE_REGISTER(cmx_startup, LOG_LEVEL_INF);

namespace cmx::platform::zephyr {

/**
 * @brief Initialize cmatrix runtime subsystems
 * Called during Zephyr system initialization
 */
static int cmx_runtime_init(void) {
    LOG_INF("Initializing cmatrix runtime for Zephyr...");
    
    // Initialize memory management subsystem
    if (!memory_init()) {
        LOG_ERR("Failed to initialize memory subsystem");
        return -1;
    }
    LOG_INF("Memory subsystem initialized");
    
    // Initialize DMA subsystem
    if (!dma_init()) {
        LOG_ERR("Failed to initialize DMA subsystem");
        return -1;
    }
    LOG_INF("DMA subsystem initialized");
    
    // Initialize timer subsystem
    if (!timer_init()) {
        LOG_ERR("Failed to initialize timer subsystem");
        return -1;
    }
    LOG_INF("Timer subsystem initialized");
    
    // Initialize platform port layer
    platform_init();
    LOG_INF("Platform port layer initialized");
    
    LOG_INF("cmatrix runtime initialization complete");
    return 0;
}

/**
 * @brief Cleanup cmatrix runtime subsystems
 * Called during system shutdown or reset
 */
void cmx_runtime_cleanup(void) {
    LOG_INF("Cleaning up cmatrix runtime...");
    
    dma_cleanup();
    memory_cleanup();
    timer_cleanup();
    
    LOG_INF("cmatrix runtime cleanup complete");
}

} // namespace cmx::platform::zephyr

// Register initialization function with Zephyr
SYS_INIT(cmx::platform::zephyr::cmx_runtime_init, APPLICATION, 
         CONFIG_APPLICATION_INIT_PRIORITY);

/**
 * @brief Main application entry point for cmatrix runtime
 * Can be called from main() or used as a Zephyr thread entry
 */
extern "C" void cmx_app_main(void) {
    using namespace cmx::platform::zephyr;
    
    printk("Starting cmatrix application...\n");
    
    // Runtime is already initialized via SYS_INIT
    // Application-specific initialization can go here
    
    // Example: Wait for system to be ready
    k_sleep(K_MSEC(100));
    
    LOG_INF("cmatrix application ready");
    
    // Main application loop would go here
    while (1) {
        // Yield to other threads
        cmx_yield()