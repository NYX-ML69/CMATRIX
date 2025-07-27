/**
 * @file cmx_zephyr_port.cpp
 * @brief Implementation of runtime-to-platform bridge for Zephyr RTOS
 */

#include "cmx_zephyr_port.hpp"

#include <zephyr/kernel.h>
#include <zephyr/logging/log.h>
#include <zephyr/sys/printk.h>
#include <zephyr/fatal.h>
#include <zephyr/sys/heap.h>

#include <stdarg.h>
#include <stdio.h>

LOG_MODULE_REGISTER(cmx_port, LOG_LEVEL_DBG);

namespace cmx::platform::zephyr {

// Static buffer for formatted logging
static char log_buffer[256];

void platform_init() {
    LOG_INF("Platform port layer initialized");
    LOG_INF("Kernel version: %s", KERNEL_VERSION_STRING);
    LOG_INF("System tick rate: %d Hz", CONFIG_SYS_CLOCK_TICKS_PER_SEC);
}

uint64_t cmx_get_timestamp_us() {
    // Zephyr's k_uptime_get() returns milliseconds
    // Convert to microseconds for higher precision
    return k_uptime_get() * 1000ULL;
}

void cmx_log(const char* message) {
    if (message == nullptr) {
        return;
    }
    
    // Use Zephyr's logging if available, otherwise fallback to printk
    #ifdef CONFIG_LOG
        LOG_INF("%s", message);
    #else
        printk("[CMX] %s\n", message);
    #endif
}

void cmx_log_fmt(const char* format, ...) {
    if (format == nullptr) {
        return;
    }
    
    va_list args;
    va_start(args, format);
    
    int ret = vsnprintf(log_buffer, sizeof(log_buffer), format, args);
    va_end(args);
    
    if (ret > 0) {
        // Ensure null termination
        log_buffer[sizeof(log_buffer) - 1] = '\0';
        cmx_log(log_buffer);
    }
}

void cmx_yield() {
    k_yield();
}

void cmx_sleep_ms(uint32_t ms) {
    k_sleep(K_MSEC(ms));
}

void cmx_sleep_us(uint32_t us) {
    // For very short delays, use busy wait
    if (us < 1000) {
        k_busy_wait(us);
    } else {
        // Convert to milliseconds for k_sleep
        k_sleep(K_USEC(us));
    }
}

uint32_t cmx_get_tick_rate() {
    return CONFIG_SYS_CLOCK_TICKS_PER_SEC;
}

bool cmx_in_isr() {
    return k_is_in_isr();
}

unsigned int cmx_irq_lock() {
    return irq_lock();
}

void cmx_irq_unlock(unsigned int key) {
    irq_unlock(key);
}

uint32_t cmx_get_thread_id() {
    k_tid_t current = k_current_get();
    // Cast thread pointer to uint32_t for simple ID
    return (uint32_t)(uintptr_t)current;
}

size_t cmx_get_free_heap() {
    #ifdef CONFIG_HEAP_MEM_POOL_SIZE
        // If heap is configured, try to get free space
        // This is a simplified implementation
        return CONFIG_HEAP_MEM_POOL_SIZE; // Placeholder
    #else
        return 0; // No heap configured
    #endif
}

[[noreturn]] void cmx_panic(const char* reason) {
    LOG_ERR("CMX PANIC: %s", reason ? reason : "Unknown error");
    
    // Disable interrupts
    irq_lock();
    
    // Use Zephyr's fatal error handler
    k_fatal_halt(K_ERR_KERNEL_PANIC);
    
    // Should never reach here
    while (1) {
        /* spin */
    }
}

} // namespace cmx::platform::zephyr