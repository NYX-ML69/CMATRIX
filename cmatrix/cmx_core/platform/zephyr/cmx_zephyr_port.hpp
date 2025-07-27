/**
 * @file cmx_zephyr_port.hpp
 * @brief Main runtime-to-platform bridge for Zephyr RTOS
 * 
 * Provides the primary abstraction layer between cmatrix runtime
 * and Zephyr kernel services including timing, logging, and scheduling.
 */

#pragma once

#include <stdint.h>
#include <stddef.h>

namespace cmx::platform::zephyr {

/**
 * @brief Initialize the platform port layer
 * Sets up logging, timing, and other core platform services
 */
void platform_init();

/**
 * @brief Get high-resolution timestamp in microseconds
 * @return Current timestamp in microseconds since boot
 * 
 * Uses Zephyr's k_uptime_get() for consistent timing
 */
uint64_t cmx_get_timestamp_us();

/**
 * @brief Log a message through Zephyr's logging subsystem
 * @param message Null-terminated string to log
 * 
 * Routes messages to configured Zephyr log backends
 */
void cmx_log(const char* message);

/**
 * @brief Log a formatted message
 * @param format Printf-style format string
 * @param ... Variable arguments for formatting
 */
void cmx_log_fmt(const char* format, ...);

/**
 * @brief Yield CPU to other threads
 * 
 * Cooperative scheduling hint to Zephyr kernel
 */
void cmx_yield();

/**
 * @brief Put current thread to sleep
 * @param ms Milliseconds to sleep
 */
void cmx_sleep_ms(uint32_t ms);

/**
 * @brief Put current thread to sleep (microseconds)
 * @param us Microseconds to sleep
 * 
 * Note: Actual resolution depends on system tick rate
 */
void cmx_sleep_us(uint32_t us);

/**
 * @brief Get system tick frequency
 * @return Ticks per second
 */
uint32_t cmx_get_tick_rate();

/**
 * @brief Check if we're running in interrupt context
 * @return true if in ISR, false otherwise
 */
bool cmx_in_isr();

/**
 * @brief Disable interrupts
 * @return Previous interrupt state key
 */
unsigned int cmx_irq_lock();

/**
 * @brief Restore interrupt state
 * @param key Interrupt state key from cmx_irq_lock()
 */
void cmx_irq_unlock(unsigned int key);

/**
 * @brief Get current thread ID
 * @return Thread identifier
 */
uint32_t cmx_get_thread_id();

/**
 * @brief Get available heap memory
 * @return Free heap bytes (0 if heap not available)
 */
size_t cmx_get_free_heap();

/**
 * @brief Platform-specific panic handler
 * @param reason Null-terminated panic reason string
 */
[[noreturn]] void cmx_panic(const char* reason);

} // namespace cmx::platform::zephyr