/**
 * @file cmx_nios2_port.hpp
 * @brief Core platform abstraction layer for Nios II
 * @author CMatrix Development Team
 * @version 1.0
 */

#pragma once

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

namespace cmx {
namespace platform {
namespace nios2 {

/**
 * @brief Platform initialization flags
 */
enum class PlatformFeature : uint32_t {
    TIMER_SUPPORT    = 0x01,
    DMA_SUPPORT      = 0x02,
    CACHE_SUPPORT    = 0x04,
    INTERRUPT_SUPPORT = 0x08,
    WATCHDOG_SUPPORT = 0x10,
    PERFORMANCE_COUNTERS = 0x20
};

/**
 * @brief System status enumeration
 */
enum class SystemStatus : uint8_t {
    UNINITIALIZED = 0,
    INITIALIZING  = 1,
    READY         = 2,
    ERROR         = 3,
    SHUTDOWN      = 4
};

/**
 * @brief Log level enumeration
 */
enum class LogLevel : uint8_t {
    DEBUG   = 0,
    INFO    = 1,
    WARNING = 2,
    ERROR   = 3,
    FATAL   = 4
};

/**
 * @brief Platform capabilities structure
 */
struct PlatformCapabilities {
    uint32_t cpu_frequency_hz;    ///< CPU clock frequency
    uint32_t supported_features;  ///< Bitmask of PlatformFeature
    uint32_t memory_size;         ///< Available memory in bytes
    uint32_t cache_line_size;     ///< Cache line size (0 if no cache)
    uint16_t timer_resolution_us; ///< Timer resolution in microseconds
    uint8_t  dma_channels;        ///< Number of DMA channels
    uint8_t  interrupt_levels;    ///< Number of interrupt priority levels
};

/**
 * @brief System statistics structure
 */
struct SystemStats {
    uint64_t uptime_us;           ///< System uptime in microseconds
    uint32_t total_allocations;   ///< Total memory allocations
    uint32_t current_allocations; ///< Current active allocations
    uint32_t peak_memory_usage;   ///< Peak memory usage in bytes
    uint32_t dma_transfers;       ///< Total DMA transfers performed
    uint32_t timer_overflows;     ///< Timer overflow events
    uint32_t interrupt_count;     ///< Total interrupts handled
    uint16_t last_error_code;     ///< Last system error code
};

// =============================================================================
// Core Platform Functions
// =============================================================================

/**
 * @brief Initialize the platform abstraction layer
 * @return true if initialization successful
 */
bool cmx_platform_init();

/**
 * @brief Cleanup platform resources
 */
void cmx_platform_cleanup();

/**
 * @brief Get current timestamp in microseconds
 * @return Current system timestamp
 */
uint64_t cmx_get_timestamp_us();

/**
 * @brief Log a message to the system console
 * @param level Log level
 * @param message Message to log
 */
void cmx_log(LogLevel level, const char* message);

/**
 * @brief Log a message with default INFO level
 * @param message Message to log
 */
void cmx_log(const char* message);

/**
 * @brief Yield CPU to other tasks/interrupts
 */
void cmx_yield();

/**
 * @brief Perform a system reset
 */
void cmx_system_reset();

// =============================================================================
// Platform Information Functions
// =============================================================================

/**
 * @brief Get platform capabilities
 * @return Platform capabilities structure
 */
const PlatformCapabilities& cmx_get_platform_capabilities();

/**
 * @brief Get current system status
 * @return Current system status
 */
SystemStatus cmx_get_system_status();

/**
 * @brief Get system statistics
 * @return System statistics structure
 */
const SystemStats& cmx_get_system_stats();

/**
 * @brief Check if a platform feature is supported
 * @param feature Feature to check
 * @return true if feature is supported
 */
bool cmx_is_feature_supported(PlatformFeature feature);

/**
 * @brief Get CPU frequency in Hz
 * @return CPU frequency
 */
uint32_t cmx_get_cpu_frequency();

/**
 * @brief Get available memory size
 * @return Available memory in bytes
 */
uint32_t cmx_get_available_memory();

// =============================================================================
// Cache Management Functions
// =============================================================================

/**
 * @brief Flush data cache for specified memory range
 * @param addr Starting address
 * @param size Size in bytes
 */
void cmx_cache_flush_range(void* addr, size_t size);

/**
 * @brief Invalidate data cache for specified memory range
 * @param addr Starting address
 * @param size Size in bytes
 */
void cmx_cache_invalidate_range(void* addr, size_t size);

/**
 * @brief Flush entire data cache
 */
void cmx_cache_flush_all();

/**
 * @brief Invalidate entire data cache
 */
void cmx_cache_invalidate_all();

// =============================================================================
// Interrupt Management Functions
// =============================================================================

/**
 * @brief Disable interrupts globally
 * @return Previous interrupt state
 */
uint32_t cmx_disable_interrupts();

/**
 * @brief Restore interrupt state
 * @param state Previous interrupt state from cmx_disable_interrupts()
 */
void cmx_restore_interrupts(uint32_t state);

/**
 * @brief Enable specific interrupt
 * @param irq_num Interrupt number
 * @return true if successful
 */
bool cmx_enable_interrupt(uint32_t irq_num);

/**
 * @brief Disable specific interrupt
 * @param irq_num Interrupt number
 * @return true if successful
 */
bool cmx_disable_interrupt(uint32_t irq_num);

// =============================================================================
// Error Handling Functions
// =============================================================================

/**
 * @brief Set system error code
 * @param error_code Error code to set
 */
void cmx_set_error_code(uint16_t error_code);

/**
 * @brief Get last system error code
 * @return Last error code
 */
uint16_t cmx_get_last_error();

/**
 * @brief Clear system error code
 */
void cmx_clear_error();

/**
 * @brief Check if system is in error state
 * @return true if system has error
 */
bool cmx_has_error();

// =============================================================================
// Power Management Functions
// =============================================================================

/**
 * @brief Enter low power mode
 * @param wake_on_interrupt Wake up on interrupt
 */
void cmx_enter_low_power(bool wake_on_interrupt = true);

/**
 * @brief Get current power consumption estimate
 * @return Power consumption in milliwatts (0 if not supported)
 */
uint32_t cmx_get_power_consumption_mw();

// =============================================================================
// Debug and Profiling Functions
// =============================================================================

/**
 * @brief Start performance profiling
 * @param profile_id Unique profile identifier
 */
void cmx_profile_start(uint32_t profile_id);

/**
 * @brief End performance profiling
 * @param profile_id Profile identifier
 * @return Elapsed time in microseconds
 */
uint64_t cmx_profile_end(uint32_t profile_id);

/**
 * @brief Get CPU cycle count
 * @return Current CPU cycle count
 */
uint64_t cmx_get_cpu_cycles();

/**
 * @brief Convert CPU cycles to microseconds
 * @param cycles CPU cycles
 * @return Time in microseconds
 */
uint64_t cmx_cycles_to_us(uint64_t cycles);

/**
 * @brief Convert microseconds to CPU cycles
 * @param us Time in microseconds
 * @return CPU cycles
 */
uint64_t cmx_us_to_cycles(uint64_t us);

// =============================================================================
// Memory Barrier Functions
// =============================================================================

/**
 * @brief Full memory barrier
 */
void cmx_memory_barrier();

/**
 * @brief Data memory barrier
 */
void cmx_data_memory_barrier();

/**
 * @brief Instruction synchronization barrier
 */
void cmx_instruction_barrier();

// =============================================================================
// Platform-Specific Constants
// =============================================================================

/// Maximum log message length
constexpr size_t CMX_MAX_LOG_MESSAGE_SIZE = 256;

/// Default memory alignment
constexpr size_t CMX_MEMORY_ALIGNMENT = 8;

/// Maximum supported DMA channels
constexpr uint8_t CMX_MAX_DMA_CHANNELS = 8;

/// System tick frequency (Hz)
constexpr uint32_t CMX_SYSTEM_TICK_HZ = 1000;

/// Default timer resolution (microseconds)
constexpr uint16_t CMX_DEFAULT_TIMER_RESOLUTION_US = 1;

/// Maximum profile sessions
constexpr uint32_t CMX_MAX_PROFILE_SESSIONS = 16;

} // namespace nios2
} // namespace platform
} // namespace cmx