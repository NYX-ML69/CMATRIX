/**
 * @file cmx_nios2_dma.hpp
 * @brief DMA-based memory transfer utility for Nios II platform
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
 * @brief DMA transfer direction
 */
enum class DmaDirection : uint8_t {
    MEMORY_TO_MEMORY = 0,
    MEMORY_TO_DEVICE = 1,
    DEVICE_TO_MEMORY = 2,
    DEVICE_TO_DEVICE = 3
};

/**
 * @brief DMA transfer priority
 */
enum class DmaPriority : uint8_t {
    LOW    = 0,
    NORMAL = 1,
    HIGH   = 2,
    URGENT = 3
};

/**
 * @brief DMA transfer status
 */
enum class DmaStatus : uint8_t {
    IDLE      = 0,
    BUSY      = 1,
    COMPLETE  = 2,
    ERROR     = 3,
    CANCELLED = 4
};

/**
 * @brief DMA channel configuration
 */
struct DmaChannelConfig {
    uint8_t channel_id;          ///< DMA channel identifier
    DmaDirection direction;      ///< Transfer direction
    DmaPriority priority;        ///< Transfer priority
    bool interrupt_enable;       ///< Enable completion interrupt
    bool auto_increment_src;     ///< Auto-increment source address
    bool auto_increment_dst;     ///< Auto-increment destination address
    uint8_t burst_size;          ///< Burst transfer size (1, 2, 4, 8)
    uint16_t timeout_ms;         ///< Transfer timeout in milliseconds
};

/**
 * @brief DMA transfer descriptor
 */
struct DmaTransfer {
    void* destination;           ///< Destination address
    const void* source;          ///< Source address
    size_t size;                 ///< Transfer size in bytes
    DmaChannelConfig config;     ///< Channel configuration
    volatile DmaStatus status;   ///< Current transfer status
    uint64_t start_time_us;      ///< Transfer start timestamp
    uint64_t end_time_us;        ///< Transfer completion timestamp
    uint32_t transfer_id;        ///< Unique transfer identifier
};

/**
 * @brief DMA statistics
 */
struct DmaStats {
    uint32_t total_transfers;    ///< Total transfers initiated
    uint32_t successful_transfers; ///< Successfully completed transfers
    uint32_t failed_transfers;   ///< Failed transfers
    uint32_t cancelled_transfers; ///< Cancelled transfers
    uint64_t total_bytes;        ///< Total bytes transferred
    uint64_t total_time_us;      ///< Total transfer time
    uint32_t max_transfer_size;  ///< Largest single transfer
    uint32_t avg_throughput_mbps; ///< Average throughput in MB/s
    uint16_t error_count[8];     ///< Error counts by type
};

/**
 * @brief DMA callback function type
 * @param transfer Completed transfer descriptor
 * @param user_data User-provided callback data
 */
typedef void (*DmaCallback)(const DmaTransfer* transfer, void* user_data);

// =============================================================================
// Core DMA Functions
// =============================================================================

/**
 * @brief Initialize DMA subsystem
 * @return true if initialization successful
 */
bool cmx_dma_init();

/**
 * @brief Cleanup DMA subsystem
 */
void cmx_dma_cleanup();

/**
 * @brief Check if DMA is available on this platform
 * @return true if DMA is supported
 */
bool cmx_dma_available();

/**
 * @brief Perform DMA transfer (blocking)
 * @param dst Destination address
 * @param src Source address
 * @param size Transfer size in bytes
 * @return true if transfer successful
 */
bool cmx_dma_transfer(void* dst, const void* src, size_t size);

/**
 * @brief Perform DMA transfer with configuration (blocking)
 * @param dst Destination address
 * @param src Source address
 * @param size Transfer size in bytes
 * @param config DMA channel configuration
 * @return true if transfer successful
 */
bool cmx_dma_transfer_config(void* dst, const void* src, size_t size, 
                            const DmaChannelConfig& config);

/**
 * @brief Start asynchronous DMA transfer
 * @param dst Destination address
 * @param src Source address
 * @param size Transfer size in bytes
 * @param config DMA channel configuration
 * @param callback Completion callback (optional)
 * @param user_data User data for callback (optional)
 * @return Transfer ID (0 if failed)
 */
uint32_t cmx_dma_transfer_async(void* dst, const void* src, size_t size,
                               const DmaChannelConfig& config,
                               DmaCallback callback = nullptr,
                               void* user_data = nullptr);

// =============================================================================
// DMA Channel Management
// =============================================================================

/**
 * @brief Allocate a DMA channel
 * @param priority Desired priority level
 * @return Channel ID (0xFF if no channels available)
 */
uint8_t cmx_dma_allocate_channel(DmaPriority priority = DmaPriority::NORMAL);

/**
 * @brief Release a DMA channel
 * @param channel_id Channel to release
 * @return true if successful
 */
bool cmx_dma_release_channel(uint8_t channel_id);

/**
 * @brief Check if channel is available
 * @param channel_id Channel to check
 * @return true if channel is available
 */
bool cmx_dma_is_channel_available(uint8_t channel_id);

/**
 * @brief Wait for channel to become idle
 * @param channel_id Channel to wait for
 * @param timeout_ms Timeout in milliseconds (0 = no timeout)
 * @return true if channel became idle
 */
bool cmx_dma_wait_channel_idle(uint8_t channel_id, uint32_t timeout_ms = 0);

// =============================================================================
// Transfer Status and Control
// =============================================================================

/**
 * @brief Get transfer status by ID
 * @param transfer_id Transfer identifier
 * @return Current transfer status
 */
DmaStatus cmx_dma_get_transfer_status(uint32_t transfer_id);

/**
 * @brief Cancel ongoing transfer
 * @param transfer_id Transfer identifier
 * @return true if cancellation successful
 */
bool cmx_dma_cancel_transfer(uint32_t transfer_id);

/**
 * @brief Wait for transfer completion
 * @param transfer_id Transfer identifier
 * @param timeout_ms Timeout in milliseconds (0 = no timeout)
 * @return true if transfer completed successfully
 */
bool cmx_dma_wait_transfer_complete(uint32_t transfer_id, uint32_t timeout_ms = 0);

/**
 * @brief Get transfer progress
 * @param transfer_id Transfer identifier
 * @return Bytes transferred so far
 */
size_t cmx_dma_get_transfer_progress(uint32_t transfer_id);

// =============================================================================
// DMA Configuration and Capabilities
// =============================================================================

/**
 * @brief Get number of available DMA channels
 * @return Number of DMA channels
 */
uint8_t cmx_dma_get_channel_count();

/**
 * @brief Get maximum transfer size
 * @return Maximum single transfer size in bytes
 */
size_t cmx_dma_get_max_transfer_size();

/**
 * @brief Get DMA alignment requirements
 * @return Required memory alignment in bytes
 */
size_t cmx_dma_get_alignment_requirement();

/**
 * @brief Check if address is DMA-accessible
 * @param addr Address to check
 * @return true if address can be used for DMA
 */
bool cmx_dma_is_address_valid(const void* addr);

/**
 * @brief Get supported burst sizes
 * @return Bitmask of supported burst sizes
 */
uint8_t cmx_dma_get_supported_burst_sizes();

// =============================================================================
// Statistics and Monitoring
// =============================================================================

/**
 * @brief Get DMA statistics
 * @return DMA statistics structure
 */
const DmaStats& cmx_dma_get_stats();

/**
 * @brief Reset DMA statistics
 */
void cmx_dma_reset_stats();

/**
 * @brief Get current DMA throughput
 * @return Current throughput in MB/s
 */
uint32_t cmx_dma_get_current_throughput();

/**
 * @brief Get channel utilization
 * @param channel_id Channel to check
 * @return Utilization percentage (0-100)
 */
uint8_t cmx_dma_get_channel_utilization(uint8_t channel_id);

// =============================================================================
// Debug and Testing Functions
// =============================================================================

/**
 * @brief Perform DMA self-test
 * @return true if self-test passed
 */
bool cmx_dma_self_test();

/**
 * @brief Benchmark DMA performance
 * @param size Transfer size for benchmark
 * @param iterations Number of test iterations
 * @return Average throughput in MB/s
 */
uint32_t cmx_dma_benchmark(size_t size, uint32_t iterations = 10);

/**
 * @brief Test DMA with specific pattern
 * @param pattern Test pattern to use
 * @param size Test buffer size
 * @return true if test passed
 */
bool cmx_dma_pattern_test(uint32_t pattern, size_t size);

/**
 * @brief Enable DMA debug logging
 * @param enable Enable or disable debug output
 */
void cmx_dma_set_debug_enabled(bool enable);

// =============================================================================
// Memory Copy Fallback Functions
// =============================================================================

/**
 * @brief Optimized memory copy (uses DMA if available, otherwise memcpy)
 * @param dst Destination address
 * @param src Source address
 * @param size Number of bytes to copy
 * @return true if copy successful
 */
bool cmx_memcpy_optimized(void* dst, const void* src, size_t size);

/**
 * @brief Memory set using DMA (if available)
 * @param dst Destination address
 * @param value Value to set
 * @param size Number of bytes to set
 * @return true if operation successful
 */
bool cmx_memset_dma(void* dst, uint8_t value, size_t size);

/**
 * @brief Memory compare using DMA acceleration (if available)
 * @param ptr1 First memory block
 * @param ptr2 Second memory block
 * @param size Number of bytes to compare
 * @return 0 if equal, non-zero if different
 */
int cmx_memcmp_dma(const void* ptr1, const void* ptr2, size_t size);

// =============================================================================
// Constants
// =============================================================================

/// Maximum number of DMA channels supported
constexpr uint8_t CMX_DMA_MAX_CHANNELS = 8;

/// Default DMA timeout in milliseconds
constexpr uint16_t CMX_DMA_DEFAULT_TIMEOUT_MS = 1000;

/// Minimum transfer size for DMA (smaller transfers use memcpy)
constexpr size_t CMX_DMA_MIN_TRANSFER_SIZE = 64;

/// DMA descriptor alignment requirement
constexpr size_t CMX_DMA_DESCRIPTOR_ALIGNMENT = 8;

/// Maximum concurrent async transfers
constexpr uint32_t CMX_DMA_MAX_ASYNC_TRANSFERS = 16;

/// DMA error codes
constexpr uint16_t CMX_DMA_ERROR_NONE = 0x0000;
constexpr uint16_t CMX_DMA_ERROR_TIMEOUT = 0x0001;
constexpr uint16_t CMX_DMA_ERROR_INVALID_ADDRESS = 0x0002;
constexpr uint16_t CMX_DMA_ERROR_ALIGNMENT = 0x0003;
constexpr uint16_t CMX_DMA_ERROR_SIZE = 0x0004;
constexpr uint16_t CMX_DMA_ERROR_CHANNEL_BUSY = 0x0005;
constexpr uint16_t CMX_DMA_ERROR_NO_CHANNELS = 0x0006;
constexpr uint16_t CMX_DMA_ERROR_HARDWARE = 0x0007;

} // namespace nios2
} // namespace platform
} // namespace cmx