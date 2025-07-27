/**
 * @file cmx_zephyr_dma.hpp
 * @brief DMA abstraction layer for Zephyr RTOS
 * 
 * Provides hardware-accelerated memory transfers using Zephyr's
 * DMA subsystem for efficient tensor and data movement.
 */

#pragma once

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

namespace cmx::platform::zephyr {

/**
 * @brief DMA transfer configuration
 */
struct DmaConfig {
    uint32_t channel_id;        ///< DMA channel identifier
    uint32_t source_width;      ///< Source data width in bytes
    uint32_t dest_width;        ///< Destination data width in bytes
    bool     increment_src;     ///< Increment source address
    bool     increment_dst;     ///< Increment destination address
    uint32_t priority;          ///< Transfer priority (0=highest)
};

/**
 * @brief DMA transfer status callback
 * @param channel DMA channel that completed
 * @param status 0 for success, negative for error
 * @param user_data User-provided callback data
 */
typedef void (*dma_callback_t)(uint32_t channel, int status, void* user_data);

/**
 * @brief Initialize DMA subsystem
 * @return true if successful, false otherwise
 * 
 * Sets up DMA controllers and allocates channel resources
 */
bool dma_init();

/**
 * @brief Cleanup DMA subsystem
 * Releases all allocated DMA resources
 */
void dma_cleanup();

/**
 * @brief Allocate a DMA channel
 * @param config DMA configuration parameters
 * @return Channel handle (>=0) on success, negative on error
 */
int dma_allocate_channel(const DmaConfig& config);

/**
 * @brief Release a DMA channel
 * @param channel Channel handle from dma_allocate_channel
 * @return true if successful, false otherwise
 */
bool dma_release_channel(int channel);

/**
 * @brief Perform synchronous DMA transfer
 * @param dst Destination buffer
 * @param src Source buffer  
 * @param size Number of bytes to transfer
 * @return true if successful, false otherwise
 * 
 * Blocks until transfer completes or fails
 */
bool cmx_dma_transfer(void* dst, const void* src, size_t size);

/**
 * @brief Perform asynchronous DMA transfer
 * @param channel DMA channel handle
 * @param dst Destination buffer
 * @param src Source buffer
 * @param size Number of bytes to transfer
 * @param callback Completion callback (optional)
 * @param user_data User data for callback (optional)
 * @return true if transfer started, false on error
 */
bool dma_transfer_async(int channel, void* dst, const void* src, size_t size,
                       dma_callback_t callback = nullptr, void* user_data = nullptr);

/**
 * @brief Wait for DMA transfer completion
 * @param channel DMA channel handle
 * @param timeout_ms Timeout in milliseconds (0 = no timeout)
 * @return true if completed successfully, false on timeout/error
 */
bool dma_wait_completion(int channel, uint32_t timeout_ms = 0);

/**
 * @brief Cancel ongoing DMA transfer
 * @param channel DMA channel handle
 * @return true if cancelled, false otherwise
 */
bool dma_cancel_transfer(int channel);

/**
 * @brief Check if DMA transfer is active
 * @param channel DMA channel handle
 * @return true if transfer in progress, false otherwise
 */
bool dma_is_active(int channel);

/**
 * @brief Get DMA transfer progress
 * @param channel DMA channel handle
 * @return Number of bytes transferred so far
 */
size_t dma_get_progress(int channel);

/**
 * @brief Configure memory-to-memory DMA transfer
 * @param channel DMA channel handle
 * @param src_width Source data width (1, 2, 4, 8 bytes)
 * @param dst_width Destination data width (1, 2, 4, 8 bytes)
 * @return true if configured successfully, false otherwise
 */
bool dma_configure_mem2mem(int channel, uint32_t src_width, uint32_t dst_width);

/**
 * @brief Get number of available DMA channels
 * @return Total number of DMA channels
 */
uint32_t dma_get_channel_count();

/**
 * @brief Check if DMA is available on this platform
 * @return true if DMA hardware is present and initialized
 */
bool dma_is_available();

/**
 * @brief Get DMA hardware capabilities
 * @return Bitmask of supported features
 */
uint32_t dma_get_capabilities();

// DMA capability flags
constexpr uint32_t DMA_CAP_MEM2MEM     = (1U << 0);  ///< Memory-to-memory transfers
constexpr uint32_t DMA_CAP_MEM2PERIPH  = (1U << 1);  ///< Memory-to-peripheral transfers  
constexpr uint32_t DMA_CAP_PERIPH2MEM  = (1U << 2);  ///< Peripheral-to-memory transfers
constexpr uint32_t DMA_CAP_SCATTER_GATHER = (1U << 3); ///< Scatter-gather support
constexpr uint32_t DMA_CAP_BURST       = (1U << 4);  ///< Burst transfers
constexpr uint32_t DMA_CAP_PRIORITY    = (1U << 5);  ///< Priority control

} // namespace cmx::platform::zephyr