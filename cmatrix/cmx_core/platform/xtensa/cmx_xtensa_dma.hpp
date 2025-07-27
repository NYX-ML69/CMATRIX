#pragma once

#include <cstdint>
#include <cstddef>

namespace cmx::platform::xtensa {

/**
 * @brief DMA transfer completion callback
 */
using DmaCallback = void(*)(void* user_data, bool success);

/**
 * @brief DMA transfer handle
 */
struct DmaHandle {
    uint8_t channel;
    bool active;
    void* user_data;
    DmaCallback callback;
};

/**
 * @brief DMA transfer parameters
 */
struct DmaTransferParams {
    void* dst;
    const void* src;
    size_t size;
    bool blocking;
    DmaCallback callback;
    void* user_data;
};

/**
 * @brief Initialize DMA subsystem
 */
void dma_init();

/**
 * @brief Synchronous DMA transfer
 * @param dst Destination address (must be DMA-accessible)
 * @param src Source address (must be DMA-accessible)
 * @param size Number of bytes to transfer
 * @return true if transfer successful, false otherwise
 */
bool cmx_dma_transfer(void* dst, const void* src, size_t size);

/**
 * @brief Asynchronous DMA transfer
 * @param params Transfer parameters
 * @return DMA handle for tracking transfer, nullptr if failed
 */
DmaHandle* cmx_dma_transfer_async(const DmaTransferParams& params);

/**
 * @brief Check if DMA transfer is complete
 * @param handle DMA transfer handle
 * @return true if complete, false if still in progress
 */
bool cmx_dma_is_complete(const DmaHandle* handle);

/**
 * @brief Wait for DMA transfer completion
 * @param handle DMA transfer handle
 * @param timeout_us Timeout in microseconds (0 = infinite)
 * @return true if completed successfully, false if timeout or error
 */
bool cmx_dma_wait(DmaHandle* handle, uint32_t timeout_us = 0);

/**
 * @brief Cancel ongoing DMA transfer
 * @param handle DMA transfer handle
 */
void cmx_dma_cancel(DmaHandle* handle);

/**
 * @brief Get DMA capabilities
 */
struct DmaCapabilities {
    uint8_t num_channels;
    size_t max_transfer_size;
    size_t alignment_requirement;
    bool supports_memory_to_memory;
    bool supports_peripheral_to_memory;
    bool supports_memory_to_peripheral;
    bool supports_scatter_gather;
};

/**
 * @brief Get DMA capabilities
 * @return DMA capabilities structure
 */
const DmaCapabilities& cmx_dma_get_capabilities();

/**
 * @brief Check if address is DMA-accessible
 * @param addr Address to check
 * @param size Size of the memory region
 * @return true if accessible by DMA, false otherwise
 */
bool cmx_dma_is_accessible(const void* addr, size_t size);

/**
 * @brief Allocate DMA-accessible memory
 * @param size Size in bytes
 * @param alignment Alignment requirement (0 = default)
 * @return Pointer to DMA-accessible memory, nullptr if failed
 */
void* cmx_dma_alloc(size_t size, size_t alignment = 0);

/**
 * @brief Free DMA-accessible memory
 * @param ptr Pointer previously returned by cmx_dma_alloc
 */
void cmx_dma_free(void* ptr);

/**
 * @brief DMA performance statistics
 */
struct DmaStats {
    uint32_t transfers_completed;
    uint32_t transfers_failed;
    uint64_t bytes_transferred;
    uint32_t avg_transfer_time_us;
    uint8_t channels_in_use;
};

/**
 * @brief Get DMA performance statistics
 * @return DMA statistics
 */
const DmaStats& cmx_dma_get_stats();

/**
 * @brief Reset DMA statistics
 */
void cmx_dma_reset_stats();

} // namespace cmx::platform::xtensa