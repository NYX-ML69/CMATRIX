#pragma once

#include <cstdint>
#include <cstddef>

namespace cmx::platform::riscv {

/**
 * @brief DMA Channel Configuration
 */
struct DMAConfig {
    uint32_t channel_id;        // DMA channel number (0-7 typically)
    uint32_t src_addr;          // Source address
    uint32_t dst_addr;          // Destination address
    uint32_t transfer_size;     // Number of bytes to transfer
    uint32_t src_increment;     // Source address increment (0=fixed, 1=increment)
    uint32_t dst_increment;     // Destination address increment (0=fixed, 1=increment)
    uint32_t data_width;        // Data width: 1=byte, 2=halfword, 4=word
    uint32_t burst_size;        // Burst transfer size
    bool enable_interrupt;      // Enable completion interrupt
};

/**
 * @brief DMA Transfer Status
 */
enum class DMAStatus : uint8_t {
    IDLE = 0,
    BUSY = 1,
    COMPLETE = 2,
    ERROR = 3
};

/**
 * @brief DMA Transfer Direction
 */
enum class DMADirection : uint8_t {
    MEM_TO_MEM = 0,
    MEM_TO_PERIPH = 1,
    PERIPH_TO_MEM = 2,
    PERIPH_TO_PERIPH = 3
};

/**
 * @brief DMA callback function type
 */
using DMACallback = void(*)(uint32_t channel, DMAStatus status, void* user_data);

class RISCV_DMA {
public:
    /**
     * @brief Initialize DMA controller
     * @return true if successful, false otherwise
     */
    static bool initialize();

    /**
     * @brief Deinitialize DMA controller
     */
    static void deinitialize();

    /**
     * @brief Configure a DMA channel
     * @param config DMA configuration structure
     * @return true if successful, false otherwise
     */
    static bool configure_channel(const DMAConfig& config);

    /**
     * @brief Start DMA transfer on specified channel
     * @param channel_id DMA channel ID
     * @param direction Transfer direction
     * @param callback Optional completion callback
     * @param user_data User data for callback
     * @return true if transfer started successfully
     */
    static bool start_transfer(uint32_t channel_id, 
                              DMADirection direction = DMADirection::MEM_TO_MEM,
                              DMACallback callback = nullptr,
                              void* user_data = nullptr);

    /**
     * @brief Stop DMA transfer on specified channel
     * @param channel_id DMA channel ID
     * @return true if stopped successfully
     */
    static bool stop_transfer(uint32_t channel_id);

    /**
     * @brief Check if DMA channel is busy
     * @param channel_id DMA channel ID
     * @return true if channel is busy
     */
    static bool is_busy(uint32_t channel_id);

    /**
     * @brief Get DMA channel status
     * @param channel_id DMA channel ID
     * @return Current status of the channel
     */
    static DMAStatus get_status(uint32_t channel_id);

    /**
     * @brief Wait for DMA transfer completion (blocking)
     * @param channel_id DMA channel ID
     * @param timeout_ms Timeout in milliseconds (0 = infinite)
     * @return true if completed successfully, false on timeout/error
     */
    static bool wait_for_completion(uint32_t channel_id, uint32_t timeout_ms = 0);

    /**
     * @brief Perform a blocking memory-to-memory copy using DMA
     * @param dst Destination address
     * @param src Source address
     * @param size Number of bytes to copy
     * @param channel_id DMA channel to use (default: 0)
     * @return true if copy completed successfully
     */
    static bool memcpy_dma(void* dst, const void* src, size_t size, uint32_t channel_id = 0);

    /**
     * @brief Perform a blocking memory set using DMA
     * @param dst Destination address
     * @param value Value to set (will be expanded to match data width)
     * @param size Number of bytes to set
     * @param channel_id DMA channel to use (default: 0)
     * @return true if memset completed successfully
     */
    static bool memset_dma(void* dst, uint8_t value, size_t size, uint32_t channel_id = 0);

    /**
     * @brief Get number of available DMA channels
     * @return Number of DMA channels
     */
    static constexpr uint32_t get_channel_count() { return MAX_DMA_CHANNELS; }

    /**
     * @brief Reset DMA channel to default state
     * @param channel_id DMA channel ID
     */
    static void reset_channel(uint32_t channel_id);

    /**
     * @brief Enable/disable DMA channel interrupt
     * @param channel_id DMA channel ID
     * @param enable true to enable, false to disable
     */
    static void set_interrupt_enable(uint32_t channel_id, bool enable);

private:
    static constexpr uint32_t MAX_DMA_CHANNELS = 8;
    static constexpr uint32_t DMA_TIMEOUT_MS = 5000;
    
    // Platform-specific register addresses (to be defined based on SoC)
    static constexpr uintptr_t DMA_BASE_ADDR = 0x40020000;
    
    // Internal helper functions
    static bool validate_channel(uint32_t channel_id);
    static bool validate_transfer_params(const DMAConfig& config);
    static void handle_dma_interrupt(uint32_t channel_id);
    static uint32_t calculate_burst_config(uint32_t burst_size);
};

// Inline utility functions for performance-critical operations
inline bool is_dma_aligned(const void* ptr, uint32_t alignment = 4) {
    return (reinterpret_cast<uintptr_t>(ptr) & (alignment - 1)) == 0;
}

inline uint32_t align_size(uint32_t size, uint32_t alignment = 4) {
    return (size + alignment - 1) & ~(alignment - 1);
}

} // namespace cmx::platform::riscv