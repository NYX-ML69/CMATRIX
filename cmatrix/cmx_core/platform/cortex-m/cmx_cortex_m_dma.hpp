#pragma once

#include <cstdint>
#include <cstddef>

namespace cmx {
namespace platform {
namespace cortex_m {

/**
 * @brief DMA transfer direction
 */
enum class DmaDirection : uint8_t {
    MEM_TO_MEM = 0,     ///< Memory to memory
    MEM_TO_PERIPH,      ///< Memory to peripheral
    PERIPH_TO_MEM,      ///< Peripheral to memory
    PERIPH_TO_PERIPH    ///< Peripheral to peripheral
};

/**
 * @brief DMA data width configuration
 */
enum class DmaDataWidth : uint8_t {
    BYTE = 1,           ///< 8-bit transfers
    HALFWORD = 2,       ///< 16-bit transfers
    WORD = 4            ///< 32-bit transfers
};

/**
 * @brief DMA priority levels
 */
enum class DmaPriority : uint8_t {
    LOW = 0,
    MEDIUM = 1,
    HIGH = 2,
    VERY_HIGH = 3
};

/**
 * @brief DMA transfer status
 */
enum class DmaStatus : uint8_t {
    IDLE = 0,           ///< DMA channel is idle
    BUSY,               ///< Transfer in progress
    COMPLETE,           ///< Transfer completed successfully
    ERROR,              ///< Transfer error occurred
    TIMEOUT             ///< Transfer timeout
};

/**
 * @brief DMA configuration structure
 */
struct DmaConfig {
    DmaDirection direction;         ///< Transfer direction
    DmaDataWidth src_width;         ///< Source data width
    DmaDataWidth dst_width;         ///< Destination data width
    DmaPriority priority;           ///< Transfer priority
    bool src_increment;             ///< Increment source address
    bool dst_increment;             ///< Increment destination address
    bool circular_mode;             ///< Enable circular mode
    bool enable_interrupts;         ///< Enable completion interrupts
    uint8_t interrupt_priority;     ///< Interrupt priority (0-15)
    uint32_t timeout_ms;            ///< Transfer timeout in milliseconds
};

/**
 * @brief DMA transfer statistics
 */
struct DmaStats {
    uint32_t total_transfers;       ///< Total number of transfers
    uint32_t successful_transfers;  ///< Successful transfers
    uint32_t failed_transfers;      ///< Failed transfers
    uint32_t timeout_transfers;     ///< Timed out transfers
    uint64_t total_bytes;           ///< Total bytes transferred
    uint32_t average_speed_kbps;    ///< Average transfer speed in KB/s
};

/**
 * @brief DMA completion callback function type
 * @param channel DMA channel that completed transfer
 * @param status Transfer completion status
 * @param bytes_transferred Number of bytes successfully transferred
 * @param user_data User-provided callback data
 */
using DmaCallback = void (*)(uint8_t channel, DmaStatus status, size_t bytes_transferred, void* user_data);

/**
 * @brief Platform-specific DMA controller wrapper for Cortex-M
 * 
 * Provides high-performance memory transfers using hardware DMA.
 * Supports both synchronous and asynchronous operations with callbacks.
 * Optimized for tensor data movement in CMX runtime.
 */
class Dma {
public:
    /**
     * @brief Maximum number of DMA channels supported
     */
    static constexpr uint8_t MAX_CHANNELS = 8;
    
    /**
     * @brief Initialize DMA subsystem
     * @param enable_power_management Enable DMA power management
     * @return true if initialization successful
     */
    static bool init(bool enable_power_management = true);
    
    /**
     * @brief Deinitialize DMA subsystem
     */
    static void deinit();
    
    /**
     * @brief Allocate a DMA channel
     * @param config DMA configuration parameters
     * @return Channel number (0-7), or 0xFF if allocation failed
     */
    static uint8_t allocate_channel(const DmaConfig& config = {});
    
    /**
     * @brief Release a DMA channel
     * @param channel Channel number to release
     * @return true if channel was released successfully
     */
    static bool release_channel(uint8_t channel);
    
    /**
     * @brief Synchronous memory-to-memory copy
     * @param dest Destination buffer
     * @param src Source buffer
     * @param size Number of bytes to copy
     * @param channel Optional specific channel to use
     * @return true if copy completed successfully
     */
    static bool copy_sync(void* dest, const void* src, size_t size, uint8_t channel = 0xFF);
    
    /**
     * @brief Asynchronous memory-to-memory copy
     * @param dest Destination buffer
     * @param src Source buffer  
     * @param size Number of bytes to copy
     * @param callback Completion callback function
     * @param user_data User data passed to callback
     * @param channel Optional specific channel to use
     * @return true if transfer was started successfully
     */
    static bool copy_async(void* dest, const void* src, size_t size, 
                          DmaCallback callback = nullptr, void* user_data = nullptr, 
                          uint8_t channel = 0xFF);
    
    /**
     * @brief Synchronous memory set operation
     * @param dest Destination buffer
     * @param value 32-bit value to set (repeated across buffer)
     * @param size Number of bytes to set
     * @param channel Optional specific channel to use
     * @return true if operation completed successfully
     */
    static bool set_sync(void* dest, uint32_t value, size_t size, uint8_t channel = 0xFF);
    
    /**
     * @brief Asynchronous memory set operation
     * @param dest Destination buffer
     * @param value 32-bit value to set (repeated across buffer)
     * @param size Number of bytes to set
     * @param callback Completion callback function
     * @param user_data User data passed to callback
     * @param channel Optional specific channel to use
     * @return true if operation was started successfully
     */
    static bool set_async(void* dest, uint32_t value, size_t size,
                         DmaCallback callback = nullptr, void* user_data = nullptr,
                         uint8_t channel = 0xFF);
    
    /**
     * @brief Configure DMA channel with custom parameters
     * @param channel Channel number to configure
     * @param config Configuration parameters
     * @return true if configuration successful
     */
    static bool configure_channel(uint8_t channel, const DmaConfig& config);
    
    /**
     * @brief Start custom DMA transfer
     * @param channel Channel number
     * @param src_addr Source address
     * @param dst_addr Destination address
     * @param transfer_count Number of data items to transfer
     * @param callback Completion callback
     * @param user_data User callback data
     * @return true if transfer started successfully
     */
    static bool start_transfer(uint8_t channel, const void* src_addr, void* dst_addr, 
                              size_t transfer_count, DmaCallback callback = nullptr, 
                              void* user_data = nullptr);
    
    /**
     * @brief Stop ongoing DMA transfer
     * @param channel Channel number to stop
     * @return true if transfer was stopped successfully
     */
    static bool stop_transfer(uint8_t channel);
    
    /**
     * @brief Check if DMA transfer is complete
     * @param channel Channel number to check
     * @return Transfer status
     */
    static DmaStatus get_transfer_status(uint8_t channel);
    
    /**
     * @brief Wait for DMA transfer completion
     * @param channel Channel number to wait for
     * @param timeout_ms Timeout in milliseconds (0 = no timeout)
     * @return Final transfer status
     */
    static DmaStatus wait_for_completion(uint8_t channel, uint32_t timeout_ms = 5000);
    
    /**
     * @brief Get remaining transfer count
     * @param channel Channel number
     * @return Number of data items remaining to transfer
     */
    static size_t get_remaining_count(uint8_t channel);
    
    /**
     * @brief Check if DMA channel is available
     * @param channel Channel number to check
     * @return true if channel is available for use
     */
    static bool is_channel_available(uint8_t channel);
    
    /**
     * @brief Get DMA transfer statistics
     * @param channel Channel number (0xFF for global stats)
     * @param stats Output statistics structure
     */
    static void get_statistics(uint8_t channel, DmaStats& stats);
    
    /**
     * @brief Reset DMA statistics
     * @param channel Channel number (0xFF for all channels)
     */
    static void reset_statistics(uint8_t channel = 0xFF);
    
    /**
     * @brief Enable/disable DMA power management
     * @param enable Enable power management
     */
    static void set_power_management(bool enable);
    
    /**
     * @brief Check if DMA subsystem is initialized
     * @return true if initialized
     */
    static bool is_initialized();
    
    /**
     * @brief Get optimal transfer size for best performance
     * @param src_addr Source address
     * @param dst_addr Destination address
     * @param total_size Total transfer size
     * @return Recommended chunk size for transfers
     */
    static size_t get_optimal_transfer_size(const void* src_addr, const void* dst_addr, size_t total_size);

private:
    Dma() = delete;
    ~Dma() = delete;
    Dma(const Dma&) = delete;
    Dma& operator=(const Dma&) = delete;
    
    /**
     * @brief Find available DMA channel
     * @return Channel number or 0xFF if none available
     */
    static uint8_t find_available_channel();
    
    /**
     * @brief Initialize hardware-specific DMA controller
     * @return true if successful
     */
    static bool init_hardware();
    
    /**
     * @brief Configure hardware DMA channel
     * @param channel Channel number
     * @param config Configuration
     * @return true if successful
     */
    static bool configure_hardware_channel(uint8_t channel, const DmaConfig& config);
    
    /**
     * @brief Process DMA interrupt (called from ISR)
     * @param channel Channel that generated interrupt
     */
    static void handle_interrupt(uint8_t channel);
};

/**
 * @brief RAII DMA channel allocator
 * 
 * Automatically allocates DMA channel on construction and releases on destruction
 */
class DmaChannel {
public:
    /**
     * @brief Allocate DMA channel with configuration
     * @param config DMA configuration parameters
     */
    explicit DmaChannel(const DmaConfig& config = {});
    
    /**
     * @brief Release DMA channel
     */
    ~DmaChannel();
    
    /**
     * @brief Get channel number
     * @return Channel number, or 0xFF if allocation failed
     */
    uint8_t channel() const { return channel_; }
    
    /**
     * @brief Check if channel allocation was successful
     * @return true if valid channel allocated
     */
    bool is_valid() const { return channel_ != 0xFF; }
    
    /**
     * @brief Synchronous copy using this channel
     * @param dest Destination buffer
     * @param src Source buffer
     * @param size Number of bytes to copy
     * @return true if successful
     */
    bool copy(void* dest, const void* src, size_t size);
    
    /**
     * @brief Asynchronous copy using this channel
     * @param dest Destination buffer
     * @param src Source buffer
     * @param size Number of bytes to copy
     * @param callback Completion callback
     * @param user_data User callback data
     * @return true if transfer started
     */
    bool copy_async(void* dest, const void* src, size_t size, 
                   DmaCallback callback = nullptr, void* user_data = nullptr);
    
    /**
     * @brief Wait for current transfer completion
     * @param timeout_ms Timeout in milliseconds
     * @return Transfer status
     */
    DmaStatus wait(uint32_t timeout_ms = 5000);
    
    /**
     * @brief Get current transfer status
     * @return Transfer status
     */
    DmaStatus status() const;

private:
    uint8_t channel_;
    
    // Non-copyable
    DmaChannel(const DmaChannel&) = delete;
    DmaChannel& operator=(const DmaChannel&) = delete;
};

} // namespace cortex_m
} // namespace platform
} // namespace cmx