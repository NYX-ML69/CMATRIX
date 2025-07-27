#pragma once

#include <cstdint>
#include <cstddef>
#include <functional>

// Forward declarations
struct gdma_channel_t;
typedef struct gdma_channel_t* gdma_channel_handle_t;

namespace cmx {
namespace platform {
namespace esp32 {

/**
 * @brief DMA operation status codes
 */
enum class DMAStatus : uint8_t {
    SUCCESS = 0,
    ERROR_INIT_FAILED,
    ERROR_INVALID_CHANNEL,
    ERROR_INVALID_PARAM,
    ERROR_TRANSFER_FAILED,
    ERROR_TIMEOUT,
    ERROR_BUSY,
    ERROR_NOT_ALIGNED
};

/**
 * @brief DMA transfer modes
 */
enum class DMAMode : uint8_t {
    BLOCKING = 0,        // Wait for completion
    NON_BLOCKING,        // Return immediately
    CALLBACK             // Call callback on completion
};

/**
 * @brief DMA transfer direction
 */
enum class DMADirection : uint8_t {
    MEM_TO_MEM = 0,     // Memory to memory
    MEM_TO_PERIPH,      // Memory to peripheral
    PERIPH_TO_MEM,      // Peripheral to memory
    PERIPH_TO_PERIPH    // Peripheral to peripheral
};

/**
 * @brief DMA alignment requirements
 */
struct DMAAlignment {
    static constexpr size_t ADDRESS_ALIGN = 4;   // 4-byte address alignment
    static constexpr size_t SIZE_ALIGN = 4;      // 4-byte size alignment
    static constexpr size_t CACHE_LINE = 32;     // Cache line size
};

/**
 * @brief DMA transfer descriptor
 */
struct DMATransfer {
    void* src;                    // Source address
    void* dst;                    // Destination address
    size_t size;                  // Transfer size in bytes
    DMADirection direction;       // Transfer direction
    DMAMode mode;                // Transfer mode
    uint32_t timeout_ms;         // Timeout for blocking transfers
    
    // Callback for async transfers
    std::function<void(DMAStatus status)> callback;
    
    // Optional peripheral configuration
    uint32_t periph_addr;        // Peripheral address
    uint8_t periph_width;        // Peripheral data width (1, 2, 4 bytes)
    bool periph_inc;             // Increment peripheral address
};

/**
 * @brief DMA channel configuration
 */
struct DMAConfig {
    uint8_t channel_id;          // DMA channel ID (0-7)
    uint32_t priority;           // Channel priority (0-3, higher = more priority)
    bool enable_interrupt;       // Enable completion interrupt
    bool auto_reload;            // Auto-reload for circular transfers
    size_t max_transfer_size;    // Maximum single transfer size
    
    // Memory configuration
    bool src_cache_enable;       // Enable source cache coherency
    bool dst_cache_enable;       // Enable destination cache coherency
    uint8_t src_burst_size;      // Source burst size (1, 4, 8, 16)
    uint8_t dst_burst_size;      // Destination burst size (1, 4, 8, 16)
};

/**
 * @brief Default DMA configuration
 */
constexpr DMAConfig DEFAULT_DMA_CONFIG = {
    .channel_id = 0,
    .priority = 2,
    .enable_interrupt = true,
    .auto_reload = false,
    .max_transfer_size = 65536,  // 64KB
    .src_cache_enable = true,
    .dst_cache_enable = true,
    .src_burst_size = 4,
    .dst_burst_size = 4
};

/**
 * @brief DMA channel statistics
 */
struct DMAStats {
    uint32_t total_transfers;    // Total number of transfers
    uint32_t successful_transfers; // Successful transfers
    uint32_t failed_transfers;   // Failed transfers
    uint32_t timeout_transfers;  // Timed out transfers
    uint64_t total_bytes;        // Total bytes transferred
    uint32_t avg_transfer_time_us; // Average transfer time
    uint32_t max_transfer_time_us; // Maximum transfer time
};

/**
 * @brief ESP32 DMA Controller
 */
class ESP32DMA {
public:
    /**
     * @brief Initialize DMA subsystem
     * @return DMAStatus::SUCCESS on success
     */
    static DMAStatus initialize();
    
    /**
     * @brief Shutdown DMA subsystem
     * @return DMAStatus::SUCCESS on success
     */
    static DMAStatus shutdown();
    
    /**
     * @brief Check if DMA is initialized
     * @return true if initialized
     */
    static bool is_initialized();

private:
    static bool initialized_;
};

/**
 * @brief DMA Channel management
 */
class DMAChannel {
public:
    /**
     * @brief Create and configure DMA channel
     * @param config Channel configuration
     */
    explicit DMAChannel(const DMAConfig& config = DEFAULT_DMA_CONFIG);
    
    /**
     * @brief Destructor - releases channel resources
     */
    ~DMAChannel();
    
    // Disable copy operations
    DMAChannel(const DMAChannel&) = delete;
    DMAChannel& operator=(const DMAChannel&) = delete;
    
    // Enable move operations
    DMAChannel(DMAChannel&& other) noexcept;
    DMAChannel& operator=(DMAChannel&& other) noexcept;
    
    /**
     * @brief Check if channel is valid and ready
     * @return true if channel is ready
     */
    bool is_ready() const;
    
    /**
     * @brief Get channel configuration
     * @return Current channel configuration
     */
    const DMAConfig& get_config() const;
    
    /**
     * @brief Perform synchronous memory-to-memory transfer
     * @param dst Destination buffer
     * @param src Source buffer
     * @param size Transfer size in bytes
     * @param timeout_ms Timeout in milliseconds
     * @return DMAStatus::SUCCESS on success
     */
    DMAStatus transfer_sync(void* dst, const void* src, size_t size, 
                           uint32_t timeout_ms = 1000);
    
    /**
     * @brief Perform asynchronous memory-to-memory transfer
     * @param dst Destination buffer
     * @param src Source buffer
     * @param size Transfer size in bytes
     * @param callback Completion callback
     * @return DMAStatus::SUCCESS if transfer started
     */
    DMAStatus transfer_async(void* dst, const void* src, size_t size,
                            std::function<void(DMAStatus)> callback = nullptr);
    
    /**
     * @brief Perform complex DMA transfer
     * @param transfer Transfer descriptor
     * @return DMAStatus::SUCCESS on success or if started (async)
     */
    DMAStatus transfer(const DMATransfer& transfer);
    
    /**
     * @brief Wait for current transfer to complete
     * @param timeout_ms Timeout in milliseconds
     * @return DMAStatus::SUCCESS on completion
     */
    DMAStatus wait_completion(uint32_t timeout_ms = 1000);
    
    /**
     * @brief Cancel current transfer
     * @return DMAStatus::SUCCESS if cancelled
     */
    DMAStatus cancel_transfer();
    
    /**
     * @brief Check if transfer is in progress
     * @return true if transfer is active
     */
    bool is_transfer_active() const;
    
    /**
     * @brief Get transfer progress (0.0 to 1.0)
     * @return Progress percentage
     */
    float get_transfer_progress() const;
    
    /**
     * @brief Get channel statistics
     * @return Channel statistics
     */
    DMAStats get_stats() const;
    
    /**
     * @brief Reset channel statistics
     */
    void reset_stats();

private:
    DMAConfig config_;
    gdma_channel_handle_t channel_handle_;
    bool initialized_;
    mutable DMAStats stats_;
    
    // Transfer state
    volatile bool transfer_active_;
    volatile size_t bytes_transferred_;
    volatile size_t total_bytes_;
    uint64_t transfer_start_time_;
    std::function<void(DMAStatus)> completion_callback_;
    
    // Internal methods
    DMAStatus configure_channel();
    DMAStatus start_transfer(const DMATransfer& transfer);
    void handle_transfer_completion(DMAStatus status);
    bool validate_transfer_params(const DMATransfer& transfer) const;
    void update_stats(DMAStatus status, uint32_t transfer_time_us);
    
    // Static callback for ESP-IDF
    static bool transfer_done_callback(gdma_channel_handle_t dma_chan, 
                                     gdma_event_data_t* event_data, 
                                     void* user_data);
};

/**
 * @brief High-level DMA utilities for common operations
 */
namespace dma_utils {
    /**
     * @brief Check if address is properly aligned for DMA
     * @param addr Address to check
     * @param alignment Required alignment (default: 4 bytes)
     * @return true if aligned
     */
    bool is_address_aligned(const void* addr, size_t alignment = DMAAlignment::ADDRESS_ALIGN);
    
    /**
     * @brief Check if size is properly aligned for DMA
     * @param size Size to check
     * @param alignment Required alignment (default: 4 bytes)
     * @return true if aligned
     */
    bool is_size_aligned(size_t size, size_t alignment = DMAAlignment::SIZE_ALIGN);
    
    /**
     * @brief Align address up to specified boundary
     * @param addr Address to align
     * @param alignment Alignment boundary
     * @return Aligned address
     */
    void* align_address_up(void* addr, size_t alignment);
    
    /**
     * @brief Align size up to specified boundary
     * @param size Size to align
     * @param alignment Alignment boundary
     * @return Aligned size
     */
    size_t align_size_up(size_t size, size_t alignment);
    
    /**
     * @brief Allocate DMA-compatible memory buffer
     * @param size Buffer size in bytes
     * @param alignment Memory alignment (default: cache line)
     * @param use_psram Use PSRAM if available
     * @return Allocated buffer or nullptr on failure
     */
    void* allocate_dma_buffer(size_t size, 
                             size_t alignment = DMAAlignment::CACHE_LINE,
                             bool use_psram = false);
    
    /**
     * @brief Free DMA-compatible memory buffer
     * @param buffer Buffer to free
     */
    void free_dma_buffer(void* buffer);
    
    /**
     * @brief Copy memory using optimal method (DMA or memcpy)
     * @param dst Destination buffer
     * @param src Source buffer
     * @param size Copy size
     * @param force_dma Force DMA usage even for small transfers
     * @return DMAStatus::SUCCESS on success
     */
    DMAStatus optimal_memcpy(void* dst, const void* src, size_t size, 
                           bool force_dma = false);
    
    /**
     * @brief Set memory using optimal method (DMA or memset)
     * @param dst Destination buffer
     * @param value Value to set
     * @param size Size to set
     * @param force_dma Force DMA usage even for small transfers
     * @return DMAStatus::SUCCESS on success
     */
    DMAStatus optimal_memset(void* dst, uint8_t value, size_t size, 
                           bool force_dma = false);
    
    /**
     * @brief Flush cache for DMA coherency
     * @param addr Buffer address
     * @param size Buffer size
     */
    void flush_cache(const void* addr, size_t size);
    
    /**
     * @brief Invalidate cache for DMA coherency
     * @param addr Buffer address
     * @param size Buffer size
     */
    void invalidate_cache(const void* addr, size_t size);
    
    /**
     * @brief Get optimal DMA transfer size for given buffer
     * @param size Requested size
     * @return Optimal transfer size
     */
    size_t get_optimal_transfer_size(size_t size);
    
    /**
     * @brief Check if memory region is DMA accessible
     * @param addr Memory address
     * @param size Memory size
     * @return true if accessible by DMA
     */
    bool is_dma_accessible(const void* addr, size_t size);
}

/**
 * @brief DMA memory pool for efficient buffer management
 */
class DMAMemoryPool {
public:
    /**
     * @brief Create DMA memory pool
     * @param pool_size Total pool size in bytes
     * @param block_size Individual block size
     * @param use_psram Use PSRAM for pool storage
     */
    DMAMemoryPool(size_t pool_size, size_t block_size, bool use_psram = false);
    
    /**
     * @brief Destructor - releases pool memory
     */
    ~DMAMemoryPool();
    
    // Disable copy operations
    DMAMemoryPool(const DMAMemoryPool&) = delete;
    DMAMemoryPool& operator=(const DMAMemoryPool&) = delete;
    
    /**
     * @brief Allocate block from pool
     * @return Allocated block or nullptr if pool is full
     */
    void* allocate_block();
    
    /**
     * @brief Free block back to pool
     * @param block Block to free
     * @return true if block was valid and freed
     */
    bool free_block(void* block);
    
    /**
     * @brief Get number of free blocks
     * @return Free block count
     */
    size_t get_free_blocks() const;
    
    /**
     * @brief Get total number of blocks
     * @return Total block count
     */
    size_t get_total_blocks() const;
    
    /**
     * @brief Check if address belongs to this pool
     * @param addr Address to check
     * @return true if address is from this pool
     */
    bool owns_address(const void* addr) const;
    
    /**
     * @brief Reset pool (free all allocated blocks)
     */
    void reset();

private:
    void* pool_memory_;
    size_t pool_size_;
    size_t block_size_;
    size_t total_blocks_;
    size_t free_blocks_;
    uint8_t* free_list_;     // Bitmap of free blocks
    bool use_psram_;
    
    size_t get_block_index(const void* addr) const;
    void* get_block_address(size_t index) const;
    void mark_block_used(size_t index);
    void mark_block_free(size_t index);
    bool is_block_free(size_t index) const;
};

} // namespace esp32
} // namespace platform
} // namespace cmx