// cmx_nios2_dma.cpp
// CMatrix Framework Implementation
/**
 * @file cmx_nios2_dma.cpp
 * @brief DMA-based memory transfer utility implementation for Nios II
 * @author CMatrix Development Team
 * @version 1.0
 */

#include "cmx_nios2_dma.hpp"
#include "cmx_nios2_port.hpp"

#include <system.h>
#include <string.h>
#include <stdio.h>

// Conditional includes for DMA hardware
#ifdef SGDMA_0_BASE
#include <altera_avalon_sgdma.h>
#include <altera_avalon_sgdma_descriptor.h>
#include <altera_avalon_sgdma_regs.h>
#endif

#ifdef MSGDMA_0_BASE
#include <altera_msgdma.h>
#endif

namespace cmx {
namespace platform {
namespace nios2 {

// =============================================================================
// Static Variables and Structures
// =============================================================================

struct DmaChannel {
    uint8_t id;
    bool allocated;
    bool busy;
    DmaPriority priority;
    uint32_t current_transfer_id;
    uint64_t last_activity_us;
};

struct AsyncTransfer {
    DmaTransfer transfer;
    DmaCallback callback;
    void* user_data;
    bool active;
};

static bool g_dma_initialized = false;
static bool g_dma_available = false;
static bool g_debug_enabled = false;
static DmaStats g_dma_stats = {0};
static DmaChannel g_channels[CMX_DMA_MAX_CHANNELS] = {0};
static AsyncTransfer g_async_transfers[CMX_DMA_MAX_ASYNC_TRANSFERS] = {0};
static uint32_t g_next_transfer_id = 1;

#ifdef SGDMA_0_BASE
static alt_sgdma_dev* g_sgdma_device = nullptr;
static alt_sgdma_descriptor g_sgdma_descriptors[CMX_DMA_MAX_CHANNELS] __attribute__((aligned(32)));
#endif

// =============================================================================
// Internal Helper Functions
// =============================================================================

/**
 * @brief Check if address is aligned for DMA
 */
static bool is_address_aligned(const void* addr) {
    uintptr_t ptr = reinterpret_cast<uintptr_t>(addr);
    return (ptr % CMX_DMA_DESCRIPTOR_ALIGNMENT) == 0;
}

/**
 * @brief Get next available transfer ID
 */
static uint32_t get_next_transfer_id() {
    uint32_t id = g_next_transfer_id++;
    if (g_next_transfer_id == 0) {
        g_next_transfer_id = 1; // Avoid ID 0
    }
    return id;
}

/**
 * @brief Find async transfer by ID
 */
static AsyncTransfer* find_async_transfer(uint32_t transfer_id) {
    for (auto& transfer : g_async_transfers) {
        if (transfer.active && transfer.transfer.transfer_id == transfer_id) {
            return &transfer;
        }
    }
    return nullptr;
}

/**
 * @brief Allocate async transfer slot
 */
static AsyncTransfer* allocate_async_transfer() {
    for (auto& transfer : g_async_transfers) {
        if (!transfer.active) {
            memset(&transfer, 0, sizeof(transfer));
            transfer.active = true;
            return &transfer;
        }
    }
    return nullptr;
}

/**
 * @brief Log DMA debug message
 */
static void dma_debug_log(const char* message) {
    if (g_debug_enabled) {
        cmx_log(LogLevel::DEBUG, message);
    }
}

#ifdef SGDMA_0_BASE
/**
 * @brief SGDMA transfer completion callback
 */
static void sgdma_callback(void* context) {
    uint32_t transfer_id = reinterpret_cast<uintptr_t>(context);
    AsyncTransfer* async_transfer = find_async_transfer(transfer_id);
    
    if (async_transfer) {
        async_transfer->transfer.status = DmaStatus::COMPLETE;
        async_transfer->transfer.end_time_us = cmx_get_timestamp_us();
        
        // Update statistics
        g_dma_stats.successful_transfers++;
        g_dma_stats.total_time_us += 
            async_transfer->transfer.end_time_us - async_transfer->transfer.start_time_us;
        
        // Call user callback if provided
        if (async_transfer->callback) {
            async_transfer->callback(&async_transfer->transfer, async_transfer->user_data);
        }
        
        // Release channel
        if (async_transfer->transfer.config.channel_id < CMX_DMA_MAX_CHANNELS) {
            g_channels[async_transfer->transfer.config.channel_id].busy = false;
            g_channels[async_transfer->transfer.config.channel_id].current_transfer_id = 0;
        }
        
        async_transfer->active = false;
    }
}
#endif

/**
 * @brief Perform memory copy using optimized method
 */
static bool optimized_memcpy(void* dst, const void* src, size_t size) {
    // Use DMA for larger transfers if available
    if (g_dma_available && size >= CMX_DMA_MIN_TRANSFER_SIZE) {
        return cmx_dma_transfer(dst, src, size);
    }
    
    // Fallback to standard memcpy
    memcpy(dst, src, size);
    return true;
}

// =============================================================================
// Core DMA Functions
// =============================================================================

bool cmx_dma_init() {
    if (g_dma_initialized) {
        return true;
    }
    
    // Initialize channel array
    for (size_t i = 0; i < CMX_DMA_MAX_CHANNELS; i++) {
        g_channels[i].id = static_cast<uint8_t>(i);
        g_channels[i].allocated = false;
        g_channels[i].busy = false;
        g_channels[i].priority = DmaPriority::NORMAL;
        g_channels[i].current_transfer_id = 0;
        g_channels[i].last_activity_us = 0;
    }
    
    // Initialize async transfer array
    for (auto& transfer : g_async_transfers) {
        transfer.active = false;
    }
    
    // Initialize statistics
    memset(&g_dma_stats, 0, sizeof(g_dma_stats));
    
#ifdef SGDMA_0_BASE
    // Initialize SGDMA if available
    g_sgdma_device = alt_avalon_sgdma_open("/dev/sgdma_0");
    if (g_sgdma_device) {
        g_dma_available = true;
        dma_debug_log("SGDMA initialized successfully");
    } else {
        dma_debug_log("SGDMA initialization failed");
    }
#endif

#ifdef MSGDMA_0_BASE
    // Initialize mSGDMA if available
    alt_msgdma_dev* msgdma_dev = alt_msgdma_open("/dev/msgdma_0");
    if (msgdma_dev) {
        g_dma_available = true;
        dma_debug_log("mSGDMA initialized successfully");
    } else {
        dma_debug_log("mSGDMA initialization failed");
    }
#endif

    if (!g_dma_available) {
        dma_debug_log("No DMA hardware found, using memcpy fallback");
    }
    
    g_dma_initialized = true;
    return true;
}

void cmx_dma_cleanup() {
    if (!g_dma_initialized) {
        return;
    }
    
    // Cancel all ongoing transfers
    for (auto& transfer : g_async_transfers) {
        if (transfer.active) {
            transfer.transfer.status = DmaStatus::CANCELLED;
            transfer.active = false;
        }
    }
    
    // Release all channels
    for (auto& channel : g_channels) {
        channel.allocated = false;
        channel.busy = false;
    }
    
#ifdef SGDMA_0_BASE
    if (g_sgdma_device) {
        // SGDMA cleanup is handled by HAL
        g_sgdma_device = nullptr;
    }
#endif

    g_dma_available = false;
    g_dma_initialized = false;
    
    dma_debug_log("DMA cleanup completed");
}

bool cmx_dma_available() {
    return g_dma_available;
}

bool cmx_dma_transfer(void* dst, const void* src, size_t size) {
    if (!g_dma_initialized) {
        return optimized_memcpy(dst, src, size);
    }
    
    // Validate parameters
    if (!dst || !src || size == 0) {
        return false;
    }
    
    // Check size limits
    if (size > cmx_dma_get_max_transfer_size()) {
        return false;
    }
    
    // Update statistics
    g_dma_stats.total_transfers++;
    g_dma_stats.total_bytes += size;
    
    uint64_t start_time = cmx_get_timestamp_us();
    
#ifdef SGDMA_0_BASE
    if (g_sgdma_device && g_dma_available) {
        // Use SGDMA for transfer
        alt_sgdma_descriptor* desc = &g_sgdma_descriptors[0];
        
        alt_avalon_sgdma_construct_mem_to_mem_dma_descriptor(
            desc,
            nullptr, // Next descriptor
            const_cast<void*>(src),
            dst,
            size,
            0, // Don't generate interrupt
            0, // Don't generate interrupt
            0, // Don't generate interrupt
            0  // Control flags
        );
        
        int result = alt_avalon_sgdma_do_sync_transfer(g_sgdma_device, desc);
        
        uint64_t end_time = cmx_get_timestamp_us();
        uint64_t duration = end_time - start_time;
        
        g_dma_stats.total_time_us += duration;
        
        if (result == 0) {
            g_dma_stats.successful_transfers++;
            
            // Update max transfer size
            if (size > g_dma_stats.max_transfer_size) {
                g_dma_stats.max_transfer_size = static_cast<uint32_t>(size);
            }
            
            return true;
        } else {
            g_dma_stats.failed_transfers++;
            g_dma_stats.error_count[0]++; // Generic error
            return false;
        }
    }
#endif

    // Fallback to memcpy
    bool result = optimized_memcpy(dst, src, size);
    
    uint64_t end_time = cmx_get_timestamp_us();
    g_dma_stats.total_time_us += (end_time - start_time);
    
    if (result) {
        g_dma_stats.successful_transfers++;
    } else {
        g_dma_stats.failed_transfers++;
    }
    
    return result;
}

bool cmx_dma_transfer_config(void* dst, const void* src, size_t size, 
                            const DmaChannelConfig& config) {
    // For blocking transfers, use the simpler interface
    // Configuration is mainly useful for async transfers
    return cmx_dma_transfer(dst, src, size);
}

uint32_t cmx_dma_transfer_async(void* dst, const void* src, size_t size,
                               const DmaChannelConfig& config,
                               DmaCallback callback, void* user_data) {
    if (!g_dma_initialized || !g_dma_available) {
        return 0; // Async not supported without DMA
    }
    
    // Allocate async transfer slot
    AsyncTransfer* async_transfer = allocate_async_transfer();
    if (!async_transfer) {
        return 0; // No slots available
    }
    
    // Setup transfer descriptor
    async_transfer->transfer.destination = dst;
    async_transfer->transfer.source = src;
    async_transfer->transfer.size = size;
    async_transfer->transfer.config = config;
    async_transfer->transfer.status = DmaStatus::BUSY;
    async_transfer->transfer.start_time_us = cmx_get_timestamp_us();
    async_transfer->transfer.transfer_id = get_next_transfer_id();
    async_transfer->callback = callback;
    async_transfer->user_data = user_data;
    
#ifdef SGDMA_0_BASE
    if (g_sgdma_device) {
        uint8_t channel_id = config.channel_id;
        if (channel_id >= CMX_DMA_MAX_CHANNELS) {
            channel_id = 0; // Use default channel
        }
        
        // Mark channel as busy
        g_channels[channel_id].busy = true;
        g_channels[channel_id].current_transfer_id = async_transfer->transfer.transfer_id;
        
        alt_sgdma_descriptor* desc = &g_sgdma_descriptors[channel_id];
        
        alt_avalon_sgdma_construct_mem_to_mem_dma_descriptor(
            desc,
            nullptr,
            const_cast<void*>(src),
            dst,
            size,
            0,
            1, // Generate interrupt on completion
            0,
            0
        );
        
        // Start async transfer with callback
        int result = alt_avalon_sgdma_do_async_transfer(
            g_sgdma_device, 
            desc,
            sgdma_callback,
            reinterpret_cast<void*>(static_cast<uintptr_t>(async_transfer->transfer.transfer_id))
        );
        
        if (result == 0) {
            g_dma_stats.total_transfers++;
            return async_transfer->transfer.transfer_id;
        } else {
            // Failed to start transfer
            g_channels[channel_id].busy = false;
            g_channels[channel_id].current_transfer_id = 0;
            async_transfer->active = false;
            g_dma_stats.failed_transfers++;
            return 0;
        }
    }
#endif

    // No async DMA available, cleanup and return failure
    async_transfer->active = false;
    return 0;
}

// =============================================================================
// DMA Channel Management
// =============================================================================

uint8_t cmx_dma_allocate_channel(DmaPriority priority) {
    if (!g_dma_initialized) {
        return 0xFF;
    }
    
    // Find first available channel
    for (size_t i = 0; i < CMX_DMA_MAX_CHANNELS; i++) {
        if (!g_channels[i].allocated) {
            g_channels[i].allocated = true;
            g_channels[i].priority = priority;
            g_channels[i].last_activity_us = cmx_get_timestamp_us();
            return g_channels[i].id;
        }
    }
    
    return 0xFF; // No channels available
}

bool cmx_dma_release_channel(uint8_t channel_id) {
    if (channel_id >= CMX_DMA_MAX_CHANNELS) {
        return false;
    }
    
    // Cancel any ongoing transfer on this channel
    if (g_channels[channel_id].busy && g_channels[channel_id].current_transfer_id != 0) {
        cmx_dma_cancel_transfer(g_channels[channel_id].current_transfer_id);
    }
    
    g_channels[channel_id].allocated = false;
    g_channels[channel_id].busy = false;
    g_channels[channel_id].current_transfer_id = 0;
    
    return true;
}

bool cmx_dma_is_channel_available(uint8_t channel_id) {
    if (channel_id >= CMX_DMA_MAX_CHANNELS) {
        return false;
    }
    
    return !g_channels[channel_id].allocated;
}

bool cmx_dma_wait_channel_idle(uint8_t channel_id, uint32_t timeout_ms) {
    if (channel_id >= CMX_DMA_MAX_CHANNELS) {
        return false;
    }
    
    uint64_t start_time = cmx_get_timestamp_us();
    uint64_t timeout_us = timeout_ms * 1000ULL;
    
    while (g_channels[channel_id].busy) {
        if (timeout_ms > 0) {
            uint64_t elapsed = cmx_get_timestamp_us() - start_time;
            if (elapsed >= timeout_us) {
                return false; // Timeout
            }
        }
        cmx_yield();
    }
    
    return true;
}

// =============================================================================
// Transfer Status and Control
// =============================================================================

DmaStatus cmx_dma_get_transfer_status(uint32_t transfer_id) {
    AsyncTransfer* transfer = find_async_transfer(transfer_id);
    if (transfer) {
        return transfer->transfer.status;
    }
    return DmaStatus::ERROR;
}

bool cmx_dma_cancel_transfer(uint32_t transfer_id) {
    AsyncTransfer* transfer = find_async_transfer(transfer_id);
    if (!transfer) {
        return false;
    }
    
    // Mark as cancelled
    transfer->transfer.status = DmaStatus::CANCELLED;
    transfer->transfer.end_time_us = cmx_get_timestamp_us();
    
    // Release channel if allocated
    uint8_t channel_id = transfer->transfer.config.channel_id;
    if (channel_id < CMX_DMA_MAX_CHANNELS) {
        g_channels[channel_id].busy = false;
        g_channels[channel_id].current_transfer_id = 0;
    }
    
#ifdef SGDMA_0_BASE
    // Stop SGDMA transfer if possible
    if (g_sgdma_device) {
        // Note: SGDMA may not support cancellation mid-transfer
        // This is hardware-dependent
    }
#endif

    transfer->active = false;
    g_dma_stats.cancelled_transfers++;
    
    return true;
}

bool cmx_dma_wait_transfer_complete(uint32_t transfer_id, uint32_t timeout_ms) {
    uint64_t start_time = cmx_get_timestamp_us();
    uint64_t timeout_us = timeout_ms * 1000ULL;
    
    while (true) {
        AsyncTransfer* transfer = find_async_transfer(transfer_id);
        if (!transfer) {
            return false; // Transfer not found
        }
        
        if (transfer->transfer.status == DmaStatus::COMPLETE) {
            return true;
        }
        
        if (transfer->transfer.status == DmaStatus::ERROR || 
            transfer->transfer.status == DmaStatus::CANCELLED) {
            return false;
        }
        
        if (timeout_ms > 0) {
            uint64_t elapsed = cmx_get_timestamp_us() - start_time;
            if (elapsed >= timeout_us) {
                return false; // Timeout
            }
        }
        
        cmx_yield();
    }
}

size_t cmx_dma_get_transfer_progress(uint32_t transfer_id) {
    AsyncTransfer* transfer = find_async_transfer(transfer_id);
    if (!transfer) {
        return 0;
    }
    
    // For Nios II DMA, progress tracking is limited
    // Return full size if complete, 0 if not started, size/2 if busy
    switch (transfer->transfer.status) {
        case DmaStatus::COMPLETE:
            return transfer->transfer.size;
        case DmaStatus::BUSY:
            return transfer->transfer.size / 2; // Approximate
        default:
            return 0;
    }
}

// =============================================================================
// DMA Configuration and Capabilities
// =============================================================================

uint8_t cmx_dma_get_channel_count() {
    return g_dma_available ? CMX_DMA_MAX_CHANNELS : 0;
}

size_t cmx_dma_get_max_transfer_size() {
    if (g_dma_available) {
#ifdef SGDMA_0_BASE
        return 0xFFFF; // 64KB - 1 (typical SGDMA limit)
#else
        return 0x10000; // 64KB
#endif
    }
    return SIZE_MAX; // No limit for memcpy fallback
}

size_t cmx_dma_get_alignment_requirement() {
    return CMX_DMA_DESCRIPTOR_ALIGNMENT;
}

bool cmx_dma_is_address_valid(const void* addr) {
    if (!addr) {
        return false;
    }
    
    // Check alignment
    if (!is_address_aligned(addr)) {
        return false;
    }
    
    // Check if address is in valid memory range
    uintptr_t ptr = reinterpret_cast<uintptr_t>(addr);
    
#ifdef ONCHIP_MEMORY2_0_BASE
    uintptr_t mem_base = ONCHIP_MEMORY2_0_BASE;
    uintptr_t mem_end = mem_base + ONCHIP_MEMORY2_0_SPAN;
    
    if (ptr >= mem_base && ptr < mem_end) {
        return true;
    }
#endif

#ifdef SDRAM_BASE
    uintptr_t sdram_base = SDRAM_BASE;
    uintptr_t sdram_end = sdram_base + SDRAM_SPAN;
    
    if (ptr >= sdram_base && ptr < sdram_end) {
        return true;
    }
#endif

    // Add other memory regions as needed
    return true; // Default to valid if we can't determine
}

uint8_t cmx_dma_get_supported_burst_sizes() {
    // Return bitmask of supported burst sizes (1, 2, 4, 8 bytes)
    return 0x0F; // Support 1, 2, 4, 8 byte bursts
}

// =============================================================================
// Statistics and Monitoring
// =============================================================================

const DmaStats& cmx_dma_get_stats() {
    // Update dynamic statistics
    if (g_dma_stats.total_time_us > 0 && g_dma_stats.total_bytes > 0) {
        // Calculate average throughput in MB/s
        uint64_t mbytes = g_dma_stats.total_bytes / (1024 * 1024);
        uint64_t seconds = g_dma_stats.total_time_us / 1000000;
        
        if (seconds > 0) {
            g_dma_stats.avg_throughput_mbps = static_cast<uint32_t>(mbytes / seconds);
        }
    }
    
    return g_dma_stats;
}

void cmx_dma_reset_stats() {
    memset(&g_dma_stats, 0, sizeof(g_dma_stats));
}

uint32_t cmx_dma_get_current_throughput() {
    return g_dma_stats.avg_throughput_mbps;
}

uint8_t cmx_dma_get_channel_utilization(uint8_t channel_id) {
    if (channel_id >= CMX_DMA_MAX_CHANNELS) {
        return 0;
    }
    
    // Simple utilization: 100% if busy, 0% if idle
    return g_channels[channel_id].busy ? 100 : 0;
}

// =============================================================================
// Debug and Testing Functions
// =============================================================================

bool cmx_dma_self_test() {
    const size_t test_size = 1024;
    static uint8_t test_src[test_size];
    static uint8_t test_dst[test_size];
    
    // Initialize test pattern
    for (size_t i = 0; i < test_size; i++) {
        test_src[i] = static_cast<uint8_t>(i & 0xFF);
        test_dst[i] = 0;
    }
    
    // Perform test transfer
    bool result = cmx_dma_transfer(test_dst, test_src, test_size);
    if (!result) {
        return false;
    }
    
    // Verify data
    for (size_t i = 0; i < test_size; i++) {
        if (test_dst[i] != test_src[i]) {
            return false;
        }
    }
    
    dma_debug_log("DMA self-test passed");
    return true;
}

uint32_t cmx_dma_benchmark(size_t size, uint32_t iterations) {
    if (iterations == 0) {
        return 0;
    }
    
    // Allocate test buffers
    uint8_t* src_buffer = new uint8_t[size];
    uint8_t* dst_buffer = new uint8_t[size];
    
    if (!src_buffer || !dst_buffer) {
        delete[] src_buffer;
        delete[] dst_buffer;
        return 0;
    }
    
    // Initialize source buffer
    for (size_t i = 0; i < size; i++) {
        src_buffer[i] = static_cast<uint8_t>(i & 0xFF);
    }
    
    uint64_t total_time = 0;
    uint32_t successful_transfers = 0;
    
    for (uint32_t i = 0; i < iterations; i++) {
        uint64_t start_time = cmx_get_timestamp_us();
        
        bool result = cmx_dma_transfer(dst_buffer, src_buffer, size);
        
        uint64_t end_time = cmx_get_timestamp_us();
        
        if (result) {
            total_time += (end_time - start_time);
            successful_transfers++;
        }
    }
    
    delete[] src_buffer;
    delete[] dst_buffer;
    
    if (successful_transfers == 0 || total_time == 0) {
        return 0;
    }
    
    // Calculate throughput in MB/s
    uint64_t total_bytes = static_cast<uint64_t>(size) * successful_transfers;
    uint64_t avg_time_us = total_time / successful_transfers;
    
    if (avg_time_us == 0) {
        return 0;
    }
    
    uint64_t throughput = (total_bytes * 1000000ULL) / (avg_time_us * 1024 * 1024);
    return static_cast<uint32_t>(throughput);
}

bool cmx_dma_pattern_test(uint32_t pattern, size_t size) {
    uint8_t* test_buffer = new uint8_t[size];
    uint8_t* verify_buffer = new uint8_t[size];
    
    if (!test_buffer || !verify_buffer) {
        delete[] test_buffer;
        delete[] verify_buffer;
        return false;
    }
    
    // Fill source buffer with pattern
    uint8_t* pattern_bytes = reinterpret_cast<uint8_t*>(&pattern);
    for (size_t i = 0; i < size; i++) {
        test_buffer[i] = pattern_bytes[i % sizeof(pattern)];
    }
    
    // Clear destination
    memset(verify_buffer, 0, size);
    
    // Perform DMA transfer
    bool result = cmx_dma_transfer(verify_buffer, test_buffer, size);
    
    if (result) {
        // Verify pattern
        result = (memcmp(test_buffer, verify_buffer, size) == 0);
    }
    
    delete[] test_buffer;
    delete[] verify_buffer;
    
    return result;
}

void cmx_dma_set_debug_enabled(bool enable) {
    g_debug_enabled = enable;
    
    if (enable) {
        dma_debug_log("DMA debug logging enabled");
    }
}

// =============================================================================
// Memory Copy Fallback Functions
// =============================================================================

bool cmx_memcpy_optimized(void* dst, const void* src, size_t size) {
    return optimized_memcpy(dst, src, size);
}

bool cmx_memset_dma(void* dst, uint8_t value, size_t size) {
    if (g_dma_available && size >= CMX_DMA_MIN_TRANSFER_SIZE) {
        // For DMA memset, we'd need a source buffer filled with the pattern
        // This is often not worth it for memset operations
    }
    
    // Use standard memset
    memset(dst, value, size);
    return true;
}

int cmx_memcmp_dma(const void* ptr1, const void* ptr2, size_t size) {
    // DMA doesn't help with comparison operations
    return memcmp(ptr1, ptr2, size);
}

} // namespace nios2
} // namespace platform
} // namespace cmx