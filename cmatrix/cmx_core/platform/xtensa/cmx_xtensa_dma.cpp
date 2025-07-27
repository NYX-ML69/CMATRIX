#include "cmx_xtensa_dma.hpp"
#include "cmx_xtensa_port.hpp"
#include "cmx_xtensa_timer.hpp"
#include <cstring>
#include <algorithm>

namespace cmx::platform::xtensa {

// DMA configuration constants
static constexpr uint8_t MAX_DMA_CHANNELS = 4;
static constexpr size_t DMA_ALIGNMENT = 4;
static constexpr size_t MAX_TRANSFER_SIZE = 4092; // Common Xtensa DMA limit

// DMA channel state
struct DmaChannel {
    bool active;
    DmaHandle handle;
    uint64_t start_time_us;
};

static DmaChannel g_dma_channels[MAX_DMA_CHANNELS];
static DmaStats g_dma_stats = {};
static bool g_dma_initialized = false;

// Memory regions that are DMA-accessible (platform specific)
struct DmaMemoryRegion {
    uintptr_t start;
    uintptr_t end;
};

static const DmaMemoryRegion g_dma_regions[] = {
    {0x3FFB0000, 0x3FFC0000}, // Example: Internal SRAM
    {0x3F800000, 0x3FC00000}, // Example: External PSRAM
};

void dma_init() {
    if (g_dma_initialized) {
        return;
    }
    
    // Initialize all channels as inactive
    for (int i = 0; i < MAX_DMA_CHANNELS; i++) {
        g_dma_channels[i].active = false;
        g_dma_channels[i].handle.channel = i;
        g_dma_channels[i].handle.active = false;
        g_dma_channels[i].handle.user_data = nullptr;
        g_dma_channels[i].handle.callback = nullptr;
    }
    
    // Reset statistics
    std::memset(&g_dma_stats, 0, sizeof(g_dma_stats));
    
    g_dma_initialized = true;
    cmx_log("DMA: Initialized with 4 channels");
}

static DmaChannel* find_free_channel() {
    for (int i = 0; i < MAX_DMA_CHANNELS; i++) {
        if (!g_dma_channels[i].active) {
            return &g_dma_channels[i];
        }
    }
    return nullptr;
}

bool cmx_dma_is_accessible(const void* addr, size_t size) {
    uintptr_t start = reinterpret_cast<uintptr_t>(addr);
    uintptr_t end = start + size;
    
    // Check if the memory range falls within DMA-accessible regions
    for (const auto& region : g_dma_regions) {
        if (start >= region.start && end <= region.end) {
            return true;
        }
    }
    
    return false;
}

static bool perform_dma_transfer_hw(void* dst, const void* src, size_t size) {
    // This would contain the actual hardware DMA programming
    // For demonstration, we'll simulate with timing
    
    // Check alignment
    if ((reinterpret_cast<uintptr_t>(dst) % DMA_ALIGNMENT) != 0 ||
        (reinterpret_cast<uintptr_t>(src) % DMA_ALIGNMENT) != 0) {
        return false;
    }
    
    // Check size limits
    if (size > MAX_TRANSFER_SIZE) {
        return false;
    }
    
    // Simulate hardware DMA transfer
    // In real implementation, this would:
    // 1. Configure DMA controller registers
    // 2. Set source/destination addresses
    // 3. Set transfer size and mode
    // 4. Start transfer
    // 5. Wait for completion or return for async
    
    // For now, use optimized memcpy as fallback
    std::memcpy(dst, src, size);
    
    return true;
}

bool cmx_dma_transfer(void* dst, const void* src, size_t size) {
    if (!g_dma_initialized) {
        dma_init();
    }
    
    if (!dst || !src || size == 0) {
        return false;
    }
    
    uint64_t start_time = cmx_now_us();
    bool success = false;
    
    // Check if both addresses are DMA-accessible
    if (cmx_dma_is_accessible(dst, size) && cmx_dma_is_accessible(src, size)) {
        success = perform_dma_transfer_hw(dst, src, size);
    }
    
    // Fallback to memcpy if DMA not available or failed
    if (!success) {
        std::memcpy(dst, src, size);
        success = true;
    }
    
    // Update statistics
    uint64_t transfer_time = cmx_now_us() - start_time;
    if (success) {
        g_dma_stats.transfers_completed++;
        g_dma_stats.bytes_transferred += size;
        
        // Update rolling average transfer time
        if (g_dma_stats.transfers_completed == 1) {
            g_dma_stats.avg_transfer_time_us = static_cast<uint32_t>(transfer_time);
        } else {
            g_dma_stats.avg_transfer_time_us = 
                (g_dma_stats.avg_transfer_time_us * 7 + transfer_time) / 8;
        }
    } else {
        g_dma_stats.transfers_failed++;
    }
    
    return success;
}

DmaHandle* cmx_dma_transfer_async(const DmaTransferParams& params) {
    if (!g_dma_initialized) {
        dma_init();
    }
    
    DmaChannel* channel = find_free_channel();
    if (!channel) {
        g_dma_stats.transfers_failed++;
        return nullptr;
    }
    
    // If blocking requested, just do synchronous transfer
    if (params.blocking) {
        bool success = cmx_dma_transfer(params.dst, params.src, params.size);
        if (params.callback) {
            params.callback(params.user_data, success);
        }
        return success ? &channel->handle : nullptr;
    }
    
    // Set up async transfer
    channel->active = true;
    channel->handle.active = true;
    channel->handle.user_data = params.user_data;
    channel->handle.callback = params.callback;
    channel->start_time_us = cmx_now_us();
    
    // Start the transfer (in real implementation, this would be async)
    bool success = cmx_dma_transfer(params.dst, params.src, params.size);
    
    // For this simulation, we complete immediately
    channel->active = false;
    channel->handle.active = false;
    
    if (channel->handle.callback) {
        channel->handle.callback(channel->handle.user_data, success);
    }
    
    g_dma_stats.channels_in_use = 0;
    for (int i = 0; i < MAX_DMA_CHANNELS; i++) {
        if (g_dma_channels[i].active) {
            g_dma_stats.channels_in_use++;
        }
    }
    
    return &channel->handle;
}

bool cmx_dma_is_complete(const DmaHandle* handle) {
    if (!handle) {
        return true;
    }
    
    return !handle->active;
}

bool cmx_dma_wait(DmaHandle* handle, uint32_t timeout_us) {
    if (!handle || !handle->active) {
        return true;
    }
    
    uint64_t start_time = cmx_now_us();
    
    while (handle->active) {
        if (timeout_us > 0) {
            uint64_t elapsed = cmx_now_us() - start_time;
            if (elapsed > timeout_us) {
                return false; // Timeout
            }
        }
        
        cmx_yield();
    }
    
    return true;
}

void cmx_dma_cancel(DmaHandle* handle) {
    if (!handle || handle->channel >= MAX_DMA_CHANNELS) {
        return;
    }
    
    DmaChannel* channel = &g_dma_channels[handle->channel];
    if (channel->active) {
        // In real implementation, would stop DMA controller
        channel->active = false;
        handle->active = false;
        
        if (handle->callback) {
            handle->callback(handle->user_data, false);
        }
        
        g_dma_stats.transfers_failed++;
    }
}

const DmaCapabilities& cmx_dma_get_capabilities() {
    static const DmaCapabilities caps = {
        .num_channels = MAX_DMA_CHANNELS,
        .max_transfer_size = MAX_TRANSFER_SIZE,
        .alignment_requirement = DMA_ALIGNMENT,
        .supports_memory_to_memory = true,
        .supports_peripheral_to_memory = true,
        .supports_memory_to_peripheral = true,
        .supports_scatter_gather = false
    };
    
    return caps;
}

void* cmx_dma_alloc(size_t size, size_t alignment) {
    if (alignment == 0) {
        alignment = DMA_ALIGNMENT;
    }
    
    // In real implementation, this would allocate from DMA-accessible heap
    // For now, use regular allocation and hope it's DMA-accessible
    void* ptr = nullptr;
    
    // Simple aligned allocation
    size_t total_size = size + alignment - 1 + sizeof(void*);
    void* raw_ptr = std::malloc(total_size);
    
    if (raw_ptr) {
        uintptr_t aligned_addr = (reinterpret_cast<uintptr_t>(raw_ptr) + 
                                  sizeof(void*) + alignment - 1) & ~(alignment - 1);
        ptr = reinterpret_cast<void*>(aligned_addr);
        
        // Store original pointer for free()
        *(reinterpret_cast<void**>(ptr) - 1) = raw_ptr;
    }
    
    return ptr;
}

void cmx_dma_free(void* ptr) {
    if (ptr) {
        void* raw_ptr = *(reinterpret_cast<void**>(ptr) - 1);
        std::free(raw_ptr);
    }
}

const DmaStats& cmx_dma_get_stats() {
    return g_dma_stats;
}

void cmx_dma_reset_stats() {
    CMX_CRITICAL_SECTION();
    std::memset(&g_dma_stats, 0, sizeof(g_dma_stats));
}

} // namespace cmx::platform::xtensa