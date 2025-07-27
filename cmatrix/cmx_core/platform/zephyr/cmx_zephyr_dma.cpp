/**
 * @file cmx_zephyr_dma.cpp
 * @brief Implementation of DMA abstraction layer for Zephyr RTOS
 */

#include "cmx_zephyr_dma.hpp"
#include "cmx_zephyr_port.hpp"

#include <zephyr/kernel.h>
#include <zephyr/logging/log.h>
#include <zephyr/drivers/dma.h>
#include <zephyr/device.h>

#include <string.h>
#include <algorithm>

LOG_MODULE_REGISTER(cmx_dma, LOG_LEVEL_DBG);

namespace cmx::platform::zephyr {

// Maximum number of DMA channels to manage
constexpr uint32_t MAX_DMA_CHANNELS = 8;

// DMA channel state
struct DmaChannelState {
    const struct device* dma_dev;
    uint32_t channel_id;
    bool allocated;
    bool active;
    size_t transfer_size;
    size_t bytes_transferred;
    dma_callback_t callback;
    void* user_data;
    struct k_sem completion_sem;
};

// Global DMA state
static struct {
    bool initialized;
    const struct device* default_dma_dev;
    DmaChannelState channels[MAX_DMA_CHANNELS];
    struct k_mutex channel_mutex;
} dma_state = { false };

// DMA completion callback
static void dma_completion_callback(const struct device* dma_dev, void* user_data,
                                  uint32_t channel, int status) {
    DmaChannelState* ch_state = static_cast<DmaChannelState*>(user_data);
    
    if (ch_state == nullptr || ch_state >= &dma_state.channels[MAX_DMA_CHANNELS]) {
        LOG_ERR("Invalid DMA channel state in callback");
        return;
    }
    
    ch_state->active = false;
    
    if (status == 0) {
        ch_state->bytes_transferred = ch_state->transfer_size;
        LOG_DBG("DMA transfer completed on channel %d", channel);
    } else {
        LOG_ERR("DMA transfer failed on channel %d: %d", channel, status);
    }
    
    // Signal completion semaphore
    k_sem_give(&ch_state->completion_sem);
    
    // Call user callback if provided
    if (ch_state->callback) {
        ch_state->callback(channel, status, ch_state->user_data);
    }
}

bool dma_init() {
    if (dma_state.initialized) {
        return true;
    }
    
    LOG_INF("Initializing DMA subsystem...");
    
    // Find default DMA device
    dma_state.default_dma_dev = DEVICE_DT_GET_ANY(dma);
    if (dma_state.default_dma_dev == nullptr || 
        !device_is_ready(dma_state.default_dma_dev)) {
        LOG_WRN("No DMA device found or not ready, DMA disabled");
        dma_state.initialized = true; // Mark as initialized but without hardware
        return true;
    }
    
    LOG_INF("Found DMA device: %s", dma_state.default_dma_dev->name);
    
    // Initialize mutex
    k_mutex_init(&dma_state.channel_mutex);
    
    // Initialize channel states
    for (uint32_t i = 0; i < MAX_DMA_CHANNELS; i++) {
        DmaChannelState* ch = &dma_state.channels[i];
        ch->dma_dev = dma_state.default_dma_dev;
        ch->channel_id = i;
        ch->allocated = false;
        ch->active = false;
        ch->transfer_size = 0;
        ch->bytes_transferred = 0;
        ch->callback = nullptr;
        ch->user_data = nullptr;
        k_sem_init(&ch->completion_sem, 0, 1);
    }
    
    dma_state.initialized = true;
    LOG_INF("DMA subsystem initialized successfully");
    return true;
}

void dma_cleanup() {
    if (!dma_state.initialized) {
        return;
    }
    
    LOG_INF("Cleaning up DMA subsystem...");
    
    k_mutex_lock(&dma_state.channel_mutex, K_FOREVER);
    
    // Stop and release all channels
    for (uint32_t i = 0; i < MAX_DMA_CHANNELS; i++) {
        DmaChannelState* ch = &dma_state.channels[i];
        if (ch->allocated) {
            if (ch->active && dma_state.default_dma_dev) {
                dma_stop(dma_state.default_dma_dev, ch->channel_id);
            }
            ch->allocated = false;
            ch->active = false;
        }
    }
    
    k_mutex_unlock(&dma_state.channel_mutex);
    
    dma_state.initialized = false;
    LOG_INF("DMA subsystem cleanup complete");
}

int dma_allocate_channel(const DmaConfig& config) {
    if (!dma_state.initialized || !dma_state.default_dma_dev) {
        return -1;
    }
    
    k_mutex_lock(&dma_state.channel_mutex, K_FOREVER);
    
    // Find free channel
    int channel_handle = -1;
    for (uint32_t i = 0; i < MAX_DMA_CHANNELS; i++) {
        if (!dma_state.channels[i].allocated) {
            dma_state.channels[i].allocated = true;
            dma_state.channels[i].channel_id = config.channel_id;
            channel_handle = static_cast<int>(i);
            break;
        }
    }
    
    k_mutex_unlock(&dma_state.channel_mutex);
    
    if (channel_handle >= 0) {
        LOG_DBG("Allocated DMA channel %d (handle %d)", config.channel_id, channel_handle);
    } else {
        LOG_ERR("No free DMA channels available");
    }
    
    return channel_handle;
}

bool dma_release_channel(int channel) {
    if (channel < 0 || channel >= MAX_DMA_CHANNELS || !dma_state.initialized) {
        return false;
    }
    
    k_mutex_lock(&dma_state.channel_mutex, K_FOREVER);
    
    DmaChannelState* ch = &dma_state.channels[channel];
    if (!ch->allocated) {
        k_mutex_unlock(&dma_state.channel_mutex);
        return false;
    }
    
    // Stop any active transfer
    if (ch->active && dma_state.default_dma_dev) {
        dma_stop(dma_state.default_dma_dev, ch->channel_id);
        ch->active = false;
    }
    
    ch->allocated = false;
    ch->callback = nullptr;
    ch->user_data = nullptr;
    
    k_mutex_unlock(&dma_state.channel_mutex);
    
    LOG_DBG("Released DMA channel %d", channel);
    return true;
}

bool cmx_dma_transfer(void* dst, const void* src, size_t size) {
    if (!dma_state.initialized || !dma_state.default_dma_dev || 
        dst == nullptr || src == nullptr || size == 0) {
        // Fallback to memcpy if DMA not available
        if (dst && src && size > 0) {
            memcpy(dst, src, size);
            return true;
        }
        return false;
    }
    
    // Allocate temporary channel for synchronous transfer
    DmaConfig config = {
        .channel_id = 0,
        .source_width = 4,
        .dest_width = 4,
        .increment_src = true,
        .increment_dst = true,
        .priority = 0
    };
    
    int channel = dma_allocate_channel(config);
    if (channel < 0) {
        // Fallback to memcpy
        memcpy(dst, src, size);
        return true;
    }
    
    bool result = dma_transfer_async(channel, dst, src, size, nullptr, nullptr);
    if (result) {
        result = dma_wait_completion(channel, 5000); // 5 second timeout
    }
    
    dma_release_channel(channel);
    
    if (!result) {
        // Fallback to memcpy on failure
        memcpy(dst, src, size);
        return true;
    }
    
    return result;
}

bool dma_transfer_async(int channel, void* dst, const void* src, size_t size,
                       dma_callback_t callback, void* user_data) {
    if (channel < 0 || channel >= MAX_DMA_CHANNELS || !dma_state.initialized ||
        !dma_state.default_dma_dev || dst == nullptr || src == nullptr || size == 0) {
        return false;
    }
    
    DmaChannelState* ch = &dma_state.channels[channel];
    if (!ch->allocated || ch->active) {
        return false;
    }
    
    // Configure DMA transfer
    struct dma_config dma_cfg = {};
    struct dma_block_config block_cfg = {};
    
    dma_cfg.channel_direction = MEMORY_TO_MEMORY;
    dma_cfg.source_data_size = 4; // 32-bit transfers
    dma_cfg.dest_data_size = 4;
    dma_cfg.source_burst_length = 1;
    dma_cfg.dest_burst_length = 1;
    dma_cfg.dma_callback = dma_completion_callback;
    dma_cfg.user_data = ch;
    dma_cfg.block_count = 1;
    dma_cfg.head_block = &block_cfg;
    
    block_cfg.source_address = (uintptr_t)src;
    block_cfg.dest_address = (uintptr_t)dst;
    block_cfg.block_size = size;
    
    ch->transfer_size = size;
    ch->bytes_transferred = 0;
    ch->callback = callback;
    ch->user_data = user_data;
    ch->active = true;
    
    // Reset completion semaphore
    k_sem_reset(&ch->completion_sem);
    
    int ret = dma_config(dma_state.default_dma_dev, ch->channel_id, &dma_cfg);
    if (ret != 0) {
        LOG_ERR("DMA config failed: %d", ret);
        ch->active = false;
        return false;
    }
    
    ret = dma_start(dma_state.default_dma_dev, ch->channel_id);
    if (ret != 0) {
        LOG_ERR("DMA start failed: %d", ret);
        ch->active = false;
        return false;
    }
    
    LOG_DBG("Started async DMA transfer: %p -> %p (%zu bytes)", src, dst, size);
    return true;
}

bool dma_wait_completion(int channel, uint32_t timeout_ms) {
    if (channel < 0 || channel >= MAX_DMA_CHANNELS || !dma_state.initialized) {
        return false;
    }
    
    DmaChannelState* ch = &dma_state.channels[channel];
    if (!ch->allocated) {
        return false;
    }
    
    k_timeout_t timeout = timeout_ms ? K_MSEC(timeout_ms) : K_FOREVER;
    int ret = k_sem_take(&ch->completion_sem, timeout);
    
    return (ret == 0);
}

bool dma_cancel_transfer(int channel) {
    if (channel < 0 || channel >= MAX_DMA_CHANNELS || !dma_state.initialized ||
        !dma_state.default_dma_dev) {
        return false;
    }
    
    DmaChannelState* ch = &dma_state.channels[channel];
    if (!ch->allocated || !ch->active) {
        return false;
    }
    
    int ret = dma_stop(dma_state.default_dma_dev, ch->channel_id);
    if (ret == 0) {
        ch->active = false;
        k_sem_give(&ch->completion_sem);
        LOG_DBG("Cancelled DMA transfer on channel %d", channel);
        return true;
    }
    
    LOG_ERR("Failed to cancel DMA transfer on channel %d: %d", channel, ret);
    return false;
}

bool dma_is_active(int channel) {
    if (channel < 0 || channel >= MAX_DMA_CHANNELS || !dma_state.initialized) {
        return false;
    }
    
    DmaChannelState* ch = &dma_state.channels[channel];
    return ch->allocated && ch->active;
}

size_t dma_get_progress(int channel) {
    if (channel < 0 || channel >= MAX_DMA_CHANNELS || !dma_state.initialized) {
        return 0;
    }
    
    DmaChannelState* ch = &dma_state.channels[channel];
    if (!ch->allocated) {
        return 0;
    }
    
    return ch->bytes_transferred;
}

bool dma_configure_mem2mem(int channel, uint32_t src_width, uint32_t dst_width) {
    if (channel < 0 || channel >= MAX_DMA_CHANNELS || !dma_state.initialized) {
        return false;
    }
    
    DmaChannelState* ch = &dma_state.channels[channel];
    if (!ch->allocated || ch->active) {
        return false;
    }
    
    // Validate data widths
    if (src_width == 0 || dst_width == 0 || 
        src_width > 8 || dst_width > 8 ||
        (src_width & (src_width - 1)) != 0 || // Must be power of 2
        (dst_width & (dst_width - 1)) != 0) {
        return false;
    }
    
    LOG_DBG("Configured mem2mem DMA channel %d: src_width=%d, dst_width=%d", 
            channel, src_width, dst_width);
    return true;
}

uint32_t dma_get_channel_count() {
    return MAX_DMA_CHANNELS;
}

bool dma_is_available() {
    return dma_state.initialized && dma_state.default_dma_dev != nullptr;
}

uint32_t dma_get_capabilities() {
    if (!dma_is_available()) {
        return 0;
    }
    
    // Return basic capabilities - actual capabilities would depend on hardware
    return DMA_CAP_MEM2MEM | DMA_CAP_BURST | DMA_CAP_PRIORITY;
}

} // namespace cmx::platform::zephyr