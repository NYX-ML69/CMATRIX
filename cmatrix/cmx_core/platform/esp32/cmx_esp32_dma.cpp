#include "cmx_esp32_dma.hpp"

// ESP-IDF includes
#include "driver/gdma.h"
#include "esp_cache.h"
#include "esp_heap_caps.h"
#include "esp_timer.h"
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"
#include "freertos/task.h"
#include "soc/soc_caps.h"

#include <cstring>
#include <algorithm>

namespace cmx {
namespace platform {
namespace esp32 {

static const char* TAG = "CMX_ESP32_DMA";

// Static member definitions
bool ESP32DMA::initialized_ = false;

// ESP32DMA Implementation
DMAStatus ESP32DMA::initialize() {
    if (initialized_) {
        ESP_LOGW(TAG, "DMA already initialized");
        return DMAStatus::SUCCESS;
    }
    
    ESP_LOGI(TAG, "Initializing DMA subsystem");
    
    // Check if GDMA is available on this chip
    if (!SOC_GDMA_SUPPORTED) {
        ESP_LOGE(TAG, "GDMA not supported on this chip");
        return DMAStatus::ERROR_INIT_FAILED;
    }
    
    initialized_ = true;
    ESP_LOGI(TAG, "DMA subsystem initialized successfully");
    return DMAStatus::SUCCESS;
}

DMAStatus ESP32DMA::shutdown() {
    if (!initialized_) {
        return DMAStatus::SUCCESS;
    }
    
    ESP_LOGI(TAG, "Shutting down DMA subsystem");
    initialized_ = false;
    return DMAStatus::SUCCESS;
}

bool ESP32DMA::is_initialized() {
    return initialized_;
}

// DMAChannel Implementation
DMAChannel::DMAChannel(const DMAConfig& config)
    : config_(config)
    , channel_handle_(nullptr)
    , initialized_(false)
    , stats_{0}
    , transfer_active_(false)
    , bytes_transferred_(0)
    , total_bytes_(0)
    , transfer_start_time_(0)
    , completion_callback_(nullptr) {
    
    if (!ESP32DMA::is_initialized()) {
        ESP_LOGE(TAG, "DMA subsystem not initialized");
        return;
    }
    
    if (configure_channel() == DMAStatus::SUCCESS) {
        initialized_ = true;
        ESP_LOGI(TAG, "DMA channel %d initialized", config_.channel_id);
    }
}

DMAChannel::~DMAChannel() {
    if (initialized_ && channel_handle_) {
        // Cancel any active transfer
        if (transfer_active_) {
            cancel_transfer();
        }
        
        // Delete the channel
        gdma_del_channel(channel_handle_);
        ESP_LOGI(TAG, "DMA channel %d destroyed", config_.channel_id);
    }
}

DMAChannel::DMAChannel(DMAChannel&& other) noexcept
    : config_(other.config_)
    , channel_handle_(other.channel_handle_)
    , initialized_(other.initialized_)
    , stats_(other.stats_)
    , transfer_active_(other.transfer_active_)
    , bytes_transferred_(other.bytes_transferred_)
    , total_bytes_(other.total_bytes_)
    , transfer_start_time_(other.transfer_start_time_)
    , completion_callback_(std::move(other.completion_callback_)) {
    
    other.channel_handle_ = nullptr;
    other.initialized_ = false;
    other.transfer_active_ = false;
}

DMAChannel& DMAChannel::operator=(DMAChannel&& other) noexcept {
    if (this != &other) {
        // Clean up current resources
        if (initialized_ && channel_handle_) {
            if (transfer_active_) {
                cancel_transfer();
            }
            gdma_del_channel(channel_handle_);
        }
        
        // Move data
        config_ = other.config_;
        channel_handle_ = other.channel_handle_;
        initialized_ = other.initialized_;
        stats_ = other.stats_;
        transfer_active_ = other.transfer_active_;
        bytes_transferred_ = other.bytes_transferred_;
        total_bytes_ = other.total_bytes_;
        transfer_start_time_ = other.transfer_start_time_;
        completion_callback_ = std::move(other.completion_callback_);
        
        // Reset other
        other.channel_handle_ = nullptr;
        other.initialized_ = false;
        other.transfer_active_ = false;
    }
    return *this;
}

bool DMAChannel::is_ready() const {
    return initialized_ && channel_handle_ != nullptr;
}

const DMAConfig& DMAChannel::get_config() const {
    return config_;
}

DMAStatus DMAChannel::transfer_sync(void* dst, const void* src, size_t size, 
                                   uint32_t timeout_ms) {
    DMATransfer transfer = {
        .src = const_cast<void*>(src),
        .dst = dst,
        .size = size,
        .direction = DMADirection::MEM_TO_MEM,
        .mode = DMAMode::BLOCKING,
        .timeout_ms = timeout_ms,
        .callback = nullptr,
        .periph_addr = 0,
        .periph_width = 0,
        .periph_inc = false
    };
    
    return this->transfer(transfer);
}

DMAStatus DMAChannel::transfer_async(void* dst, const void* src, size_t size,
                                    std::function<void(DMAStatus)> callback) {
    DMATransfer transfer = {
        .src = const_cast<void*>(src),
        .dst = dst,
        .size = size,
        .direction = DMADirection::MEM_TO_MEM,
        .mode = DMAMode::NON_BLOCKING,
        .timeout_ms = 0,
        .callback = callback,
        .periph_addr = 0,
        .periph_width = 0,
        .periph_inc = false
    };
    
    return this->transfer(transfer);
}

DMAStatus DMAChannel::transfer(const DMATransfer& transfer) {
    if (!is_ready()) {
        return DMAStatus::ERROR_INVALID_CHANNEL;
    }
    
    if (transfer_active_) {
        return DMAStatus::ERROR_BUSY;
    }
    
    if (!validate_transfer_params(transfer)) {
        return DMAStatus::ERROR_INVALID_PARAM;
    }
    
    uint64_t start_time = esp_timer_get_time();
    DMAStatus status = start_transfer(transfer);
    
    if (status != DMAStatus::SUCCESS) {
        return status;
    }
    
    // Handle blocking mode
    if (transfer.mode == DMAMode::BLOCKING) {
        status = wait_completion(transfer.timeout_ms);
        uint32_t transfer_time = (uint32_t)(esp_timer_get_time() - start_time);
        update_stats(status, transfer_time);
        return status;
    }
    
    return DMAStatus::SUCCESS;
}

DMAStatus DMAChannel::wait_completion(uint32_t timeout_ms) {
    if (!transfer_active_) {
        return DMAStatus::SUCCESS;
    }
    
    uint64_t start_time = esp_timer_get_time();
    uint64_t timeout_us = timeout_ms * 1000ULL;
    
    while (transfer_active_) {
        if (esp_timer_get_time() - start_time > timeout_us) {
            cancel_transfer();
            return DMAStatus::ERROR_TIMEOUT;
        }
        vTaskDelay(pdMS_TO_TICKS(1));
    }
    
    return DMAStatus::SUCCESS;
}

DMAStatus DMAChannel::cancel_transfer() {
    if (!is_ready() || !transfer_active_) {
        return DMAStatus::SUCCESS;
    }
    
    // Stop the channel
    esp_err_t err = gdma_stop(channel_handle_);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Failed to stop DMA channel: %s", esp_err_to_name(err));
        return DMAStatus::ERROR_TRANSFER_FAILED;
    }
    
    transfer_active_ = false;
    ESP_LOGW(TAG, "DMA transfer cancelled");
    
    return DMAStatus::SUCCESS;
}

bool DMAChannel::is_transfer_active() const {
    return transfer_active_;
}

float DMAChannel::get_transfer_progress() const {
    if (!transfer_active_ || total_bytes_ == 0) {
        return 0.0f;
    }
    
    return (float)bytes_transferred_ / (float)total_bytes_;
}

DMAStats DMAChannel::get_stats() const {
    return stats_;
}

void DMAChannel::reset_stats() {
    stats_ = {0};
}

DMAStatus DMAChannel::configure_channel() {
    // Create GDMA channel
    gdma_channel_alloc_config_t alloc_config = {
        .sibling_chan = nullptr,
        .direction = GDMA_CHANNEL_DIRECTION_TX, // Will be reconfigured per transfer
        .flags = {
            .reserve_sibling = 0
        }
    };
    
    esp_err_t err = gdma_new_channel(&alloc_config, &channel_handle_);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Failed to create GDMA channel: %s", esp_err_to_name(err));
        return DMAStatus::ERROR_INIT_FAILED;
    }
    
    // Set up callback if interrupts are enabled
    if (config_.enable_interrupt) {
        gdma_event_callbacks_t callbacks = {
            .on_trans_eof = transfer_done_callback
        };
        
        err = gdma_register_event_callbacks(channel_handle_, &callbacks, this);
        if (err != ESP_OK) {
            ESP_LOGE(TAG, "Failed to register DMA callbacks: %s", esp_err_to_name(err));
            gdma_del_channel(channel_handle_);
            channel_handle_ = nullptr;
            return DMAStatus::ERROR_INIT_FAILED;
        }
    }
    
    return DMAStatus::SUCCESS;
}

DMAStatus DMAChannel::start_transfer(const DMATransfer& transfer) {
    // Store transfer state
    transfer_active_ = true;
    bytes_transferred_ = 0;
    total_bytes_ = transfer.size;
    transfer_start_time_ = esp_timer_get_time();
    completion_callback_ = transfer.callback;
    
    // Flush/invalidate cache for coherency
    if (config_.src_cache_enable) {
        dma_utils::flush_cache(transfer.src, transfer.size);
    }
    if (config_.dst_cache_enable) {
        dma_utils::invalidate_cache(transfer.dst, transfer.size);
    }
    
    // Start the transfer
    esp_err_t err = gdma_start(channel_handle_, (intptr_t)transfer.src, 
                              (intptr_t)transfer.dst, transfer.size);
    
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Failed to start DMA transfer: %s", esp_err_to_name(err));
        transfer_active_ = false;
        return DMAStatus::ERROR_TRANSFER_FAILED;
    }
    
    stats_.total_transfers++;
    return DMAStatus::SUCCESS;
}

void DMAChannel::handle_transfer_completion(DMAStatus status) {
    transfer_active_ = false;
    bytes_transferred_ = total_bytes_;
    
    uint32_t transfer_time = (uint32_t)(esp_timer_get_time() - transfer_start_time_);
    update_stats(status, transfer_time);
    
    // Call completion callback if provided
    if (completion_callback_) {
        completion_callback_(status);
        completion_callback_ = nullptr;
    }
}

bool DMAChannel::validate_transfer_params(const DMATransfer& transfer) const {
    // Check alignment
    if (!dma_utils::is_address_aligned(transfer.src) || 
        !dma_utils::is_address_aligned(transfer.dst)) {
        ESP_LOGE(TAG, "Transfer addresses not aligned");
        return false;
    }
    
    if (!dma_utils::is_size_aligned(transfer.size)) {
        ESP_LOGE(TAG, "Transfer size not aligned");
        return false;
    }
    
    // Check size limits
    if (transfer.size == 0 || transfer.size > config_.max_transfer_size) {
        ESP_LOGE(TAG, "Invalid transfer size: %zu", transfer.size);
        return false;
    }
    
    // Check DMA accessibility
    if (!dma_utils::is_dma_accessible(transfer.src, transfer.size) ||
        !dma_utils::is_dma_accessible(transfer.dst, transfer.size)) {
        ESP_LOGE(TAG, "Memory not accessible by DMA");
        return false;
    }
    
    return true;
}

void DMAChannel::update_stats(DMAStatus status, uint32_t transfer_time_us) {
    if (status == DMAStatus::SUCCESS) {
        stats_.successful_transfers++;
        stats_.total_bytes += total_bytes_;
        
        // Update timing statistics
        if (stats_.successful_transfers == 1) {
            stats_.avg_transfer_time_us = transfer_time_us;
            stats_.max_transfer_time_us = transfer_time_us;
        } else {
            stats_.avg_transfer_time_us = 
                (stats_.avg_transfer_time_us * (stats_.successful_transfers - 1) + 
                 transfer_time_us) / stats_.successful_transfers;
            stats_.max_transfer_time_us = 
                std::max(stats_.max_transfer_time_us, transfer_time_us);
        }
    } else if (status == DMAStatus::ERROR_TIMEOUT) {
        stats_.timeout_transfers++;
        stats_.failed_transfers++;
    } else {
        stats_.failed_transfers++;
    }
}

bool DMAChannel::transfer_done_callback(gdma_channel_handle_t dma_chan, 
                                       gdma_event_data_t* event_data, 
                                       void* user_data) {
    DMAChannel* channel = static_cast<DMAChannel*>(user_data);
    if (channel) {
        channel->handle_transfer_completion(DMAStatus::SUCCESS);
    }
    return false; // Don't yield from ISR
}

// DMA utilities implementation
namespace dma_utils {

bool is_address_aligned(const void* addr, size_t alignment) {
    return (reinterpret_cast<uintptr_t>(addr) % alignment) == 0;
}

bool is_size_aligned(size_t size, size_t alignment) {
    return (size % alignment) == 0;
}

void* align_address_up(void* addr, size_t alignment) {
    uintptr_t aligned = (reinterpret_cast<uintptr_t>(addr) + alignment - 1) & 
                       ~(alignment - 1);
    return reinterpret_cast<void*>(aligned);
}

size_t align_size_up(size_t size, size_t alignment) {
    return (size + alignment - 1) & ~(alignment - 1);
}

void* allocate_dma_buffer(size_t size, size_t alignment, bool use_psram) {
    uint32_t caps = MALLOC_CAP_DMA;
    
    if (use_psram) {
        caps |= MALLOC_CAP_SPIRAM;
    } else {
        caps |= MALLOC_CAP_INTERNAL;
    }
    
    // Align size to cache line
    size_t aligned_size = align_size_up(size, alignment);
    
    void* buffer = heap_caps_aligned_alloc(alignment, aligned_size, caps);
    
    if (buffer) {
        ESP_LOGD(TAG, "Allocated DMA buffer: %p, size: %zu", buffer, aligned_size);
    } else {
        ESP_LOGE(TAG, "Failed to allocate DMA buffer of size %zu", aligned_size);
    }
    
    return buffer;
}

void free_dma_buffer(void* buffer) {
    if (buffer) {
        heap_caps_free(buffer);
        ESP_LOGD(TAG, "Freed DMA buffer: %p", buffer);
    }
}

DMAStatus optimal_memcpy(void* dst, const void* src, size_t size, bool force_dma) {
    // Use regular memcpy for small transfers unless forced
    if (!force_dma && size < 64) {
        std::memcpy(dst, src, size);
        return DMAStatus::SUCCESS;
    }
    
    // Check if DMA is beneficial and possible
    if (!is_address_aligned(dst) || !is_address_aligned(src) || 
        !is_size_aligned(size)) {
        std::memcpy(dst, src, size);
        return DMAStatus::SUCCESS;
    }
    
    // Use DMA for larger, aligned transfers
    static DMAChannel dma_channel; // Static channel for utility functions
    if (dma_channel.is_ready()) {
        return dma_channel.transfer_sync(dst, src, size, 1000);
    } else {
        std::memcpy(dst, src, size);
        return DMAStatus::SUCCESS;
    }
}

DMAStatus optimal_memset(void* dst, uint8_t value, size_t size, bool force_dma) {
    // DMA memset is more complex and typically not worth it for most cases
    std::memset(dst, value, size);
    return DMAStatus::SUCCESS;
}

void flush_cache(const void* addr, size_t size) {
    esp_cache_msync((void*)addr, size, ESP_CACHE_MSYNC_FLAG_DIR_C2M);
}

void invalidate_cache(const void* addr, size_t size) {
    esp_cache_msync((void*)addr, size, ESP_CACHE_MSYNC_FLAG_DIR_M2C);
}

size_t get_optimal_transfer_size(size_t size) {
    // Align to cache line for optimal performance
    return align_size_up(size, DMAAlignment::CACHE_LINE);
}

bool is_dma_accessible(const void* addr, size_t size) {
    return heap_caps_check_integrity_addr((intptr_t)addr, MALLOC_CAP_DMA) &&
           heap_caps_check_integrity_addr((intptr_t)addr + size - 1, MALLOC_CAP_DMA);
}

} // namespace dma_utils

// DMAMemoryPool Implementation
DMAMemoryPool::DMAMemoryPool(size_t pool_size, size_t block_size, bool use_psram)
    : pool_memory_(nullptr)
    , pool_size_(pool_size)
    , block_size_(dma_utils::align_size_up(block_size, DMAAlignment::CACHE_LINE))
    , total_blocks_(pool_size / block_size_)
    , free_blocks_(total_blocks_)
    , free_list_(nullptr)
    , use_psram_(use_psram) {
    
    // Allocate pool memory
    pool_memory_ = dma_utils::allocate_dma_buffer(pool_size_, 
                                                 DMAAlignment::CACHE_LINE, 
                                                 use_psram_);
    
    if (!pool_memory_) {
        ESP_LOGE(TAG, "Failed to allocate DMA memory pool");
        return;
    }
    
    // Allocate free list bitmap
    size_t bitmap_size = (total_blocks_ + 7) / 8; // Round up to bytes
    free_list_ = static_cast<uint8_t*>(heap_caps_malloc(bitmap_size, MALLOC_CAP_INTERNAL));
    
    if (!free_list_) {
        ESP_LOGE(TAG, "Failed to allocate free list bitmap");
        dma_utils::free_dma_buffer(pool_memory_);
        pool_memory_ = nullptr;
        return;
    }
    
    // Initialize all blocks as free
    std::memset(free_list_, 0, bitmap_size);
    
    ESP_LOGI(TAG, "DMA memory pool created: %zu blocks of %zu bytes each", 
             total_blocks_, block_size_);
}

DMAMemoryPool::~DMAMemoryPool() {
    if (pool_memory_) {
        dma_utils::free_dma_buffer(pool_memory_);
    }
    if (free_list_) {
        heap_caps_free(free_list_);
    }
}

void* DMAMemoryPool::allocate_block() {
    if (free_blocks_ == 0) {
        return nullptr;
    }
    
    // Find first free block
    for (size_t i = 0; i < total_blocks_; i++) {
        if (is_block_free(i)) {
            mark_block_used(i);
            free_blocks_--;
            return get_block_address(i);
        }
    }
    
    return nullptr;
}

bool DMAMemoryPool::free_block(void* block) {
    if (!owns_address(block)) {
        return false;
    }
    
    size_t index = get_block_index(block);
    if (is_block_free(index)) {
        return false; // Already free
    }
    
    mark_block_free(index);
    free_blocks_++;
    return true;
}

size_t DMAMemoryPool::get_free_blocks() const {
    return free_blocks_;
}

size_t DMAMemoryPool::get_total_blocks() const {
    return total_blocks_;
}

bool DMAMemoryPool::owns_address(const void* addr) const {
    if (!pool_memory_ || !addr) {
        return false;
    }
    
    uintptr_t pool_start = reinterpret_cast<uintptr_t>(pool_memory_);
    uintptr_t pool_end = pool_start + pool_size_;
    uintptr_t addr_val = reinterpret_cast<uintptr_t>(addr);
    
    return addr_val >= pool_start && addr_val < pool_end;
}

void DMAMemoryPool::mark_block_used(size_t index) {
    size_t byte_index = index / 8;
    size_t bit_index = index % 8;
    free_list_[byte_index] |= (1 << bit_index);
}

void DMAMemoryPool::mark_block_free(size_t index) {
    size_t byte_index = index / 8;
    size_t bit_index = index % 8;
    free_list_[byte_index] &= ~(1 << bit_index);
}

bool DMAMemoryPool::is_block_free(size_t index) const {
    size_t byte_index = index / 8;
    size_t bit_index = index % 8;
    return (free_list_[byte_index] & (1 << bit_index)) == 0;
}

} // namespace esp32
} // namespace platform
} // namespace cmx::reset() {
    if (free_list_) {
        size_t bitmap_size = (total_blocks_ + 7) / 8;
        std::memset(free_list_, 0, bitmap_size);
        free_blocks_ = total_blocks_;
    }
}

size_t DMAMemoryPool::get_block_index(const void* addr) const {
    uintptr_t pool_start = reinterpret_cast<uintptr_t>(pool_memory_);
    uintptr_t addr_val = reinterpret_cast<uintptr_t>(addr);
    return (addr_val - pool_start) / block_size_;
}

void* DMAMemoryPool::get_block_address(size_t index) const {
    uintptr_t pool_start = reinterpret_cast<uintptr_t>(pool_memory_);
    return reinterpret_cast<void*>(pool_start + index * block_size_);
}

void DMAMemoryPool