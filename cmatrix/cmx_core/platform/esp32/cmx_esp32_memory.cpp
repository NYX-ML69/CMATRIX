#include "cmx_esp32_memory.hpp"

#ifdef ESP_PLATFORM
#include "esp_heap_caps.h"
#include "esp_psram.h"
#include "esp_cache.h"
#include "esp_log.h"
#include "soc/soc_memory_layout.h"
#include "freertos/FreeRTOS.h"
#else
#include <cstdlib>
#include <cstring>
#endif

namespace cmx {
namespace platform {
namespace esp32 {

#ifdef ESP_PLATFORM
static const char* TAG = "CMX_ESP32_MEMORY";
#endif

// Static member definitions
bool MemoryManager::initialized_ = false;
bool MemoryManager::psram_available_ = false;
size_t MemoryManager::internal_heap_size_ = 0;
size_t MemoryManager::psram_heap_size_ = 0;

bool MemoryManager::initialize() {
    if (initialized_) {
        return true;
    }

#ifdef ESP_PLATFORM
    // Check if PSRAM is available
    psram_available_ = esp_psram_is_initialized();
    
    // Get initial heap sizes
    internal_heap_size_ = heap_caps_get_free_size(MALLOC_CAP_INTERNAL);
    if (psram_available_) {
        psram_heap_size_ = heap_caps_get_free_size(MALLOC_CAP_SPIRAM);
    }

    ESP_LOGI(TAG, "Memory Manager initialized");
    ESP_LOGI(TAG, "Internal RAM: %zu bytes", internal_heap_size_);
    if (psram_available_) {
        ESP_LOGI(TAG, "PSRAM available: %zu bytes", psram_heap_size_);
    } else {
        ESP_LOGI(TAG, "PSRAM not available");
    }
#else
    // Non-ESP32 platform fallback
    internal_heap_size_ = 1024 * 1024; // Assume 1MB for testing
    psram_available_ = false;
#endif

    initialized_ = true;
    return true;
}

size_t MemoryManager::get_available_memory(MemoryRegion region) {
    if (!initialized_) {
        initialize();
    }

#ifdef ESP_PLATFORM
    switch (region) {
        case MemoryRegion::INTERNAL_RAM:
            return heap_caps_get_free_size(MALLOC_CAP_INTERNAL);
        
        case MemoryRegion::PSRAM:
        case MemoryRegion::SPIRAM:
            return psram_available_ ? heap_caps_get_free_size(MALLOC_CAP_SPIRAM) : 0;
        
        case MemoryRegion::DMA_CAPABLE:
            return heap_caps_get_free_size(MALLOC_CAP_DMA);
        
        case MemoryRegion::IRAM:
            return heap_caps_get_free_size(MALLOC_CAP_EXEC);
        
        default:
            return 0;
    }
#else
    // Non-ESP32 fallback
    switch (region) {
        case MemoryRegion::INTERNAL_RAM:
            return internal_heap_size_;
        default:
            return 0;
    }
#endif
}

size_t MemoryManager::get_total_memory(MemoryRegion region) {
    if (!initialized_) {
        initialize();
    }

#ifdef ESP_PLATFORM
    switch (region) {
        case MemoryRegion::INTERNAL_RAM:
            return heap_caps_get_total_size(MALLOC_CAP_INTERNAL);
        
        case MemoryRegion::PSRAM:
        case MemoryRegion::SPIRAM:
            return psram_available_ ? heap_caps_get_total_size(MALLOC_CAP_SPIRAM) : 0;
        
        case MemoryRegion::DMA_CAPABLE:
            return heap_caps_get_total_size(MALLOC_CAP_DMA);
        
        case MemoryRegion::IRAM:
            return heap_caps_get_total_size(MALLOC_CAP_EXEC);
        
        default:
            return 0;
    }
#else
    // Non-ESP32 fallback
    switch (region) {
        case MemoryRegion::INTERNAL_RAM:
            return internal_heap_size_;
        default:
            return 0;
    }
#endif
}

void* MemoryManager::allocate_model_buffer(size_t size, const MemoryAttributes& attributes) {
    if (!initialized_) {
        initialize();
    }

    if (size == 0) {
        return nullptr;
    }

    // Align size to cache line boundary if requested
    if (attributes.cache_aligned) {
        size = memory_utils::align_size(size, memory_utils::get_cache_line_size());
    }

#ifdef ESP_PLATFORM
    uint32_t caps = 0;

    // Determine capability flags based on attributes
    if (attributes.dma_capable) {
        caps |= MALLOC_CAP_DMA;
    }

    if (attributes.prefer_internal) {
        caps |= MALLOC_CAP_INTERNAL;
    } else if (psram_available_) {
        caps |= MALLOC_CAP_SPIRAM;
    } else {
        caps |= MALLOC_CAP_INTERNAL;
    }

    // Try to allocate with alignment
    void* ptr = heap_caps_aligned_alloc(attributes.alignment, size, caps);
    
    if (!ptr && attributes.prefer_internal && psram_available_) {
        // Fallback to PSRAM if internal allocation failed
        caps = (caps & ~MALLOC_CAP_INTERNAL) | MALLOC_CAP_SPIRAM;
        ptr = heap_caps_aligned_alloc(attributes.alignment, size, caps);
    }

    if (ptr) {
        ESP_LOGD(TAG, "Allocated model buffer: %p, size: %zu", ptr, size);
    } else {
        ESP_LOGE(TAG, "Failed to allocate model buffer, size: %zu", size);
    }

    return ptr;
#else
    // Non-ESP32 fallback
    void* ptr = aligned_alloc(attributes.alignment, size);
    if (ptr) {
        memset(ptr, 0, size);
    }
    return ptr;
#endif
}

void* MemoryManager::allocate_tensor_arena(size_t size, const MemoryAttributes& attributes) {
    if (!initialized_) {
        initialize();
    }

    if (size == 0) {
        return nullptr;
    }

    // Align size to cache line boundary
    size = memory_utils::align_size(size, memory_utils::get_cache_line_size());

#ifdef ESP_PLATFORM
    uint32_t caps = 0;

    // For tensor arena, prioritize fast access
    if (attributes.prefer_internal) {
        caps |= MALLOC_CAP_INTERNAL;
    }

    if (attributes.dma_capable) {
        caps |= MALLOC_CAP_DMA;
    }

    // Try internal RAM first for better performance
    void* ptr = heap_caps_aligned_alloc(attributes.alignment, size, 
                                        caps | MALLOC_CAP_INTERNAL);
    
    if (!ptr && psram_available_ && !attributes.prefer_internal) {
        // Fallback to PSRAM if internal allocation failed
        caps = (caps & ~MALLOC_CAP_INTERNAL) | MALLOC_CAP_SPIRAM;
        ptr = heap_caps_aligned_alloc(attributes.alignment, size, caps);
    }

    if (ptr) {
        ESP_LOGD(TAG, "Allocated tensor arena: %p, size: %zu", ptr, size);
    } else {
        ESP_LOGE(TAG, "Failed to allocate tensor arena, size: %zu", size);
    }

    return ptr;
#else
    // Non-ESP32 fallback
    void* ptr = aligned_alloc(attributes.alignment, size);
    if (ptr) {
        memset(ptr, 0, size);
    }
    return ptr;
#endif
}

void* MemoryManager::allocate_dma_buffer(size_t size, uint32_t alignment) {
    if (!initialized_) {
        initialize();
    }

    if (size == 0) {
        return nullptr;
    }

    // Ensure minimum cache line alignment
    alignment = (alignment < memory_utils::get_cache_line_size()) ? 
                memory_utils::get_cache_line_size() : alignment;

    size = memory_utils::align_size(size, alignment);

#ifdef ESP_PLATFORM
    void* ptr = heap_caps_aligned_alloc(alignment, size, MALLOC_CAP_DMA);
    
    if (ptr) {
        ESP_LOGD(TAG, "Allocated DMA buffer: %p, size: %zu", ptr, size);
    } else {
        ESP_LOGE(TAG, "Failed to allocate DMA buffer, size: %zu", size);
    }

    return ptr;
#else
    // Non-ESP32 fallback
    void* ptr = aligned_alloc(alignment, size);
    if (ptr) {
        memset(ptr, 0, size);
    }
    return ptr;
#endif
}

void MemoryManager::free_buffer(void* ptr) {
    if (!ptr) {
        return;
    }

#ifdef ESP_PLATFORM
    heap_caps_free(ptr);
    ESP_LOGD(TAG, "Freed buffer: %p", ptr);
#else
    free(ptr);
#endif
}

bool MemoryManager::is_dma_capable(const void* ptr) {
    if (!ptr) {
        return false;
    }

#ifdef ESP_PLATFORM
    return heap_caps_check_integrity_addr((intptr_t)ptr, true) && 
           heap_caps_match(ptr, MALLOC_CAP_DMA);
#else
    // Non-ESP32 fallback - assume all memory is DMA capable
    return true;
#endif
}

bool MemoryManager::is_internal_ram(const void* ptr) {
    if (!ptr) {
        return false;
    }

#ifdef ESP_PLATFORM
    return heap_caps_check_integrity_addr((intptr_t)ptr, true) && 
           heap_caps_match(ptr, MALLOC_CAP_INTERNAL);
#else
    // Non-ESP32 fallback
    return true;
#endif
}

bool MemoryManager::is_psram(const void* ptr) {
    if (!ptr || !psram_available_) {
        return false;
    }

#ifdef ESP_PLATFORM
    return heap_caps_check_integrity_addr((intptr_t)ptr, true) && 
           heap_caps_match(ptr, MALLOC_CAP_SPIRAM);
#else
    // Non-ESP32 fallback
    return false;
#endif
}

MemoryRegion MemoryManager::get_optimal_region(size_t size, const MemoryAttributes& attributes) {
    if (!initialized_) {
        initialize();
    }

    // For DMA operations, must use DMA-capable memory
    if (attributes.dma_capable) {
        return MemoryRegion::DMA_CAPABLE;
    }

    // For small buffers or when preferring internal RAM
    if (attributes.prefer_internal || size < 32 * 1024) {
        if (get_available_memory(MemoryRegion::INTERNAL_RAM) >= size) {
            return MemoryRegion::INTERNAL_RAM;
        }
    }

    // For large buffers, prefer PSRAM if available
    if (psram_available_ && size >= 32 * 1024) {
        if (get_available_memory(MemoryRegion::PSRAM) >= size) {
            return MemoryRegion::PSRAM;
        }
    }

    // Fallback to internal RAM
    return MemoryRegion::INTERNAL_RAM;
}

void MemoryManager::flush_cache(const void* ptr, size_t size) {
    if (!ptr || size == 0) {
        return;
    }

#ifdef ESP_PLATFORM
    // Flush data cache for the given memory range
    esp_cache_msync((void*)ptr, size, ESP_CACHE_MSYNC_FLAG_DIR_C2M);
#else
    // Non-ESP32 fallback - no cache operations needed
    (void)ptr;
    (void)size;
#endif
}

void MemoryManager::invalidate_cache(const void* ptr, size_t size) {
    if (!ptr || size == 0) {
        return;
    }

#ifdef ESP_PLATFORM
    // Invalidate data cache for the given memory range
    esp_cache_msync((void*)ptr, size, ESP_CACHE_MSYNC_FLAG_DIR_M2C);
#else
    // Non-ESP32 fallback - no cache operations needed
    (void)ptr;
    (void)size;
#endif
}

void MemoryManager::print_memory_stats() {
    if (!initialized_) {
        initialize();
    }

#ifdef ESP_PLATFORM
    ESP_LOGI(TAG, "=== Memory Statistics ===");
    ESP_LOGI(TAG, "Internal RAM - Free: %zu, Total: %zu", 
             get_available_memory(MemoryRegion::INTERNAL_RAM),
             get_total_memory(MemoryRegion::INTERNAL_RAM));
    
    if (psram_available_) {
        ESP_LOGI(TAG, "PSRAM - Free: %zu, Total: %zu", 
                 get_available_memory(MemoryRegion::PSRAM),
                 get_total_memory(MemoryRegion::PSRAM));
    }
    
    ESP_LOGI(TAG, "DMA Capable - Free: %zu, Total: %zu", 
             get_available_memory(MemoryRegion::DMA_CAPABLE),
             get_total_memory(MemoryRegion::DMA_CAPABLE));
             
    // Print heap information
    heap_caps_print_heap_info(MALLOC_CAP_INTERNAL);
    if (psram_available_) {
        heap_caps_print_heap_info(MALLOC_CAP_SPIRAM);
    }
#else
    // Non-ESP32 fallback
    printf("=== Memory Statistics ===\n");
    printf("Internal RAM - Available: %zu\n", internal_heap_size_);
#endif
}

} // namespace esp32
} // namespace platform
} // namespace cmx