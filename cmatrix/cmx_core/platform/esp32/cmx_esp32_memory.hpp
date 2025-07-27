#pragma once

#include <cstdint>
#include <cstddef>

#ifdef ESP_PLATFORM
#include "esp_heap_caps.h"
#include "esp_psram.h"
#include "soc/soc_memory_layout.h"
#endif

namespace cmx {
namespace platform {
namespace esp32 {

/**
 * @brief Memory region types available on ESP32
 */
enum class MemoryRegion {
    INTERNAL_RAM,    // Fast internal SRAM (DRAM)
    PSRAM,          // External PSRAM (if available)
    IRAM,           // Instruction RAM (for code execution)
    DMA_CAPABLE,    // DMA-capable memory regions
    SPIRAM          // SPI RAM (external)
};

/**
 * @brief Memory allocation attributes for ESP32
 */
struct MemoryAttributes {
    bool dma_capable = false;       // Memory accessible by DMA
    bool cache_aligned = true;      // Align to cache line boundaries
    bool prefer_internal = true;    // Prefer internal RAM over PSRAM
    uint32_t alignment = 32;        // Memory alignment in bytes
};

/**
 * @brief ESP32 Memory Manager for CMX Runtime
 * 
 * Provides optimized memory allocation and management for ML inference
 * on ESP32 platforms. Manages both internal SRAM and external PSRAM
 * with consideration for DMA access and cache alignment.
 */
class MemoryManager {
public:
    /**
     * @brief Initialize the memory manager
     * @return true if initialization successful, false otherwise
     */
    static bool initialize();

    /**
     * @brief Get available memory information
     * @param region Memory region to query
     * @return Available memory in bytes, 0 if region not available
     */
    static size_t get_available_memory(MemoryRegion region);

    /**
     * @brief Get total memory size for a region
     * @param region Memory region to query
     * @return Total memory size in bytes, 0 if region not available
     */
    static size_t get_total_memory(MemoryRegion region);

    /**
     * @brief Allocate memory buffer for ML model weights
     * @param size Size in bytes to allocate
     * @param attributes Memory allocation attributes
     * @return Pointer to allocated memory, nullptr if allocation failed
     */
    static void* allocate_model_buffer(size_t size, const MemoryAttributes& attributes = {});

    /**
     * @brief Allocate tensor arena for inference operations
     * @param size Size in bytes to allocate
     * @param attributes Memory allocation attributes
     * @return Pointer to allocated memory, nullptr if allocation failed
     */
    static void* allocate_tensor_arena(size_t size, const MemoryAttributes& attributes = {});

    /**
     * @brief Allocate DMA-capable buffer
     * @param size Size in bytes to allocate
     * @param alignment Memory alignment requirement
     * @return Pointer to allocated memory, nullptr if allocation failed
     */
    static void* allocate_dma_buffer(size_t size, uint32_t alignment = 32);

    /**
     * @brief Free previously allocated memory
     * @param ptr Pointer to memory to free
     */
    static void free_buffer(void* ptr);

    /**
     * @brief Check if pointer is in DMA-capable memory
     * @param ptr Pointer to check
     * @return true if DMA-capable, false otherwise
     */
    static bool is_dma_capable(const void* ptr);

    /**
     * @brief Check if pointer is in internal RAM
     * @param ptr Pointer to check
     * @return true if in internal RAM, false otherwise
     */
    static bool is_internal_ram(const void* ptr);

    /**
     * @brief Check if pointer is in PSRAM
     * @param ptr Pointer to check
     * @return true if in PSRAM, false otherwise
     */
    static bool is_psram(const void* ptr);

    /**
     * @brief Get optimal memory region for given size and attributes
     * @param size Size in bytes
     * @param attributes Memory allocation attributes
     * @return Recommended memory region
     */
    static MemoryRegion get_optimal_region(size_t size, const MemoryAttributes& attributes);

    /**
     * @brief Flush cache for given memory range (if applicable)
     * @param ptr Pointer to memory range
     * @param size Size of memory range in bytes
     */
    static void flush_cache(const void* ptr, size_t size);

    /**
     * @brief Invalidate cache for given memory range (if applicable)
     * @param ptr Pointer to memory range
     * @param size Size of memory range in bytes
     */
    static void invalidate_cache(const void* ptr, size_t size);

    /**
     * @brief Print memory usage statistics (debug)
     */
    static void print_memory_stats();

private:
    static bool initialized_;
    static bool psram_available_;
    static size_t internal_heap_size_;
    static size_t psram_heap_size_;
    
    // Prevent instantiation
    MemoryManager() = delete;
    ~MemoryManager() = delete;
    MemoryManager(const MemoryManager&) = delete;
    MemoryManager& operator=(const MemoryManager&) = delete;
};

/**
 * @brief RAII wrapper for automatic memory cleanup
 */
template<typename T>
class ManagedBuffer {
public:
    explicit ManagedBuffer(size_t count, const MemoryAttributes& attributes = {})
        : ptr_(static_cast<T*>(MemoryManager::allocate_tensor_arena(
            count * sizeof(T), attributes))), size_(count) {}

    ~ManagedBuffer() {
        if (ptr_) {
            MemoryManager::free_buffer(ptr_);
        }
    }

    // Move constructor
    ManagedBuffer(ManagedBuffer&& other) noexcept
        : ptr_(other.ptr_), size_(other.size_) {
        other.ptr_ = nullptr;
        other.size_ = 0;
    }

    // Move assignment
    ManagedBuffer& operator=(ManagedBuffer&& other) noexcept {
        if (this != &other) {
            if (ptr_) {
                MemoryManager::free_buffer(ptr_);
            }
            ptr_ = other.ptr_;
            size_ = other.size_;
            other.ptr_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }

    // Delete copy constructor and assignment
    ManagedBuffer(const ManagedBuffer&) = delete;
    ManagedBuffer& operator=(const ManagedBuffer&) = delete;

    T* get() const { return ptr_; }
    size_t size() const { return size_; }
    bool valid() const { return ptr_ != nullptr; }

    T& operator[](size_t index) { return ptr_[index]; }
    const T& operator[](size_t index) const { return ptr_[index]; }

private:
    T* ptr_;
    size_t size_;
};

// Memory utility functions
namespace memory_utils {

/**
 * @brief Calculate aligned size for given alignment
 * @param size Original size
 * @param alignment Alignment requirement
 * @return Aligned size
 */
constexpr size_t align_size(size_t size, size_t alignment) {
    return (size + alignment - 1) & ~(alignment - 1);
}

/**
 * @brief Check if address is aligned
 * @param ptr Pointer to check
 * @param alignment Alignment requirement
 * @return true if aligned, false otherwise
 */
constexpr bool is_aligned(const void* ptr, size_t alignment) {
    return (reinterpret_cast<uintptr_t>(ptr) & (alignment - 1)) == 0;
}

/**
 * @brief Get cache line size for ESP32
 * @return Cache line size in bytes
 */
constexpr size_t get_cache_line_size() {
    return 32; // ESP32 cache line size
}

} // namespace memory_utils

} // namespace esp32
} // namespace platform
} // namespace cmx