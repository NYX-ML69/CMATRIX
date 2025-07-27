#pragma once

#include <cstdint>
#include <cstddef>
#include <type_traits>

namespace cmx::platform::riscv {

/**
 * @brief Memory alignment requirements for RISC-V
 */
constexpr size_t CACHE_LINE_SIZE = 64;    // Typical cache line size
constexpr size_t WORD_ALIGNMENT = 8;      // 64-bit word alignment
constexpr size_t DMA_ALIGNMENT = 32;      // DMA buffer alignment requirement

/**
 * @brief Memory pool configuration
 */
struct MemoryPoolConfig {
    void* base_address;     ///< Start of memory pool
    size_t size;           ///< Total size of pool in bytes
    size_t alignment;      ///< Alignment requirement for allocations
};

/**
 * @brief Memory region types
 */
enum class MemoryRegion : uint8_t {
    SRAM,           ///< Fast SRAM for temporary data
    FLASH,          ///< Flash memory for constants/weights
    SCRATCHPAD,     ///< Scratchpad memory for intermediate calculations
    DMA_COHERENT    ///< DMA-coherent memory region
};

/**
 * @brief RISC-V Memory Manager
 * 
 * Manages static memory pools without dynamic allocation.
 * Provides aligned buffer allocation from pre-defined memory regions.
 */
class MemoryManager {
public:
    /**
     * @brief Initialize memory manager with static pools
     * @param pools Array of memory pool configurations
     * @param num_pools Number of memory pools
     * @return true if initialization successful
     */
    static bool initialize(const MemoryPoolConfig* pools, size_t num_pools);

    /**
     * @brief Allocate aligned buffer from specified memory region
     * @param size Size in bytes to allocate
     * @param alignment Alignment requirement (must be power of 2)
     * @param region Memory region to allocate from
     * @return Pointer to allocated buffer, nullptr if allocation failed
     */
    static void* allocate(size_t size, size_t alignment = WORD_ALIGNMENT, 
                         MemoryRegion region = MemoryRegion::SRAM);

    /**
     * @brief Free previously allocated buffer
     * @param ptr Pointer to buffer to free
     * @param size Size of buffer being freed
     */
    static void deallocate(void* ptr, size_t size);

    /**
     * @brief Get available memory in specified region
     * @param region Memory region to query
     * @return Available bytes in the region
     */
    static size_t get_available_memory(MemoryRegion region);

    /**
     * @brief Get total memory in specified region
     * @param region Memory region to query
     * @return Total bytes in the region
     */
    static size_t get_total_memory(MemoryRegion region);

    /**
     * @brief Reset all allocations in a memory region
     * @param region Memory region to reset
     */
    static void reset_region(MemoryRegion region);

    /**
     * @brief Check if pointer is properly aligned
     * @param ptr Pointer to check
     * @param alignment Required alignment
     * @return true if pointer is aligned
     */
    static inline bool is_aligned(const void* ptr, size_t alignment) {
        return (reinterpret_cast<uintptr_t>(ptr) & (alignment - 1)) == 0;
    }

    /**
     * @brief Align size to next boundary
     * @param size Size to align
     * @param alignment Alignment boundary
     * @return Aligned size
     */
    static inline size_t align_size(size_t size, size_t alignment) {
        return (size + alignment - 1) & ~(alignment - 1);
    }

    /**
     * @brief Memory barrier for cache coherency
     */
    static inline void memory_barrier() {
        asm volatile("fence" ::: "memory");
    }

    /**
     * @brief Data cache flush for DMA operations
     * @param ptr Pointer to data
     * @param size Size of data in bytes
     */
    static void flush_dcache(const void* ptr, size_t size);

    /**
     * @brief Data cache invalidate for DMA operations
     * @param ptr Pointer to data
     * @param size Size of data in bytes
     */
    static void invalidate_dcache(const void* ptr, size_t size);

private:
    static constexpr size_t MAX_MEMORY_POOLS = 4;
    
    struct MemoryPool {
        void* base_address;
        size_t total_size;
        size_t used_size;
        size_t alignment;
        MemoryRegion region;
        bool initialized;
    };
    
    static MemoryPool pools_[MAX_MEMORY_POOLS];
    static size_t num_pools_;
    static bool initialized_;
    
    static MemoryPool* find_pool(MemoryRegion region);
    static void* allocate_from_pool(MemoryPool* pool, size_t size, size_t alignment);
};

/**
 * @brief RAII memory buffer with automatic alignment
 * 
 * Template class for managing aligned memory buffers
 */
template<typename T, size_t Alignment = WORD_ALIGNMENT>
class AlignedBuffer {
public:
    /**
     * @brief Construct aligned buffer
     * @param count Number of elements
     * @param region Memory region to allocate from
     */
    explicit AlignedBuffer(size_t count, MemoryRegion region = MemoryRegion::SRAM)
        : ptr_(nullptr), size_(count * sizeof(T)), region_(region) {
        
        static_assert(std::is_trivially_destructible_v<T>, 
                     "T must be trivially destructible");
        static_assert((Alignment & (Alignment - 1)) == 0, 
                     "Alignment must be power of 2");
        
        ptr_ = static_cast<T*>(MemoryManager::allocate(size_, Alignment, region));
    }

    /**
     * @brief Destructor - free the buffer
     */
    ~AlignedBuffer() {
        if (ptr_) {
            MemoryManager::deallocate(ptr_, size_);
        }
    }

    // Non-copyable
    AlignedBuffer(const AlignedBuffer&) = delete;
    AlignedBuffer& operator=(const AlignedBuffer&) = delete;

    // Movable
    AlignedBuffer(AlignedBuffer&& other) noexcept 
        : ptr_(other.ptr_), size_(other.size_), region_(other.region_) {
        other.ptr_ = nullptr;
        other.size_ = 0;
    }

    AlignedBuffer& operator=(AlignedBuffer&& other) noexcept {
        if (this != &other) {
            if (ptr_) {
                MemoryManager::deallocate(ptr_, size_);
            }
            ptr_ = other.ptr_;
            size_ = other.size_;
            region_ = other.region_;
            other.ptr_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }

    /**
     * @brief Get pointer to buffer
     * @return Pointer to buffer data
     */
    T* data() { return ptr_; }
    const T* data() const { return ptr_; }

    /**
     * @brief Get buffer size in bytes
     * @return Size in bytes
     */
    size_t size_bytes() const { return size_; }

    /**
     * @brief Get number of elements
     * @return Number of elements
     */
    size_t count() const { return size_ / sizeof(T); }

    /**
     * @brief Check if allocation was successful
     * @return true if buffer is valid
     */
    bool is_valid() const { return ptr_ != nullptr; }

    /**
     * @brief Array access operator
     */
    T& operator[](size_t index) { return ptr_[index]; }
    const T& operator[](size_t index) const { return ptr_[index]; }

    /**
     * @brief Flush cache for DMA operations
     */
    void flush_cache() const {
        if (ptr_) {
            MemoryManager::flush_dcache(ptr_, size_);
        }
    }

    /**
     * @brief Invalidate cache for DMA operations
     */
    void invalidate_cache() const {
        if (ptr_) {
            MemoryManager::invalidate_dcache(ptr_, size_);
        }
    }

private:
    T* ptr_;
    size_t size_;
    MemoryRegion region_;
};

} // namespace cmx::platform::riscv