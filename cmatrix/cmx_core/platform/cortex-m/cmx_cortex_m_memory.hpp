#pragma once

#include <cstdint>
#include <cstddef>

namespace cmx {
namespace platform {
namespace cortex_m {

/**
 * @brief Memory region configuration
 */
struct MemoryRegion {
    void* base_address;     ///< Base address of memory region
    size_t size;           ///< Size of memory region in bytes
    bool is_cacheable;     ///< Whether region is cacheable
    bool is_dma_coherent;  ///< Whether region is DMA coherent
    const char* name;      ///< Human-readable name for debugging
};

/**
 * @brief Memory allocation alignment options
 */
enum class MemoryAlignment : uint8_t {
    BYTE_1 = 1,
    BYTE_2 = 2,
    BYTE_4 = 4,
    BYTE_8 = 8,
    BYTE_16 = 16,
    BYTE_32 = 32,
    CACHE_LINE = 64  ///< Typical ARM cache line size
};

/**
 * @brief Fast memory access utilities for Cortex-M
 * 
 * Provides optimized memory operations, static buffer management,
 * and memory-mapped interface access for CMX runtime.
 * No dynamic allocation - uses pre-allocated static buffers.
 */
class Memory {
public:
    /**
     * @brief Initialize memory subsystem
     * @param tensor_arena_size Size of static tensor arena in bytes
     * @return true if initialization successful
     */
    static bool init(size_t tensor_arena_size = 65536);
    
    /**
     * @brief Deinitialize memory subsystem
     */
    static void deinit();
    
    /**
     * @brief Get pointer to tensor arena
     * @return Pointer to static tensor arena, nullptr if not initialized
     */
    static void* get_tensor_arena();
    
    /**
     * @brief Get size of tensor arena
     * @return Size of tensor arena in bytes
     */
    static size_t get_tensor_arena_size();
    
    /**
     * @brief Allocate aligned buffer from tensor arena
     * @param size Size in bytes to allocate
     * @param alignment Memory alignment requirement
     * @return Pointer to allocated buffer, nullptr if insufficient space
     */
    static void* allocate_aligned(size_t size, MemoryAlignment alignment = MemoryAlignment::BYTE_4);
    
    /**
     * @brief Reset tensor arena allocation pointer
     * Allows reuse of arena from beginning
     */
    static void reset_arena();
    
    /**
     * @brief Get remaining free space in tensor arena
     * @return Free bytes remaining in arena
     */
    static size_t get_free_arena_space();
    
    /**
     * @brief Fast memory copy (optimized for Cortex-M)
     * @param dest Destination buffer
     * @param src Source buffer  
     * @param size Number of bytes to copy
     */
    static void fast_copy(void* dest, const void* src, size_t size);
    
    /**
     * @brief Fast memory set (optimized for Cortex-M)
     * @param dest Destination buffer
     * @param value Byte value to set
     * @param size Number of bytes to set
     */
    static void fast_set(void* dest, uint8_t value, size_t size);
    
    /**
     * @brief Fast memory compare
     * @param ptr1 First buffer
     * @param ptr2 Second buffer
     * @param size Number of bytes to compare
     * @return 0 if equal, non-zero if different
     */
    static int fast_compare(const void* ptr1, const void* ptr2, size_t size);
    
    /**
     * @brief Register external memory region
     * @param region Memory region configuration
     * @return true if registration successful
     */
    static bool register_memory_region(const MemoryRegion& region);
    
    /**
     * @brief Get registered memory region by name
     * @param name Region name to search for
     * @return Pointer to region info, nullptr if not found
     */
    static const MemoryRegion* get_memory_region(const char* name);
    
    /**
     * @brief Check if address is in valid memory range
     * @param address Address to validate
     * @param size Size of memory access
     * @return true if address range is valid
     */
    static bool is_valid_address(const void* address, size_t size);
    
    /**
     * @brief Get memory usage statistics
     * @param total_arena_size Output: total arena size
     * @param used_arena_size Output: used arena size
     * @param free_arena_size Output: free arena size
     */
    static void get_memory_stats(size_t* total_arena_size, 
                                size_t* used_arena_size, 
                                size_t* free_arena_size);
    
    /**
     * @brief Flush cache for memory region (if caching enabled)
     * @param address Start address
     * @param size Size of region to flush
     */
    static void flush_cache(const void* address, size_t size);
    
    /**
     * @brief Invalidate cache for memory region (if caching enabled)
     * @param address Start address  
     * @param size Size of region to invalidate
     */
    static void invalidate_cache(const void* address, size_t size);
    
    /**
     * @brief Check if memory subsystem is initialized
     * @return true if initialized
     */
    static bool is_initialized();

private:
    Memory() = delete;
    ~Memory() = delete;
    Memory(const Memory&) = delete;
    Memory& operator=(const Memory&) = delete;
    
    /**
     * @brief Align pointer to specified boundary
     * @param ptr Pointer to align
     * @param alignment Alignment boundary
     * @return Aligned pointer
     */
    static void* align_pointer(void* ptr, size_t alignment);
    
    /**
     * @brief Check if pointer is aligned
     * @param ptr Pointer to check
     * @param alignment Alignment to check against
     * @return true if aligned
     */
    static bool is_aligned(const void* ptr, size_t alignment);
};

/**
 * @brief RAII memory buffer allocator
 * 
 * Automatically allocates from tensor arena on construction
 * and tracks allocation for debugging purposes.
 */
class MemoryBuffer {
public:
    /**
     * @brief Allocate buffer from tensor arena
     * @param size Size in bytes to allocate
     * @param alignment Memory alignment requirement
     */
    MemoryBuffer(size_t size, MemoryAlignment alignment = MemoryAlignment::BYTE_4);
    
    /**
     * @brief Buffer is not owned, so destructor does nothing
     * (Arena-based allocation doesn't support individual deallocation)
     */
    ~MemoryBuffer() = default;
    
    /**
     * @brief Get pointer to allocated buffer
     * @return Buffer pointer, nullptr if allocation failed
     */
    void* data() const { return buffer_; }
    
    /**
     * @brief Get size of allocated buffer
     * @return Buffer size in bytes
     */
    size_t size() const { return size_; }
    
    /**
     * @brief Check if allocation was successful
     * @return true if buffer is valid
     */
    bool is_valid() const { return buffer_ != nullptr; }
    
    /**
     * @brief Cast buffer to specific type
     * @tparam T Type to cast to
     * @return Typed pointer to buffer
     */
    template<typename T>
    T* as() const {
        return static_cast<T*>(buffer_);
    }

private:
    void* buffer_;
    size_t size_;
    
    // Non-copyable
    MemoryBuffer(const MemoryBuffer&) = delete;
    MemoryBuffer& operator=(const MemoryBuffer&) = delete;
};

} // namespace cortex_m
} // namespace platform  
} // namespace cmx