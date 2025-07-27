/**
 * @file cmx_zephyr_memory.hpp
 * @brief Memory management abstraction for Zephyr RTOS
 * 
 * Provides static and region-based memory allocation for tensors
 * and workspace without relying on dynamic heap allocation.
 */

#pragma once

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

namespace cmx::platform::zephyr {

/**
 * @brief Memory pool statistics
 */
struct MemoryStats {
    size_t total_size;        ///< Total pool size in bytes
    size_t used_size;         ///< Currently allocated bytes
    size_t free_size;         ///< Available bytes
    size_t largest_free;      ///< Size of largest free block
    uint32_t alloc_count;     ///< Total allocations made
    uint32_t free_count;      ///< Total deallocations made
    uint32_t active_blocks;   ///< Currently active allocations
};

/**
 * @brief Memory alignment requirements
 */
enum class MemAlign : size_t {
    BYTE_1   = 1,
    BYTE_2   = 2,
    BYTE_4   = 4,
    BYTE_8   = 8,
    BYTE_16  = 16,
    BYTE_32  = 32,
    BYTE_64  = 64,
    CACHE_LINE = 64  ///< Typical cache line size
};

/**
 * @brief Memory pool types
 */
enum class PoolType {
    TENSOR_POOL,      ///< Large blocks for tensor data
    WORKSPACE_POOL,   ///< Medium blocks for temporary calculations
    SMALL_POOL,       ///< Small blocks for metadata and structures
    DMA_POOL          ///< DMA-coherent memory pool
};

/**
 * @brief Initialize memory management subsystem
 * @return true if successful, false otherwise
 * 
 * Sets up memory pools and internal tracking structures
 */
bool memory_init();

/**
 * @brief Cleanup memory management subsystem
 * Releases all memory pools and checks for leaks
 */
void memory_cleanup();

/**
 * @brief Allocate memory from default pool
 * @param size Number of bytes to allocate
 * @return Pointer to allocated memory, nullptr on failure
 */
void* cmx_alloc(size_t size);

/**
 * @brief Allocate aligned memory from default pool
 * @param size Number of bytes to allocate
 * @param align Memory alignment requirement
 * @return Pointer to allocated memory, nullptr on failure
 */
void* cmx_alloc_aligned(size_t size, MemAlign align);

/**
 * @brief Allocate memory from specific pool
 * @param size Number of bytes to allocate
 * @param pool_type Pool to allocate from
 * @return Pointer to allocated memory, nullptr on failure
 */
void* cmx_alloc_from_pool(size_t size, PoolType pool_type);

/**
 * @brief Allocate aligned memory from specific pool
 * @param size Number of bytes to allocate
 * @param align Memory alignment requirement
 * @param pool_type Pool to allocate from
 * @return Pointer to allocated memory, nullptr on failure
 */
void* cmx_alloc_from_pool_aligned(size_t size, MemAlign align, PoolType pool_type);

/**
 * @brief Free previously allocated memory
 * @param ptr Pointer returned by cmx_alloc* functions
 */
void cmx_free(void* ptr);

/**
 * @brief Allocate and zero-initialize memory
 * @param count Number of elements
 * @param size Size of each element
 * @return Pointer to allocated memory, nullptr on failure
 */
void* cmx_calloc(size_t count, size_t size);

/**
 * @brief Reallocate memory block
 * @param ptr Existing memory block (or nullptr)
 * @param new_size New size in bytes
 * @return Pointer to reallocated memory, nullptr on failure
 */
void* cmx_realloc(void* ptr, size_t new_size);

/**
 * @brief Get size of allocated memory block
 * @param ptr Pointer returned by cmx_alloc* functions
 * @return Size of allocated block, 0 if invalid pointer
 */
size_t cmx_get_block_size(void* ptr);

/**
 * @brief Check if pointer is valid allocated memory
 * @param ptr Pointer to check
 * @return true if valid allocation, false otherwise
 */
bool cmx_is_valid_ptr(void* ptr);

/**
 * @brief Get memory pool statistics
 * @param pool_type Pool to query
 * @param stats Output statistics structure
 * @return true if successful, false if pool doesn't exist
 */
bool cmx_get_memory_stats(PoolType pool_type, MemoryStats& stats);

/**
 * @brief Get total memory statistics across all pools
 * @param stats Output statistics structure
 */
void cmx_get_total_memory_stats(MemoryStats& stats);

/**
 * @brief Check for memory leaks
 * @return Number of leaked blocks (0 = no leaks)
 */
uint32_t cmx_check_memory_leaks();

/**
 * @brief Dump memory pool information for debugging
 * @param pool_type Pool to dump (or all pools if not specified)
 */
void cmx_dump_memory_info(PoolType pool_type = PoolType::TENSOR_POOL);

/**
 * @brief Set memory allocation failure callback
 * @param callback Function to call when allocation fails
 */
typedef void (*memory_failure_callback_t)(size_t size, PoolType pool_type);
void cmx_set_memory_failure_callback(memory_failure_callback_t callback);

/**
 * @brief Stack-based memory allocator for temporary allocations
 */
class ScratchAllocator {
public:
    /**
     * @brief Initialize scratch allocator
     * @param size Size of scratch buffer in bytes
     */
    explicit ScratchAllocator(size_t size);
    
    /**
     * @brief Destructor - releases all scratch memory
     */
    ~ScratchAllocator();
    
    /**
     * @brief Allocate from scratch buffer
     * @param size Number of bytes to allocate
     * @param align Alignment requirement
     * @return Pointer to allocated memory, nullptr on failure
     */
    void* alloc(size_t size, MemAlign align = MemAlign::BYTE_8);
    
    /**
     * @brief Reset scratch allocator (free all allocations)
     */
    void reset();
    
    /**
     * @brief Get remaining scratch space
     * @return Available bytes in scratch buffer
     */
    size_t get_remaining() const;
    
    /**
     * @brief Get total scratch space
     * @return Total scratch buffer size
     */
    size_t get_total_size() const;

private:
    void* scratch_buffer_;
    size_t total_size_;
    size_t current_offset_;
    bool owns_buffer_;
};

/**
 * @brief Memory region for batch allocations
 */
class MemoryRegion {
public:
    /**
     * @brief Create memory region
     * @param size Total region size
     * @param pool_type Pool to allocate from
     */
    MemoryRegion(size_t size, PoolType pool_type = PoolType::TENSOR_POOL);
    
    /**
     * @brief Destructor - releases region memory
     */
    ~MemoryRegion();
    
    /**
     * @brief Allocate from region
     * @param size Number of bytes to allocate
     * @param align Alignment requirement
     * @return Pointer to allocated memory, nullptr on failure
     */
    void* alloc(size_t size, MemAlign align = MemAlign::BYTE_8);
    
    /**
     * @brief Get region base address
     * @return Base pointer of memory region
     */
    void* get_base() const { return base_ptr_; }
    
    /**
     * @brief Get region size
     * @return Total region size in bytes
     */
    size_t get_size() const { return total_size_; }
    
    /**
     * @brief Get used space in region
     * @return Number of bytes allocated from region
     */
    size_t get_used() const { return used_size_; }
    
    /**
     * @brief Get remaining space in region
     * @return Number of bytes available in region
     */
    size_t get_remaining() const { return total_size_ - used_size_; }

private:
    void* base_ptr_;
    size_t total_size_;
    size_t used_size_;
    PoolType pool_type_;
};

} // namespace cmx::platform::zephyr