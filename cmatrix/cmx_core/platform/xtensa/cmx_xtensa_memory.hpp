#pragma once

#include <cstdint>
#include <cstddef>

namespace cmx::platform::xtensa {

/**
 * @brief Memory pool types
 */
enum class MemoryPoolType {
    FAST_RAM,        // Internal SRAM for time-critical data
    SLOW_RAM,        // External PSRAM for large allocations
    DMA_CAPABLE,     // Memory accessible by DMA
    INSTRUCTION      // Memory for executable code
};

/**
 * @brief Memory allocation flags
 */
enum MemoryFlags : uint32_t {
    ZERO_MEMORY = 0x01,      // Zero allocated memory
    ALIGN_4_BYTE = 0x02,     // 4-byte alignment
    ALIGN_8_BYTE = 0x04,     // 8-byte alignment
    ALIGN_16_BYTE = 0x08,    // 16-byte alignment
    ALIGN_32_BYTE = 0x10,    // 32-byte alignment
    NO_FAIL = 0x20           // Don't fail allocation, return null instead
};

/**
 * @brief Initialize memory subsystem
 */
void cmx_memory_init();

/**
 * @brief Allocate memory from default pool
 * @param size Size in bytes
 * @return Pointer to allocated memory, nullptr if failed
 */
void* cmx_alloc(size_t size);

/**
 * @brief Free memory allocated by cmx_alloc
 * @param ptr Pointer to free
 */
void cmx_free(void* ptr);

/**
 * @brief Allocate memory with specific flags
 * @param size Size in bytes
 * @param flags Memory allocation flags
 * @return Pointer to allocated memory, nullptr if failed
 */
void* cmx_alloc_flags(size_t size, uint32_t flags);

/**
 * @brief Allocate memory from specific pool
 * @param size Size in bytes
 * @param pool_type Pool type to allocate from
 * @param flags Optional flags
 * @return Pointer to allocated memory, nullptr if failed
 */
void* cmx_alloc_pool(size_t size, MemoryPoolType pool_type, uint32_t flags = 0);

/**
 * @brief Reallocate memory
 * @param ptr Existing pointer (can be nullptr)
 * @param new_size New size in bytes
 * @return Pointer to reallocated memory, nullptr if failed
 */
void* cmx_realloc(void* ptr, size_t new_size);

/**
 * @brief Allocate aligned memory
 * @param size Size in bytes
 * @param alignment Alignment in bytes (must be power of 2)
 * @return Pointer to aligned memory, nullptr if failed
 */
void* cmx_alloc_aligned(size_t size, size_t alignment);

/**
 * @brief Free aligned memory
 * @param ptr Pointer returned by cmx_alloc_aligned
 */
void cmx_free_aligned(void* ptr);

/**
 * @brief Get size of allocated block
 * @param ptr Pointer to allocated memory
 * @return Size of the block, 0 if invalid pointer
 */
size_t cmx_get_block_size(void* ptr);

/**
 * @brief Memory statistics
 */
struct MemoryStats {
    size_t total_bytes;
    size_t used_bytes;
    size_t free_bytes;
    size_t largest_free_block;
    uint32_t allocations;
    uint32_t deallocations;
    uint32_t allocation_failures;
    float fragmentation_ratio;
};

/**
 * @brief Get memory statistics for specific pool
 * @param pool_type Pool to get stats for
 * @return Memory statistics
 */
const MemoryStats& cmx_get_memory_stats(MemoryPoolType pool_type);

/**
 * @brief Get overall memory statistics
 * @return Combined memory statistics
 */
const MemoryStats& cmx_get_total_memory_stats();

/**
 * @brief Memory pool configuration
 */
struct MemoryPoolConfig {
    void* base_address;
    size_t size_bytes;
    size_t min_block_size;
    size_t alignment;
    bool enable_guard_pages;
    bool enable_statistics;
};

/**
 * @brief Configure memory pool
 * @param pool_type Pool type to configure
 * @param config Pool configuration
 * @return true if successful, false otherwise
 */
bool cmx_configure_memory_pool(MemoryPoolType pool_type, const MemoryPoolConfig& config);

/**
 * @brief Defragment memory pool
 * @param pool_type Pool to defragment
 * @return Number of blocks moved during defragmentation
 */
uint32_t cmx_defragment_pool(MemoryPoolType pool_type);

/**
 * @brief Check memory integrity
 * @param pool_type Pool to check (or all pools if not specified)
 * @return true if integrity check passed, false if corruption detected
 */
bool cmx_check_memory_integrity(MemoryPoolType pool_type);

/**
 * @brief Memory leak detection
 */
struct MemoryLeak {
    void* address;
    size_t size;
    const char* file;
    int line;
    uint32_t allocation_id;
};

/**
 * @brief Get memory leaks (debug builds only)
 * @param leaks Array to store leak information
 * @param max_leaks Maximum number of leaks to return
 * @return Number of leaks found
 */
uint32_t cmx_get_memory_leaks(MemoryLeak* leaks, uint32_t max_leaks);

/**
 * @brief Reset memory statistics
 * @param pool_type Pool to reset stats for
 */
void cmx_reset_memory_stats(MemoryPoolType pool_type);

/**
 * @brief Memory pool information
 */
struct MemoryPoolInfo {
    MemoryPoolType type;
    const char* name;
    void* base_address;
    size_t total_size;
    size_t block_size;
    size_t alignment;
    bool is_dma_capable;
    bool is_cache_coherent;
};

/**
 * @brief Get memory pool information
 * @param pool_type Pool type
 * @return Pool information
 */
const MemoryPoolInfo& cmx_get_pool_info(MemoryPoolType pool_type);

/**
 * @brief Stack-based memory allocator for temporary allocations
 */
class StackAllocator {
public:
    StackAllocator(size_t size);
    ~StackAllocator();
    
    void* alloc(size_t size, size_t alignment = 4);
    void reset();
    size_t get_used() const;
    size_t get_remaining() const;
    
private:
    void* buffer_;
    size_t total_size_;
    size_t current_offset_;
};

} // namespace cmx::platform::xtensa

// Debug macros for memory leak detection
#ifdef CMX_DEBUG_MEMORY
#define CMX_ALLOC(size) cmx::platform::xtensa::cmx_alloc_debug(size, __FILE__, __LINE__)
#define CMX_FREE(ptr) cmx::platform::xtensa::cmx_free_debug(ptr, __FILE__, __LINE__)
void* cmx_alloc_debug(size_t size, const char* file, int line);
void cmx_free_debug(void* ptr, const char* file, int line);
#else
#define CMX_ALLOC(size) cmx::platform::xtensa::cmx_alloc(size)
#define CMX_FREE(ptr) cmx::platform::xtensa::cmx_free(ptr)
#endif