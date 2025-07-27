#pragma once

#include <cstddef>
#include <cstdint>

namespace cmx {
namespace platform {
namespace nios2 {

/**
 * @brief Memory management configuration for Nios II embedded systems
 */
struct MemoryConfig {
    static constexpr size_t MEMORY_POOL_SIZE = 64 * 1024;  // 64KB default pool
    static constexpr size_t ALIGNMENT = 4;                 // 32-bit alignment
    static constexpr size_t MAX_ALLOCATIONS = 256;         // Max concurrent allocations
};

/**
 * @brief Memory block descriptor for tracking allocations
 */
struct MemoryBlock {
    void* ptr;
    size_t size;
    bool in_use;
    uint32_t magic;  // For corruption detection
};

/**
 * @brief Initialize the memory management system
 * Must be called before any memory allocation operations
 */
void cmx_memory_init();

/**
 * @brief Allocate memory from the static pool
 * @param size Number of bytes to allocate
 * @return Pointer to allocated memory, nullptr on failure
 */
void* cmx_alloc(size_t size);

/**
 * @brief Free previously allocated memory
 * @param ptr Pointer to memory to free (must be from cmx_alloc)
 */
void cmx_free(void* ptr);

/**
 * @brief Get current memory usage statistics
 * @param used_bytes Output: bytes currently allocated
 * @param free_bytes Output: bytes available for allocation
 * @param fragmentation Output: fragmentation percentage (0-100)
 */
void cmx_memory_stats(size_t* used_bytes, size_t* free_bytes, uint8_t* fragmentation);

/**
 * @brief Perform memory pool defragmentation
 * @return Number of blocks moved during defragmentation
 */
size_t cmx_memory_defrag();

/**
 * @brief Check memory pool integrity
 * @return true if all blocks are valid, false if corruption detected
 */
bool cmx_memory_check();

/**
 * @brief Aligned allocation for DMA-compatible buffers
 * @param size Number of bytes to allocate
 * @param alignment Alignment requirement (must be power of 2)
 * @return Pointer to aligned memory, nullptr on failure
 */
void* cmx_alloc_aligned(size_t size, size_t alignment);

} // namespace nios2
} // namespace platform
} // namespace cmx