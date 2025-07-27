// cmx_allocator.hpp
#pragma once

#include "cmx_runtime_config.hpp"
#include <cstddef>
#include <cstdint>
#include <atomic>

namespace cmx {
namespace runtime {

/**
 * @brief Arena allocator for fixed-size memory blocks
 * 
 * Provides fast allocation from a pre-allocated memory arena.
 * No dynamic allocation at runtime - all memory is reserved during initialization.
 */
class CMXAllocator {
public:
    /**
     * @brief Allocation statistics for debugging and profiling
     */
    struct Stats {
        size_t total_size;
        size_t used_size;
        size_t peak_usage;
        size_t allocation_count;
        size_t deallocation_count;
        
        Stats() : total_size(0), used_size(0), peak_usage(0), 
                 allocation_count(0), deallocation_count(0) {}
    };

    /**
     * @brief Initialize allocator with memory arena
     * @param memory_ptr Pointer to pre-allocated memory block
     * @param size Size of memory block in bytes
     */
    explicit CMXAllocator(void* memory_ptr, size_t size);
    
    /**
     * @brief Destructor
     */
    ~CMXAllocator() = default;
    
    // Non-copyable, non-movable
    CMXAllocator(const CMXAllocator&) = delete;
    CMXAllocator& operator=(const CMXAllocator&) = delete;
    CMXAllocator(CMXAllocator&&) = delete;
    CMXAllocator& operator=(CMXAllocator&&) = delete;
    
    /**
     * @brief Allocate aligned memory block
     * @param size Size in bytes
     * @param alignment Alignment requirement (default: MEMORY_ALIGNMENT)
     * @return Pointer to allocated memory or nullptr on failure
     */
    void* allocate(size_t size, size_t alignment = RuntimeConfig::MEMORY_ALIGNMENT);
    
    /**
     * @brief Deallocate memory block
     * @param ptr Pointer to memory to deallocate
     * @note Currently no-op for arena allocator, memory is freed on reset()
     */
    void deallocate(void* ptr);
    
    /**
     * @brief Reset allocator to initial state
     * @note Invalidates all previously allocated pointers
     */
    void reset();
    
    /**
     * @brief Get current allocation statistics
     */
    Stats get_stats() const;
    
    /**
     * @brief Check if allocator is valid and initialized
     */
    bool is_valid() const { return memory_start_ != nullptr; }
    
    /**
     * @brief Get remaining available memory
     */
    size_t available_memory() const;

private:
    void* memory_start_;        // Start of memory arena
    size_t memory_size_;        // Total size of memory arena
    std::atomic<size_t> offset_; // Current allocation offset
    mutable Stats stats_;       // Allocation statistics
    
    /**
     * @brief Align pointer to specified boundary
     */
    static void* align_pointer(void* ptr, size_t alignment);
    
    /**
     * @brief Calculate aligned size
     */
    static size_t align_size(size_t size, size_t alignment);
};

} // namespace runtime
} // namespace cmx