// cmx_memory_pool.hpp
#pragma once

#include "cmx_allocator.hpp"
#include "cmx_runtime_config.hpp"
#include <memory>
#include <mutex>

namespace cmx {
namespace runtime {

/**
 * @brief Centralized memory pool manager
 * 
 * Manages multiple memory pools for different purposes:
 * - Tensor data pool
 * - Temporary buffer pool
 * - General purpose pool
 */
class CMXMemoryPool {
public:
    /**
     * @brief Memory pool types
     */
    enum class PoolType {
        TENSOR_POOL,
        TEMP_BUFFER_POOL,
        GENERAL_POOL
    };
    
    /**
     * @brief Memory usage statistics
     */
    struct MemoryStats {
        size_t total_size;
        size_t tensor_pool_used;
        size_t temp_buffer_used;
        size_t general_pool_used;
        size_t peak_usage;
    };
    
    /**
     * @brief Get singleton instance
     */
    static CMXMemoryPool& getInstance();
    
    /**
     * @brief Initialize memory pools
     * @param total_size Total memory size to allocate
     * @return True on success, false on failure
     */
    bool initialize(size_t total_size = CMX_MEMORY_POOL_SIZE);
    
    /**
     * @brief Shutdown and cleanup memory pools
     */
    void shutdown();
    
    /**
     * @brief Get allocator for specific pool type
     * @param pool_type Type of memory pool
     * @return Pointer to allocator or nullptr if not initialized
     */
    CMXAllocator* get_allocator(PoolType pool_type = PoolType::GENERAL_POOL);
    
    /**
     * @brief Free all memory in all pools
     */
    void free_all();
    
    /**
     * @brief Check if memory pool is initialized
     */
    bool is_initialized() const { return initialized_; }
    
    /**
     * @brief Get total memory usage statistics
     */
    MemoryStats get_memory_stats() const;

private:
    CMXMemoryPool() = default;
    ~CMXMemoryPool();
    
    // Non-copyable, non-movable
    CMXMemoryPool(const CMXMemoryPool&) = delete;
    CMXMemoryPool& operator=(const CMXMemoryPool&) = delete;
    CMXMemoryPool(CMXMemoryPool&&) = delete;
    CMXMemoryPool& operator=(CMXMemoryPool&&) = delete;
    
    bool initialized_ = false;
    std::unique_ptr<char[]> memory_block_;
    size_t total_memory_size_ = 0;
    
    // Individual allocators for different pool types
    std::unique_ptr<CMXAllocator> tensor_allocator_;
    std::unique_ptr<CMXAllocator> temp_buffer_allocator_;
    std::unique_ptr<CMXAllocator> general_allocator_;
    
    mutable std::mutex mutex_;
    
    /**
     * @brief Calculate memory split for different pools
     */
    void calculate_pool_sizes(size_t total_size, size_t& tensor_size, 
                             size_t& temp_size, size_t& general_size);
};

} // namespace runtime
} // namespace cmx