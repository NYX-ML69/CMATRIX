// cmx_memory_pool.cpp
#include "cmx_memory_pool.hpp"
#include <algorithm>
#include <cassert>

namespace cmx {
namespace runtime {

CMXMemoryPool& CMXMemoryPool::getInstance() {
    static CMXMemoryPool instance;
    return instance;
}

CMXMemoryPool::~CMXMemoryPool() {
    shutdown();
}

bool CMXMemoryPool::initialize(size_t total_size) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (initialized_) {
        return true; // Already initialized
    }
    
    // Allocate main memory block
    memory_block_ = std::make_unique<char[]>(total_size);
    if (!memory_block_) {
        return false;
    }
    
    total_memory_size_ = total_size;
    
    // Calculate pool sizes
    size_t tensor_size, temp_size, general_size;
    calculate_pool_sizes(total_size, tensor_size, temp_size, general_size);
    
    // Create individual allocators
    char* memory_ptr = memory_block_.get();
    
    tensor_allocator_ = std::make_unique<CMXAllocator>(memory_ptr, tensor_size);
    memory_ptr += tensor_size;
    
    temp_buffer_allocator_ = std::make_unique<CMXAllocator>(memory_ptr, temp_size);
    memory_ptr += temp_size;
    
    general_allocator_ = std::make_unique<CMXAllocator>(memory_ptr, general_size);
    
    initialized_ = true;
    return true;
}

void CMXMemoryPool::shutdown() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (!initialized_) {
        return;
    }
    
    // Reset all allocators
    tensor_allocator_.reset();
    temp_buffer_allocator_.reset();
    general_allocator_.reset();
    
    // Free main memory block
    memory_block_.reset();
    total_memory_size_ = 0;
    initialized_ = false;
}

CMXAllocator* CMXMemoryPool::get_allocator(PoolType pool_type) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (!initialized_) {
        return nullptr;
    }
    
    switch (pool_type) {
        case PoolType::TENSOR_POOL:
            return tensor_allocator_.get();
        case PoolType::TEMP_BUFFER_POOL:
            return temp_buffer_allocator_.get();
        case PoolType::GENERAL_POOL:
            return general_allocator_.get();
        default:
            return nullptr;
    }
}

void CMXMemoryPool::free_all() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (!initialized_) {
        return;
    }
    
    tensor_allocator_->reset();
    temp_buffer_allocator_->reset();
    general_allocator_->reset();
}

CMXMemoryPool::MemoryStats CMXMemoryPool::get_memory_stats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    MemoryStats stats = {};
    
    if (!initialized_) {
        return stats;
    }
    
    stats.total_size = total_memory_size_;
    
    auto tensor_stats = tensor_allocator_->get_stats();
    auto temp_stats = temp_buffer_allocator_->get_stats();
    auto general_stats = general_allocator_->get_stats();
    
    stats.tensor_pool_used = tensor_stats.used_size;
    stats.temp_buffer_used = temp_stats.used_size;
    stats.general_pool_used = general_stats.used_size;
    stats.peak_usage = tensor_stats.peak_usage + temp_stats.peak_usage + general_stats.peak_usage;
    
    return stats;
}

void CMXMemoryPool::calculate_pool_sizes(size_t total_size, size_t& tensor_size, 
                                        size_t& temp_size, size_t& general_size) {
    // Default split: 60% tensor, 25% temp, 15% general
    tensor_size = (total_size * 60) / 100;
    temp_size = (total_size * 25) / 100;
    general_size = total_size - tensor_size - temp_size;
    
    // Ensure minimum sizes
    tensor_size = std::max(tensor_size, size_t(RuntimeConfig::DEFAULT_TENSOR_POOL_SIZE));
    temp_size = std::max(temp_size, size_t(RuntimeConfig::DEFAULT_TEMP_BUFFER_SIZE));
    general_size = std::max(general_size, size_t(64 * 1024)); // 64KB minimum
}

} // namespace runtime
} // namespace cmx