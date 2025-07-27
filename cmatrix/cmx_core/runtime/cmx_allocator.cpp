// cmx_allocator.cpp
#include "cmx_allocator.hpp"
#include <algorithm>
#include <cassert>

namespace cmx {
namespace runtime {

CMXAllocator::CMXAllocator(void* memory_ptr, size_t size) 
    : memory_start_(memory_ptr), memory_size_(size), offset_(0) {
    assert(memory_ptr != nullptr);
    assert(size > 0);
    
    stats_.total_size = size;
}

void* CMXAllocator::allocate(size_t size, size_t alignment) {
    if (size == 0 || !is_valid()) {
        return nullptr;
    }
    
    // Calculate aligned size
    size_t aligned_size = align_size(size, alignment);
    
    // Atomic compare-and-swap loop for thread safety
    size_t current_offset = offset_.load();
    size_t new_offset;
    
    do {
        // Check if we have enough space
        if (current_offset + aligned_size > memory_size_) {
            return nullptr; // Out of memory
        }
        
        new_offset = current_offset + aligned_size;
    } while (!offset_.compare_exchange_weak(current_offset, new_offset));
    
    // Calculate aligned pointer
    void* ptr = static_cast<char*>(memory_start_) + current_offset;
    void* aligned_ptr = align_pointer(ptr, alignment);
    
    // Update statistics
    stats_.used_size = new_offset;
    stats_.peak_usage = std::max(stats_.peak_usage, stats_.used_size);
    stats_.allocation_count++;
    
    return aligned_ptr;
}

void CMXAllocator::deallocate(void* ptr) {
    // Arena allocator doesn't support individual deallocation
    // Memory is freed on reset()
    if (ptr != nullptr) {
        stats_.deallocation_count++;
    }
}

void CMXAllocator::reset() {
    offset_.store(0);
    stats_.used_size = 0;
    stats_.allocation_count = 0;
    stats_.deallocation_count = 0;
}

CMXAllocator::Stats CMXAllocator::get_stats() const {
    Stats current_stats = stats_;
    current_stats.used_size = offset_.load();
    return current_stats;
}

size_t CMXAllocator::available_memory() const {
    size_t used = offset_.load();
    return used <= memory_size_ ? memory_size_ - used : 0;
}

void* CMXAllocator::align_pointer(void* ptr, size_t alignment) {
    uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
    uintptr_t aligned_addr = (addr + alignment - 1) & ~(alignment - 1);
    return reinterpret_cast<void*>(aligned_addr);
}

size_t CMXAllocator::align_size(size_t size, size_t alignment) {
    return (size + alignment - 1) & ~(alignment - 1);
}

} // namespace runtime
} // namespace cmx