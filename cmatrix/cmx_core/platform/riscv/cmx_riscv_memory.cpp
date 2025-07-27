#include "cmx_riscv_memory.hpp"
#include <cstring>

namespace cmx::platform::riscv {

// Static member definitions
MemoryManager::MemoryPool MemoryManager::pools_[MAX_MEMORY_POOLS] = {};
size_t MemoryManager::num_pools_ = 0;
bool MemoryManager::initialized_ = false;

bool MemoryManager::initialize(const MemoryPoolConfig* pools, size_t num_pools) {
    if (initialized_ || num_pools > MAX_MEMORY_POOLS) {
        return false;
    }

    // Initialize memory pools
    for (size_t i = 0; i < num_pools; ++i) {
        pools_[i].base_address = pools[i].base_address;
        pools_[i].total_size = pools[i].size;
        pools_[i].used_size = 0;
        pools_[i].alignment = pools[i].alignment;
        pools_[i].region = static_cast<MemoryRegion>(i); // Simple mapping for now
        pools_[i].initialized = true;
        
        // Verify alignment
        if (!is_aligned(pools[i].base_address, pools[i].alignment)) {
            return false;
        }
    }
    
    num_pools_ = num_pools;
    initialized_ = true;
    
    return true;
}

void* MemoryManager::allocate(size_t size, size_t alignment, MemoryRegion region) {
    if (!initialized_ || size == 0) {
        return nullptr;
    }

    MemoryPool* pool = find_pool(region);
    if (!pool) {
        return nullptr;
    }

    return allocate_from_pool(pool, size, alignment);
}

void MemoryManager::deallocate(void* ptr, size_t size) {
    // Simple implementation - doesn't actually free memory
    // In a more sophisticated implementation, we could maintain a free list
    // For now, memory is only reclaimed when reset_region() is called
    (void)ptr;
    (void)size;
}

size_t MemoryManager::get_available_memory(MemoryRegion region) {
    if (!initialized_) {
        return 0;
    }

    MemoryPool* pool = find_pool(region);
    if (!pool) {
        return 0;
    }

    return pool->total_size - pool->used_size;
}

size_t MemoryManager::get_total_memory(MemoryRegion region) {
    if (!initialized_) {
        return 0;
    }

    MemoryPool* pool = find_pool(region);
    if (!pool) {
        return 0;
    }

    return pool->total_size;
}

void MemoryManager::reset_region(MemoryRegion region) {
    if (!initialized_) {
        return;
    }

    MemoryPool* pool = find_pool(region);
    if (pool) {
        pool->used_size = 0;
    }
}

void MemoryManager::flush_dcache(const void* ptr, size_t size) {
    // RISC-V cache management - implementation depends on specific processor
    // For processors with cache management instructions:
    
    if (!ptr || size == 0) {
        return;
    }

    const uintptr_t start = reinterpret_cast<uintptr_t>(ptr);
    const uintptr_t end = start + size;
    
    // Align to cache line boundaries
    const uintptr_t start_aligned = start & ~(CACHE_LINE_SIZE - 1);
    const uintptr_t end_aligned = (end + CACHE_LINE_SIZE - 1) & ~(CACHE_LINE_SIZE - 1);
    
    // Flush cache lines - this is processor specific
    // Many RISC-V implementations may not have cache management instructions
    // In that case, a full memory barrier might be sufficient
    memory_barrier();
    
    // For processors that support cache management:
    // for (uintptr_t addr = start_aligned; addr < end_aligned; addr += CACHE_LINE_SIZE) {
    //     asm volatile("cflush.d1 %0" :: "r"(addr) : "memory");
    // }
}

void MemoryManager::invalidate_dcache(const void* ptr, size_t size) {
    // RISC-V cache management - implementation depends on specific processor
    
    if (!ptr || size == 0) {
        return;
    }

    const uintptr_t start = reinterpret_cast<uintptr_t>(ptr);
    const uintptr_t end = start + size;
    
    // Align to cache line boundaries
    const uintptr_t start_aligned = start & ~(CACHE_LINE_SIZE - 1);
    const uintptr_t end_aligned = (end + CACHE_LINE_SIZE - 1) & ~(CACHE_LINE_SIZE - 1);
    
    // Invalidate cache lines - this is processor specific
    memory_barrier();
    
    // For processors that support cache management:
    // for (uintptr_t addr = start_aligned; addr < end_aligned; addr += CACHE_LINE_SIZE) {
    //     asm volatile("cdiscard.d1 %0" :: "r"(addr) : "memory");
    // }
}

MemoryManager::MemoryPool* MemoryManager::find_pool(MemoryRegion region) {
    for (size_t i = 0; i < num_pools_; ++i) {
        if (pools_[i].initialized && pools_[i].region == region) {
            return &pools_[i];
        }
    }
    return nullptr;
}

void* MemoryManager::allocate_from_pool(MemoryPool* pool, size_t size, size_t alignment) {
    if (!pool || size == 0) {
        return nullptr;
    }

    // Align the current allocation pointer
    uintptr_t current_ptr = reinterpret_cast<uintptr_t>(pool->base_address) + pool->used_size;
    uintptr_t aligned_ptr = (current_ptr + alignment - 1) & ~(alignment - 1);
    
    // Calculate actual size needed including alignment padding
    size_t padding = aligned_ptr - current_ptr;
    size_t total_size = padding + size;
    
    // Check if we have enough space
    if (pool->used_size + total_size > pool->total_size) {
        return nullptr;
    }
    
    // Update pool usage
    pool->used_size += total_size;
    
    return reinterpret_cast<void*>(aligned_ptr);
}

} // namespace cmx::platform::riscv