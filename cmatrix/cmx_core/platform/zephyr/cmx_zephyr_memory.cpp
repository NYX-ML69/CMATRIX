#include "cmx_zephyr_memory.hpp"
#include <zephyr/kernel.h>
#include <zephyr/sys/heap.h>
#include <zephyr/logging/log.h>

LOG_MODULE_REGISTER(cmx_memory, LOG_LEVEL_INF);

namespace cmx::platform::zephyr {

// Static memory pool configuration
#define CMX_MEMORY_POOL_SIZE (64 * 1024)  // 64KB default pool
#define CMX_MEMORY_BLOCK_ALIGN 8          // 8-byte alignment
#define CMX_MAX_BLOCKS 256                // Maximum allocation blocks

// Static memory buffer for the pool
static uint8_t memory_pool_buffer[CMX_MEMORY_POOL_SIZE] __aligned(CMX_MEMORY_BLOCK_ALIGN);
static sys_heap heap;
static bool memory_initialized = false;
static k_mutex memory_mutex;

// Allocation tracking for debugging
struct allocation_info {
    void* ptr;
    size_t size;
    bool in_use;
};

static allocation_info allocations[CMX_MAX_BLOCKS];
static int allocation_count = 0;

void cmx_memory_init() {
    if (memory_initialized) {
        LOG_WRN("Memory already initialized");
        return;
    }

    // Initialize mutex for thread-safe operations
    k_mutex_init(&memory_mutex);

    // Initialize system heap
    sys_heap_init(&heap, memory_pool_buffer, CMX_MEMORY_POOL_SIZE);

    // Clear allocation tracking
    memset(allocations, 0, sizeof(allocations));
    allocation_count = 0;

    memory_initialized = true;
    LOG_INF("CMX memory subsystem initialized with %d bytes", CMX_MEMORY_POOL_SIZE);
}

void* cmx_alloc(size_t size) {
    if (!memory_initialized) {
        LOG_ERR("Memory not initialized");
        return nullptr;
    }

    if (size == 0) {
        LOG_WRN("Attempted to allocate 0 bytes");
        return nullptr;
    }

    // Align size to required boundary
    size_t aligned_size = (size + CMX_MEMORY_BLOCK_ALIGN - 1) & ~(CMX_MEMORY_BLOCK_ALIGN - 1);

    k_mutex_lock(&memory_mutex, K_FOREVER);

    void* ptr = sys_heap_alloc(&heap, aligned_size);
    
    if (ptr) {
        // Track allocation for debugging and cleanup
        if (allocation_count < CMX_MAX_BLOCKS) {
            allocations[allocation_count].ptr = ptr;
            allocations[allocation_count].size = aligned_size;
            allocations[allocation_count].in_use = true;
            allocation_count++;
        }
        LOG_DBG("Allocated %d bytes at %p", aligned_size, ptr);
    } else {
        LOG_ERR("Failed to allocate %d bytes", aligned_size);
    }

    k_mutex_unlock(&memory_mutex);
    return ptr;
}

void cmx_free(void* ptr) {
    if (!ptr) {
        LOG_WRN("Attempted to free null pointer");
        return;
    }

    if (!memory_initialized) {
        LOG_ERR("Memory not initialized");
        return;
    }

    k_mutex_lock(&memory_mutex, K_FOREVER);

    // Find and mark allocation as freed
    bool found = false;
    for (int i = 0; i < allocation_count; i++) {
        if (allocations[i].ptr == ptr && allocations[i].in_use) {
            allocations[i].in_use = false;
            found = true;
            LOG_DBG("Freed %d bytes at %p", allocations[i].size, ptr);
            break;
        }
    }

    if (!found) {
        LOG_WRN("Attempted to free untracked pointer %p", ptr);
    }

    // Free from heap
    sys_heap_free(&heap, ptr);

    k_mutex_unlock(&memory_mutex);
}

void* cmx_aligned_alloc(size_t size, size_t alignment) {
    if (!memory_initialized) {
        LOG_ERR("Memory not initialized");
        return nullptr;
    }

    if (alignment == 0 || (alignment & (alignment - 1)) != 0) {
        LOG_ERR("Invalid alignment: %d (must be power of 2)", alignment);
        return nullptr;
    }

    // For simplicity, allocate extra space and align manually
    size_t total_size = size + alignment - 1 + sizeof(void*);
    void* raw_ptr = cmx_alloc(total_size);
    
    if (!raw_ptr) {
        return nullptr;
    }

    // Calculate aligned address
    uintptr_t addr = (uintptr_t)raw_ptr + sizeof(void*);
    uintptr_t aligned_addr = (addr + alignment - 1) & ~(alignment - 1);
    void* aligned_ptr = (void*)aligned_addr;

    // Store original pointer just before aligned pointer
    ((void**)aligned_ptr)[-1] = raw_ptr;

    LOG_DBG("Allocated %d bytes aligned to %d at %p (raw: %p)", 
            size, alignment, aligned_ptr, raw_ptr);

    return aligned_ptr;
}

void cmx_aligned_free(void* ptr) {
    if (!ptr) {
        return;
    }

    // Retrieve original pointer
    void* raw_ptr = ((void**)ptr)[-1];
    cmx_free(raw_ptr);
}

size_t cmx_get_free_memory() {
    if (!memory_initialized) {
        return 0;
    }

    k_mutex_lock(&memory_mutex, K_FOREVER);
    
    // Calculate free memory from heap statistics
    struct sys_heap_stats stats;
    sys_heap_runtime_stats_get(&heap, &stats);
    
    size_t free_bytes = stats.free_bytes;
    
    k_mutex_unlock(&memory_mutex);
    
    return free_bytes;
}

size_t cmx_get_used_memory() {
    if (!memory_initialized) {
        return 0;
    }

    k_mutex_lock(&memory_mutex, K_FOREVER);
    
    struct sys_heap_stats stats;
    sys_heap_runtime_stats_get(&heap, &stats);
    
    size_t used_bytes = stats.allocated_bytes;
    
    k_mutex_unlock(&memory_mutex);
    
    return used_bytes;
}

void cmx_memory_stats() {
    if (!memory_initialized) {
        LOG_ERR("Memory not initialized");
        return;
    }

    k_mutex_lock(&memory_mutex, K_FOREVER);
    
    struct sys_heap_stats stats;
    sys_heap_runtime_stats_get(&heap, &stats);
    
    LOG_INF("Memory Statistics:");
    LOG_INF("  Total size: %d bytes", CMX_MEMORY_POOL_SIZE);
    LOG_INF("  Allocated: %d bytes", stats.allocated_bytes);
    LOG_INF("  Free: %d bytes", stats.free_bytes);
    LOG_INF("  Max allocated: %d bytes", stats.max_allocated_bytes);
    
    // Count active allocations
    int active_allocs = 0;
    for (int i = 0; i < allocation_count; i++) {
        if (allocations[i].in_use) {
            active_allocs++;
        }
    }
    
    LOG_INF("  Active allocations: %d", active_allocs);
    
    k_mutex_unlock(&memory_mutex);
}

void cmx_memory_cleanup() {
    if (!memory_initialized) {
        return;
    }

    k_mutex_lock(&memory_mutex, K_FOREVER);
    
    // Free all tracked allocations
    int freed_count = 0;
    for (int i = 0; i < allocation_count; i++) {
        if (allocations[i].in_use) {
            sys_heap_free(&heap, allocations[i].ptr);
            allocations[i].in_use = false;
            freed_count++;
        }
    }
    
    if (freed_count > 0) {
        LOG_WRN("Cleaned up %d leaked allocations", freed_count);
    }
    
    allocation_count = 0;
    
    k_mutex_unlock(&memory_mutex);
    
    LOG_INF("Memory cleanup completed");
}

} // namespace cmx::platform::zephyr