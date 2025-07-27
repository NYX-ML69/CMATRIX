#include "cmx_xtensa_memory.hpp"
#include "cmx_xtensa_port.hpp"
#include <cstring>
#include <algorithm>
#include <cstdlib>

namespace cmx::platform::xtensa {

// Memory pool constants
static constexpr size_t FAST_RAM_SIZE = 128 * 1024;    // 128KB internal SRAM
static constexpr size_t SLOW_RAM_SIZE = 2 * 1024 * 1024; // 2MB external PSRAM
static constexpr size_t DMA_RAM_SIZE = 64 * 1024;      // 64KB DMA-capable memory
static constexpr size_t INSTRUCTION_SIZE = 256 * 1024;  // 256KB for instructions

static constexpr size_t MIN_BLOCK_SIZE = 16;
static constexpr size_t HEADER_SIZE = sizeof(size_t) * 2; // Size + flags

// Memory block header
struct BlockHeader {
    size_t size;
    uint32_t flags;
    BlockHeader* next;
    BlockHeader* prev;
};

// Memory pool structure
struct MemoryPool {
    MemoryPoolType type;
    void* base_address;
    size_t total_size;
    size_t alignment;
    BlockHeader* free_list;
    MemoryStats stats;
    bool initialized;
};

static MemoryPool g_memory_pools[4];
static bool g_memory_initialized = false;

// External memory regions (would be defined in linker script)
extern "C" {
    extern char _heap_start[];
    extern char _heap_end[];
    extern char _dma_heap_start[];
    extern char _dma_heap_end[];
}

static void init_memory_pool(MemoryPool& pool, MemoryPoolType type, 
                           void* base, size_t size, size_t alignment) {
    pool.type = type;
    pool.base_address = base;
    pool.total_size = size;
    pool.alignment = alignment;
    pool.initialized = true;
    
    // Initialize free list with single large block
    pool.free_list = static_cast<BlockHeader*>(base);
    pool.free_list->size = size - sizeof(BlockHeader);
    pool.free_list->flags = 0;
    pool.free_list->next = nullptr;
    pool.free_list->prev = nullptr;
    
    // Initialize statistics
    pool.stats.total_bytes = size;
    pool.stats.used_bytes = sizeof(BlockHeader);
    pool.stats.free_bytes = size - sizeof(BlockHeader);
    pool.stats.largest_free_block = pool.free_list->size;
    pool.stats.allocations = 0;
    pool.stats.deallocations = 0;
    pool.stats.allocation_failures = 0;
    pool.stats.fragmentation_ratio = 0.0f;
}

void cmx_memory_init() {
    if (g_memory_initialized) {
        return;
    }
    
    // Initialize memory pools
    // Note: In real implementation, these addresses would come from linker script
    
    // Fast RAM pool (internal SRAM)
    static uint8_t fast_ram_buffer[FAST_RAM_SIZE] __attribute__((aligned(32)));
    init_memory_pool(g_memory_pools[0], MemoryPoolType::FAST_RAM, 
                    fast_ram_buffer, FAST_RAM_SIZE, 4);
    
    // Slow RAM pool (external PSRAM)
    static uint8_t slow_ram_buffer[SLOW_RAM_SIZE] __attribute__((aligned(32)));
    init_memory_pool(g_memory_pools[1], MemoryPoolType::SLOW_RAM,
                    slow_ram_buffer, SLOW_RAM_SIZE, 4);
    
    // DMA-capable memory pool
    static uint8_t dma_ram_buffer[DMA_RAM_SIZE] __attribute__((aligned(32)));
    init_memory_pool(g_memory_pools[2], MemoryPoolType::DMA_CAPABLE,
                    dma_ram_buffer, DMA_RAM_SIZE, 32);
    
    // Instruction memory pool
    static uint8_t instruction_buffer[INSTRUCTION_SIZE] __attribute__((aligned(32)));
    init_memory_pool(g_memory_pools[3], MemoryPoolType::INSTRUCTION,
                    instruction_buffer, INSTRUCTION_SIZE, 4);
    
    g_memory_initialized = true;
    cmx_log("MEMORY: Initialized 4 memory pools");
}

static MemoryPool* get_pool(MemoryPoolType type) {
    int index = static_cast<int>(type);
    if (index >= 0 && index < 4 && g_memory_pools[index].initialized) {
        return &g_memory_pools[index];
    }
    return nullptr;
}

static size_t align_size(size_t size, size_t alignment) {
    return (size + alignment - 1) & ~(alignment - 1);
}

static void* allocate_from_pool(MemoryPool& pool, size_t size, uint32_t flags) {
    CMX_CRITICAL_SECTION();
    
    size_t aligned_size = align_size(size, pool.alignment);
    size_t total_size = aligned_size + sizeof(BlockHeader);
    
    // Find suitable free block
    BlockHeader* current = pool.free_list;
    BlockHeader* best_fit = nullptr;
    
    while (current) {
        if (current->size >= total_size) {
            if (!best_fit || current->size < best_fit->size) {
                best_fit = current;
            }
        }
        current = current->next;
    }
    
    if (!best_fit) {
        pool.stats.allocation_failures++;
        return nullptr;
    }
    
    // Remove from free list
    if (best_fit->prev) {
        best_fit->prev->next = best_fit->next;
    } else {
        pool.free_list = best_fit->next;
    }
    
    if (best_fit->next) {
        best_fit->next->prev = best_fit->prev;
    }
    
    // Split block if necessary
    if (best_fit->size > total_size + MIN_BLOCK_SIZE) {
        BlockHeader* new_block = reinterpret_cast<BlockHeader*>(
            reinterpret_cast<uint8_t*>(best_fit) + total_size);
        
        new_block->size = best_fit->size - total_size;
        new_block->flags = 0;
        new_block->next = pool.free_list;
        new_block->prev = nullptr;
        
        if (pool.free_list) {
            pool.free_list->prev = new_block;
        }
        pool.free_list = new_block;
        
        best_fit->size = aligned_size;
    }
    
    best_fit->flags = flags | 0x80000000; // Mark as allocated
    
    void* user_ptr = reinterpret_cast<uint8_t*>(best_fit) + sizeof(BlockHeader);
    
    // Zero memory if requested
    if (flags & ZERO_MEMORY) {
        std::memset(user_ptr, 0, aligned_size);
    }
    
    // Update statistics
    pool.stats.allocations++;
    pool.stats.used_bytes += total_size;
    pool.stats.free_bytes -= total_size;
    
    return user_ptr;
}

void* cmx_alloc(size_t size) {
    if (!g_memory_initialized) {
        cmx_memory_init();
    }
    
    if (size == 0) {
        return nullptr;
    }
    
    // Default allocation from fast RAM for small allocations, slow RAM for large
    MemoryPoolType pool_type = (size <= 1024) ? 
        MemoryPoolType::FAST_RAM : MemoryPoolType::SLOW_RAM;
    
    return cmx_alloc_pool(size, pool_type, 0);
}

void cmx_free(void* ptr) {
    if (!ptr) {
        return;
    }
    
    CMX_CRITICAL_SECTION();
    
    // Get block header
    BlockHeader* block = reinterpret_cast<BlockHeader*>(
        reinterpret_cast<uint8_t*>(ptr) - sizeof(BlockHeader));
    
    if (!(block->flags & 0x80000000)) {
        // Double free or invalid pointer
        return;
    }
    
    // Find which pool this block belongs to
    MemoryPool* pool = nullptr;
    for (int i = 0; i < 4; i++) {
        uintptr_t pool_start = reinterpret_cast<uintptr_t>(g_memory_pools[i].base_address);
        uintptr_t pool_end = pool_start + g_memory_pools[i].total_size;
        uintptr_t block_addr = reinterpret_cast<uintptr_t>(block);
        
        if (block_addr >= pool_start && block_addr < pool_end) {
            pool = &g_memory_pools[i];
            break;
        }
    }
    
    if (!pool) {
        return; // Invalid pointer
    }
    
    // Clear allocated flag
    block->flags &= ~0x80000000;
    
    // Add to free list
    block->next = pool->free_list;
    block->prev = nullptr;
    
    if (pool->free_list) {
        pool->free_list->prev = block;
    }
    pool->free_list = block;
    
    // Update statistics
    pool->stats.deallocations++;
    size_t total_size = block->size + sizeof(BlockHeader);
    pool->stats.used_bytes -= total_size;
    pool->stats.free_bytes += total_size;
    
    // TODO: Coalesce adjacent free blocks
}

void* cmx_alloc_flags(size_t size, uint32_t flags) {
    MemoryPoolType pool_type = MemoryPoolType::FAST_RAM;
    
    // Choose pool based on alignment requirements
    if (flags & (ALIGN_16_BYTE | ALIGN_32_BYTE)) {
        pool_type = MemoryPoolType::DMA_CAPABLE;
    }
    
    return cmx_alloc_pool(size, pool_type, flags);
}

void* cmx_alloc_pool(size_t size, MemoryPoolType pool_type, uint32_t flags) {
    if (!g_memory_initialized) {
        cmx_memory_init();
    }
    
    MemoryPool* pool = get_pool(pool_type);
    if (!pool) {
        return nullptr;
    }
    
    return allocate_from_pool(*pool, size, flags);
}

void* cmx_realloc(void* ptr, size_t new_size) {
    if (!ptr) {
        return cmx_alloc(new_size);
    }
    
    if (new_size == 0) {
        cmx_free(ptr);
        return nullptr;
    }
    
    size_t old_size = cmx_get_block_size(ptr);
    if (old_size >= new_size) {
        return ptr; // No need to reallocate
    }
    
    void* new_ptr = cmx_alloc(new_size);
    if (new_ptr) {
        std::memcpy(new_ptr, ptr, old_size);
        cmx_free(ptr);
    }
    
    return new_ptr;
}

void* cmx_alloc_aligned(size_t size, size_t alignment) {
    // Simple implementation - in real code, would be more efficient
    size_t total_size = size + alignment - 1 + sizeof(void*);
    void* raw_ptr = cmx_alloc(total_size);
    
    if (!raw_ptr) {
        return nullptr;
    }
    
    uintptr_t aligned_addr = (reinterpret_cast<uintptr_t>(raw_ptr) + 
                             sizeof(void*) + alignment - 1) & ~(alignment - 1);
    void* aligned_ptr = reinterpret_cast<void*>(aligned_addr);
    
    // Store original pointer for free
    *(reinterpret_cast<void**>(aligned_ptr) - 1) = raw_ptr;
    
    return aligned_ptr;
}

void cmx_free_aligned(void* ptr) {
    if (ptr) {
        void* raw_ptr = *(reinterpret_cast<void**>(ptr) - 1);
        cmx_free(raw_ptr);
    }
}

size_t cmx_get_block_size(void* ptr) {
    if (!ptr) {
        return 0;
    }
    
    BlockHeader* block = reinterpret_cast<BlockHeader*>(
        reinterpret_cast<uint8_t*>(ptr) - sizeof(BlockHeader));
    
    if (!(block->flags & 0x80000000)) {
        return 0; // Not allocated
    }
    
    return block->size;
}

const MemoryStats& cmx_get_memory_stats(MemoryPoolType pool_type) {
    static MemoryStats empty_stats = {};
    
    MemoryPool* pool = get_pool(pool_type);
    if (!pool) {
        return empty_stats;
    }
    
    return pool->stats;
}

const MemoryStats& cmx_get_total_memory_stats() {
    static MemoryStats total_stats = {};
    
    total_stats = {};
    for (int i = 0; i < 4; i++) {
        if (g_memory_pools[i].initialized) {
            const auto& pool_stats = g_memory_pools[i].stats;
            total_stats.total_bytes += pool_stats.total_bytes;
            total_stats.used_bytes += pool_stats.used_bytes;
            total_stats.free_bytes += pool_stats.free_bytes;
            total_stats.allocations += pool_stats.allocations;
            total_stats.deallocations += pool_stats.deallocations;
            total_stats.allocation_failures += pool_stats.allocation_failures;
            
            if (pool_stats.largest_free_block > total_stats.largest_free_block) {
                total_stats.largest_free_block = pool_stats.largest_free_block;
            }
        }
    }
    
    // Calculate fragmentation ratio
    if (total_stats.free_bytes > 0) {
        total_stats.fragmentation_ratio = 1.0f - 
            (static_cast<float>(total_stats.largest_free_block) / total_stats.free_bytes);
    }
    
    return total_stats;
}

const MemoryPoolInfo& cmx_get_pool_info(MemoryPoolType pool_type) {
    static const MemoryPoolInfo pool_infos[4] = {
        {MemoryPoolType::FAST_RAM, "Fast RAM", nullptr, FAST_RAM_SIZE, MIN_BLOCK_SIZE, 4, false, true},
        {MemoryPoolType::SLOW_RAM, "Slow RAM", nullptr, SLOW_RAM_SIZE, MIN_BLOCK_SIZE, 4, false, false},
        {MemoryPoolType::DMA_CAPABLE, "DMA RAM", nullptr, DMA_RAM_SIZE, MIN_BLOCK_SIZE, 32, true, true},
        {MemoryPoolType::INSTRUCTION, "Instruction", nullptr, INSTRUCTION_SIZE, MIN_BLOCK_SIZE, 4, false, true}
    };
    
    int index = static_cast<int>(pool_type);
    if (index >= 0 && index < 4) {
        return pool_infos[index];
    }
    
    return pool_infos[0]; // Fallback
}

// Stack Allocator Implementation
StackAllocator::StackAllocator(size_t size) 
    : total_size_(size), current_offset_(0) {
    buffer_ = cmx_alloc(size);
}

StackAllocator::~StackAllocator() {
    if (buffer_) {
        cmx_free(buffer_);
    }
}

void* StackAllocator::alloc(size_t size, size_t alignment) {
    if (!buffer_ || size == 0) {
        return nullptr;
    }
    
    size_t aligned_offset = align_size(current_offset_, alignment);
    if (aligned_offset + size > total_size_) {
        return nullptr; // Out of space
    }
    
    void* ptr = reinterpret_cast<uint8_t*>(buffer_) + aligned_offset;
    current_offset_ = aligned_offset + size;
    
    return ptr;
}

void StackAllocator::reset() {
    current_offset_ = 0;
}

size_t StackAllocator::get_used() const {
    return current_offset_;
}

size_t StackAllocator::get_remaining() const {
    return total_size_ - current_offset_;
}

} // namespace cmx::platform::xtensa