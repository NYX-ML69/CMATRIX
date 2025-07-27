// cmx_nios2_memory.cpp
// CMatrix Framework Implementation
#include "cmx_nios2_memory.hpp"
#include <cstring>
#include <algorithm>

namespace cmx {
namespace platform {
namespace nios2 {

// Magic values for corruption detection
static constexpr uint32_t BLOCK_MAGIC_USED = 0xDEADBEEF;
static constexpr uint32_t BLOCK_MAGIC_FREE = 0xFEEDFACE;

// Static memory pool - allocated at compile time
static uint8_t memory_pool[MemoryConfig::MEMORY_POOL_SIZE] __attribute__((aligned(8)));
static MemoryBlock block_table[MemoryConfig::MAX_ALLOCATIONS];
static bool memory_initialized = false;
static size_t next_block_index = 0;

/**
 * @brief Round up size to alignment boundary
 */
static inline size_t align_size(size_t size, size_t alignment) {
    return (size + alignment - 1) & ~(alignment - 1);
}

/**
 * @brief Find free block that can satisfy the allocation request
 */
static MemoryBlock* find_free_block(size_t size) {
    for (size_t i = 0; i < MemoryConfig::MAX_ALLOCATIONS; ++i) {
        MemoryBlock* block = &block_table[i];
        if (!block->in_use && block->ptr != nullptr && block->size >= size) {
            return block;
        }
    }
    return nullptr;
}

/**
 * @brief Find block descriptor for given pointer
 */
static MemoryBlock* find_block_by_ptr(void* ptr) {
    for (size_t i = 0; i < MemoryConfig::MAX_ALLOCATIONS; ++i) {
        MemoryBlock* block = &block_table[i];
        if (block->ptr == ptr && block->in_use) {
            return block;
        }
    }
    return nullptr;
}

/**
 * @brief Split a block if it's larger than needed
 */
static void split_block(MemoryBlock* block, size_t needed_size) {
    if (block->size <= needed_size + sizeof(void*)) {
        return; // Not worth splitting
    }
    
    // Find empty slot for new block
    for (size_t i = 0; i < MemoryConfig::MAX_ALLOCATIONS; ++i) {
        MemoryBlock* new_block = &block_table[i];
        if (new_block->ptr == nullptr) {
            // Create new free block from remainder
            new_block->ptr = static_cast<uint8_t*>(block->ptr) + needed_size;
            new_block->size = block->size - needed_size;
            new_block->in_use = false;
            new_block->magic = BLOCK_MAGIC_FREE;
            
            // Adjust original block
            block->size = needed_size;
            break;
        }
    }
}

/**
 * @brief Coalesce adjacent free blocks
 */
static void coalesce_free_blocks() {
    bool found_merge = true;
    
    while (found_merge) {
        found_merge = false;
        
        for (size_t i = 0; i < MemoryConfig::MAX_ALLOCATIONS && !found_merge; ++i) {
            MemoryBlock* block1 = &block_table[i];
            if (block1->in_use || block1->ptr == nullptr) continue;
            
            for (size_t j = i + 1; j < MemoryConfig::MAX_ALLOCATIONS; ++j) {
                MemoryBlock* block2 = &block_table[j];
                if (block2->in_use || block2->ptr == nullptr) continue;
                
                // Check if blocks are adjacent
                uint8_t* end1 = static_cast<uint8_t*>(block1->ptr) + block1->size;
                uint8_t* start2 = static_cast<uint8_t*>(block2->ptr);
                
                if (end1 == start2) {
                    // Merge block2 into block1
                    block1->size += block2->size;
                    block2->ptr = nullptr;
                    block2->size = 0;
                    block2->magic = 0;
                    found_merge = true;
                    break;
                } else if (static_cast<uint8_t*>(block2->ptr) + block2->size == 
                          static_cast<uint8_t*>(block1->ptr)) {
                    // Merge block1 into block2
                    block2->size += block1->size;
                    block1->ptr = nullptr;
                    block1->size = 0;
                    block1->magic = 0;
                    found_merge = true;
                    break;
                }
            }
        }
    }
}

void cmx_memory_init() {
    if (memory_initialized) {
        return;
    }
    
    // Clear block table
    std::memset(block_table, 0, sizeof(block_table));
    
    // Initialize first block to cover entire pool
    block_table[0].ptr = memory_pool;
    block_table[0].size = MemoryConfig::MEMORY_POOL_SIZE;
    block_table[0].in_use = false;
    block_table[0].magic = BLOCK_MAGIC_FREE;
    
    memory_initialized = true;
}

void* cmx_alloc(size_t size) {
    if (!memory_initialized || size == 0) {
        return nullptr;
    }
    
    // Align size to platform requirements
    size_t aligned_size = align_size(size, MemoryConfig::ALIGNMENT);
    
    // Find suitable free block
    MemoryBlock* block = find_free_block(aligned_size);
    if (block == nullptr) {
        return nullptr;
    }
    
    // Split block if necessary
    split_block(block, aligned_size);
    
    // Mark block as used
    block->in_use = true;
    block->magic = BLOCK_MAGIC_USED;
    
    return block->ptr;
}

void cmx_free(void* ptr) {
    if (ptr == nullptr || !memory_initialized) {
        return;
    }
    
    MemoryBlock* block = find_block_by_ptr(ptr);
    if (block == nullptr || block->magic != BLOCK_MAGIC_USED) {
        return; // Invalid pointer or double free
    }
    
    // Mark block as free
    block->in_use = false;
    block->magic = BLOCK_MAGIC_FREE;
    
    // Clear memory for security
    std::memset(block->ptr, 0, block->size);
    
    // Coalesce with adjacent free blocks
    coalesce_free_blocks();
}

void cmx_memory_stats(size_t* used_bytes, size_t* free_bytes, uint8_t* fragmentation) {
    if (!memory_initialized) {
        if (used_bytes) *used_bytes = 0;
        if (free_bytes) *free_bytes = 0;
        if (fragmentation) *fragmentation = 0;
        return;
    }
    
    size_t total_used = 0;
    size_t total_free = 0;
    size_t free_blocks = 0;
    
    for (size_t i = 0; i < MemoryConfig::MAX_ALLOCATIONS; ++i) {
        const MemoryBlock* block = &block_table[i];
        if (block->ptr == nullptr) continue;
        
        if (block->in_use) {
            total_used += block->size;
        } else {
            total_free += block->size;
            free_blocks++;
        }
    }
    
    if (used_bytes) *used_bytes = total_used;
    if (free_bytes) *free_bytes = total_free;
    
    // Simple fragmentation metric: ratio of free blocks to total free memory
    if (fragmentation) {
        if (total_free > 0 && free_blocks > 1) {
            *fragmentation = static_cast<uint8_t>(
                std::min(100UL, (free_blocks * 100) / (total_free / 1024 + 1))
            );
        } else {
            *fragmentation = 0;
        }
    }
}

size_t cmx_memory_defrag() {
    if (!memory_initialized) {
        return 0;
    }
    
    size_t blocks_moved = 0;
    coalesce_free_blocks();
    
    // Simple compaction: move all used blocks to the beginning
    uint8_t* compact_ptr = memory_pool;
    
    for (size_t i = 0; i < MemoryConfig::MAX_ALLOCATIONS; ++i) {
        MemoryBlock* block = &block_table[i];
        if (!block->in_use || block->ptr == nullptr) continue;
        
        if (block->ptr != compact_ptr) {
            // Move block data
            std::memmove(compact_ptr, block->ptr, block->size);
            block->ptr = compact_ptr;
            blocks_moved++;
        }
        
        compact_ptr += block->size;
    }
    
    // Create single large free block from remaining space
    if (compact_ptr < memory_pool + MemoryConfig::MEMORY_POOL_SIZE) {
        // Find or create free block slot
        for (size_t i = 0; i < MemoryConfig::MAX_ALLOCATIONS; ++i) {
            MemoryBlock* block = &block_table[i];
            if (block->ptr == nullptr || (!block->in_use && block->ptr >= compact_ptr)) {
                block->ptr = compact_ptr;
                block->size = (memory_pool + MemoryConfig::MEMORY_POOL_SIZE) - compact_ptr;
                block->in_use = false;
                block->magic = BLOCK_MAGIC_FREE;
                break;
            }
        }
    }
    
    return blocks_moved;
}

bool cmx_memory_check() {
    if (!memory_initialized) {
        return false;
    }
    
    for (size_t i = 0; i < MemoryConfig::MAX_ALLOCATIONS; ++i) {
        const MemoryBlock* block = &block_table[i];
        if (block->ptr == nullptr) continue;
        
        // Check magic numbers
        uint32_t expected_magic = block->in_use ? BLOCK_MAGIC_USED : BLOCK_MAGIC_FREE;
        if (block->magic != expected_magic) {
            return false;
        }
        
        // Check bounds
        if (block->ptr < memory_pool || 
            static_cast<uint8_t*>(block->ptr) + block->size > memory_pool + MemoryConfig::MEMORY_POOL_SIZE) {
            return false;
        }
    }
    
    return true;
}

void* cmx_alloc_aligned(size_t size, size_t alignment) {
    if (!memory_initialized || size == 0 || alignment == 0) {
        return nullptr;
    }
    
    // Ensure alignment is power of 2
    if ((alignment & (alignment - 1)) != 0) {
        return nullptr;
    }
    
    // Allocate extra space for alignment
    size_t total_size = size + alignment - 1;
    void* raw_ptr = cmx_alloc(total_size);
    
    if (raw_ptr == nullptr) {
        return nullptr;
    }
    
    // Calculate aligned address
    uintptr_t addr = reinterpret_cast<uintptr_t>(raw_ptr);
    uintptr_t aligned_addr = (addr + alignment - 1) & ~(alignment - 1);
    
    // If alignment didn't change the address, return as-is
    if (aligned_addr == addr) {
        return raw_ptr;
    }
    
    // For simplicity, free and try again with exact alignment
    // In production, you might want to adjust the block table entry
    cmx_free(raw_ptr);
    
    // Find aligned position in pool and allocate from there
    for (uintptr_t pool_addr = reinterpret_cast<uintptr_t>(memory_pool);
         pool_addr < reinterpret_cast<uintptr_t>(memory_pool + MemoryConfig::MEMORY_POOL_SIZE);
         pool_addr += alignment) {
        
        if ((pool_addr & (alignment - 1)) == 0) {
            // Check if we can allocate at this aligned address
            void* test_ptr = reinterpret_cast<void*>(pool_addr);
            
            // Simple check: try to allocate and see if we get the right address
            void* alloc_ptr = cmx_alloc(size);
            if (alloc_ptr == test_ptr) {
                return alloc_ptr;
            } else if (alloc_ptr != nullptr) {
                cmx_free(alloc_ptr);
            }
        }
    }
    
    return nullptr; // Could not satisfy alignment requirement
}

} // namespace nios2
} // namespace platform
} // namespace cmx

