#include "cmx_cortex_m_memory.hpp"
#include <cstring>

#ifdef __arm__
#include "cmsis_gcc.h"
#endif

namespace cmx {
namespace platform {
namespace cortex_m {

// Static memory arena for tensor operations
static uint8_t* tensor_arena = nullptr;
static size_t tensor_arena_size = 0;
static size_t tensor_arena_used = 0;
static bool memory_initialized = false;

// Memory region registry
static constexpr uint8_t MAX_MEMORY_REGIONS = 8;
static MemoryRegion memory_regions[MAX_MEMORY_REGIONS];
static uint8_t num_memory_regions = 0;

// Default memory regions for common Cortex-M layouts
static uint8_t static_tensor_arena[65536] __attribute__((aligned(32)));

bool Memory::init(size_t arena_size) {
    if (memory_initialized) {
        return true;
    }
    
    // Use static arena if size fits, otherwise fail
    if (arena_size > sizeof(static_tensor_arena)) {
        return false;
    }
    
    tensor_arena = static_tensor_arena;
    tensor_arena_size = arena_size;
    tensor_arena_used = 0;
    
    // Clear the arena
    fast_set(tensor_arena, 0, tensor_arena_size);
    
    // Register default memory regions
    MemoryRegion sram_region = {
        .base_address = tensor_arena,
        .size = tensor_arena_size,
        .is_cacheable = false,
        .is_dma_coherent = true,
        .name = "SRAM_TENSOR_ARENA"
    };
    register_memory_region(sram_region);
    
    memory_initialized = true;
    return true;
}

void Memory::deinit() {
    if (!memory_initialized) {
        return;
    }
    
    tensor_arena = nullptr;
    tensor_arena_size = 0;
    tensor_arena_used = 0;
    num_memory_regions = 0;
    memory_initialized = false;
}

void* Memory::get_tensor_arena() {
    return memory_initialized ? tensor_arena : nullptr;
}

size_t Memory::get_tensor_arena_size() {
    return memory_initialized ? tensor_arena_size : 0;
}

void* Memory::allocate_aligned(size_t size, MemoryAlignment alignment) {
    if (!memory_initialized || size == 0) {
        return nullptr;
    }
    
    size_t align_value = static_cast<size_t>(alignment);
    
    // Align current position
    void* current_pos = tensor_arena + tensor_arena_used;
    void* aligned_pos = align_pointer(current_pos, align_value);
    
    // Calculate aligned offset from arena start
    size_t aligned_offset = static_cast<uint8_t*>(aligned_pos) - tensor_arena;
    
    // Check if allocation fits
    if (aligned_offset + size > tensor_arena_size) {
        return nullptr;
    }
    
    // Update used space
    tensor_arena_used = aligned_offset + size;
    
    return aligned_pos;
}

void Memory::reset_arena() {
    if (!memory_initialized) {
        return;
    }
    tensor_arena_used = 0;
}

size_t Memory::get_free_arena_space() {
    if (!memory_initialized) {
        return 0;
    }
    return tensor_arena_size - tensor_arena_used;
}

void Memory::fast_copy(void* dest, const void* src, size_t size) {
    if (!dest || !src || size == 0) {
        return;
    }
    
    uint8_t* d = static_cast<uint8_t*>(dest);
    const uint8_t* s = static_cast<const uint8_t*>(src);
    
    // Optimize for 4-byte aligned copies when possible
    if (is_aligned(dest, 4) && is_aligned(src, 4) && (size >= 4)) {
        uint32_t* d32 = static_cast<uint32_t*>(dest);
        const uint32_t* s32 = static_cast<const uint32_t*>(src);
        size_t words = size / 4;
        
        // Copy 4 bytes at a time
        for (size_t i = 0; i < words; i++) {
            d32[i] = s32[i];
        }
        
        // Handle remaining bytes
        size_t remaining = size % 4;
        if (remaining > 0) {
            d += words * 4;
            s += words * 4;
            for (size_t i = 0; i < remaining; i++) {
                d[i] = s[i];
            }
        }
    } else {
        // Fallback to byte-wise copy
        std::memcpy(dest, src, size);
    }
}

void Memory::fast_set(void* dest, uint8_t value, size_t size) {
    if (!dest || size == 0) {
        return;
    }
    
    uint8_t* d = static_cast<uint8_t*>(dest);
    
    // Optimize for 4-byte aligned sets when possible
    if (is_aligned(dest, 4) && (size >= 4)) {
        uint32_t value32 = (static_cast<uint32_t>(value) << 24) |
                          (static_cast<uint32_t>(value) << 16) |
                          (static_cast<uint32_t>(value) << 8) |
                          static_cast<uint32_t>(value);
        
        uint32_t* d32 = static_cast<uint32_t*>(dest);
        size_t words = size / 4;
        
        // Set 4 bytes at a time
        for (size_t i = 0; i < words; i++) {
            d32[i] = value32;
        }
        
        // Handle remaining bytes
        size_t remaining = size % 4;
        if (remaining > 0) {
            d += words * 4;
            for (size_t i = 0; i < remaining; i++) {
                d[i] = value;
            }
        }
    } else {
        // Fallback to standard memset
        std::memset(dest, value, size);
    }
}

int Memory::fast_compare(const void* ptr1, const void* ptr2, size_t size) {
    if (!ptr1 || !ptr2 || size == 0) {
        return 0;
    }
    
    const uint8_t* p1 = static_cast<const uint8_t*>(ptr1);
    const uint8_t* p2 = static_cast<const uint8_t*>(ptr2);
    
    // Optimize for 4-byte aligned comparison when possible
    if (is_aligned(ptr1, 4) && is_aligned(ptr2, 4) && (size >= 4)) {
        const uint32_t* p1_32 = static_cast<const uint32_t*>(ptr1);
        const uint32_t* p2_32 = static_cast<const uint32_t*>(ptr2);
        size_t words = size / 4;
        
        // Compare 4 bytes at a time
        for (size_t i = 0; i < words; i++) {
            if (p1_32[i] != p2_32[i]) {
                return (p1_32[i] < p2_32[i]) ? -1 : 1;
            }
        }
        
        // Handle remaining bytes
        size_t remaining = size % 4;
        if (remaining > 0) {
            p1 += words * 4;
            p2 += words * 4;
            return std::memcmp(p1, p2, remaining);
        }
        
        return 0;
    } else {
        // Fallback to standard memcmp
        return std::memcmp(ptr1, ptr2, size);
    }
}

bool Memory::register_memory_region(const MemoryRegion& region) {
    if (num_memory_regions >= MAX_MEMORY_REGIONS || !region.base_address || region.size == 0) {
        return false;
    }
    
    memory_regions[num_memory_regions] = region;
    num_memory_regions++;
    return true;
}

const MemoryRegion* Memory::get_memory_region(const char* name) {
    if (!name) {
        return nullptr;
    }
    
    for (uint8_t i = 0; i < num_memory_regions; i++) {
        if (memory_regions[i].name && std::strcmp(memory_regions[i].name, name) == 0) {
            return &memory_regions[i];
        }
    }
    
    return nullptr;
}

bool Memory::is_valid_address(const void* address, size_t size) {
    if (!address || size == 0) {
        return false;
    }
    
    uintptr_t addr = reinterpret_cast<uintptr_t>(address);
    uintptr_t end_addr = addr + size - 1;
    
    // Check against registered memory regions
    for (uint8_t i = 0; i < num_memory_regions; i++) {
        uintptr_t region_start = reinterpret_cast<uintptr_t>(memory_regions[i].base_address);
        uintptr_t region_end = region_start + memory_regions[i].size - 1;
        
        if (addr >= region_start && end_addr <= region_end) {
            return true;
        }
    }
    
    return false;
}

void Memory::get_memory_stats(size_t* total_arena_size, 
                             size_t* used_arena_size, 
                             size_t* free_arena_size) {
    if (total_arena_size) {
        *total_arena_size = memory_initialized ? tensor_arena_size : 0;
    }
    if (used_arena_size) {
        *used_arena_size = memory_initialized ? tensor_arena_used : 0;
    }
    if (free_arena_size) {
        *free_arena_size = memory_initialized ? (tensor_arena_size - tensor_arena_used) : 0;
    }
}

void Memory::flush_cache(const void* address, size_t size) {
    if (!address || size == 0) {
        return;
    }
    
#ifdef __arm__
    // ARM Cortex-M with cache support
    #if (__CORTEX_M >= 7U)
    if (SCB->CCR & SCB_CCR_DC_Msk) {
        // Data cache is enabled, perform cache maintenance
        uintptr_t addr = reinterpret_cast<uintptr_t>(address);
        uintptr_t end_addr = addr + size;
        
        // Align to cache line boundaries
        addr &= ~(32U - 1U); // Assume 32-byte cache lines
        
        __DSB();
        while (addr < end_addr) {
            SCB->DCCMVAC = addr;
            addr += 32;
        }
        __DSB();
        __ISB();
    }
    #endif
#endif
}

void Memory::invalidate_cache(const void* address, size_t size) {
    if (!address || size == 0) {
        return;
    }
    
#ifdef __arm__
    // ARM Cortex-M with cache support
    #if (__CORTEX_M >= 7U)
    if (SCB->CCR & SCB_CCR_DC_Msk) {
        // Data cache is enabled, perform cache maintenance
        uintptr_t addr = reinterpret_cast<uintptr_t>(address);
        uintptr_t end_addr = addr + size;
        
        // Align to cache line boundaries
        addr &= ~(32U - 1U); // Assume 32-byte cache lines
        
        __DSB();
        while (addr < end_addr) {
            SCB->DCIMVAC = addr;
            addr += 32;
        }
        __DSB();
        __ISB();
    }
    #endif
#endif
}

bool Memory::is_initialized() {
    return memory_initialized;
}

void* Memory::align_pointer(void* ptr, size_t alignment) {
    if (!ptr || alignment == 0 || (alignment & (alignment - 1)) != 0) {
        return ptr; // Invalid alignment (not power of 2)
    }
    
    uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
    uintptr_t aligned_addr = (addr + alignment - 1) & ~(alignment - 1);
    return reinterpret_cast<void*>(aligned_addr);
}

bool Memory::is_aligned(const void* ptr, size_t alignment) {
    if (!ptr || alignment == 0) {
        return false;
    }
    
    uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
    return (addr & (alignment - 1)) == 0;
}

// MemoryBuffer implementation
MemoryBuffer::MemoryBuffer(size_t size, MemoryAlignment alignment) 
    : buffer_(nullptr), size_(size) {
    
    if (size > 0) {
        buffer_ = Memory::allocate_aligned(size, alignment);
        if (buffer_) {
            // Initialize buffer to zero
            Memory::fast_set(buffer_, 0, size);
        }
    }
}

} // namespace cortex_m
} // namespace platform
} // namespace cmx