#include "cmx_platform_abstraction.hpp"

#include <cstdio>
#include <cstring>
#include <cstdlib>

// Include platform-specific headers based on compilation target
#if defined(__ARM_ARCH) || defined(__arm__)
    #include <arm_neon.h>
#elif defined(__riscv)
    // RISC-V specific includes would go here
#elif defined(ESP_PLATFORM)
    #include "freertos/FreeRTOS.h"
    #include "freertos/task.h"
    #include "esp_timer.h"
    #include "esp_log.h"
#endif

// Platform detection macros
#ifndef CMX_PLATFORM_DETECTED
    #define CMX_PLATFORM_GENERIC
#endif

namespace cmx {
namespace platform {

// =============================================================================
// INTERNAL STATE AND UTILITIES
// =============================================================================

namespace {
    // Platform initialization state
    static bool platform_initialized = false;
    
    // Simple memory pool for scratch allocations (fallback implementation)
    static constexpr size_t SCRATCH_POOL_SIZE = 64 * 1024; // 64KB
    static uint8_t scratch_pool[SCRATCH_POOL_SIZE];
    static size_t scratch_pool_offset = 0;
    
    // Platform capabilities (filled during initialization)
    static PlatformCapabilities platform_caps = {};
    
    // Simple timer fallback
    static uint64_t platform_start_time = 0;
    
    // Logging prefix for different levels
    static const char* log_level_prefixes[] = {
        "[DEBUG] ",
        "[INFO]  ",
        "[WARN]  ",
        "[ERROR] ",
        "[CRIT]  "
    };
}

// =============================================================================
// PLATFORM INITIALIZATION - Weak implementations
// =============================================================================

__attribute__((weak))
InitResult init_platform(const void* config) {
    if (platform_initialized) {
        return InitResult::ALREADY_INIT;
    }
    
    // Initialize platform capabilities with generic defaults
    platform_caps.has_dma = false;
    platform_caps.has_cache = false;
    platform_caps.has_fpu = true;  // Assume FPU available
    platform_caps.has_dsp = false;
    platform_caps.has_vector_unit = false;
    platform_caps.cache_line_size = 32;
    platform_caps.timer_resolution_us = 1;
    platform_caps.max_dma_transfer = 0;
    platform_caps.fast_memory_size = SCRATCH_POOL_SIZE;
    platform_caps.total_memory_size = SCRATCH_POOL_SIZE * 2;
    
    // Reset scratch pool
    scratch_pool_offset = 0;
    memset(scratch_pool, 0, SCRATCH_POOL_SIZE);
    
    // Initialize platform-specific timing
    platform_start_time = 0;
    
#ifdef ESP_PLATFORM
    platform_caps.has_dma = true;
    platform_caps.has_cache = true;
    platform_caps.timer_resolution_us = 1;
    ESP_LOGI("CMX_PLATFORM", "Platform abstraction layer initialized (ESP32)");
#elif defined(__ARM_ARCH)
    platform_caps.has_dsp = true;
    #ifdef __ARM_NEON
    platform_caps.has_vector_unit = true;
    #endif
    printf("CMX Platform: Initialized (ARM Cortex)\n");
#elif defined(__riscv)
    printf("CMX Platform: Initialized (RISC-V)\n");
#else
    printf("CMX Platform: Initialized (Generic)\n");
#endif
    
    platform_initialized = true;
    return InitResult::SUCCESS;
}

__attribute__((weak))
bool deinit_platform() {
    if (!platform_initialized) {
        return false;
    }
    
    // Reset scratch pool
    scratch_pool_offset = 0;
    
    platform_initialized = false;
    
#ifdef ESP_PLATFORM
    ESP_LOGI("CMX_PLATFORM", "Platform abstraction layer deinitialized");
#else
    printf("CMX Platform: Deinitialized\n");
#endif
    
    return true;
}

__attribute__((weak))
bool is_platform_initialized() {
    return platform_initialized;
}

__attribute__((weak))
PlatformCapabilities get_platform_capabilities() {
    return platform_caps;
}

// =============================================================================
// TIMING AND PROFILING - Weak implementations
// =============================================================================

__attribute__((weak))
uint64_t get_timestamp_us() {
#ifdef ESP_PLATFORM
    return esp_timer_get_time();
#elif defined(__ARM_ARCH)
    // Use ARM cycle counter if available
    static bool cycle_counter_enabled = false;
    if (!cycle_counter_enabled) {
        // Enable cycle counter (implementation dependent)
        cycle_counter_enabled = true;
    }
    // This is a placeholder - actual implementation would read cycle counter
    return platform_start_time++;
#else
    // Generic fallback using system clock
    static auto start = std::chrono::high_resolution_clock::now();
    auto now = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::microseconds>(now - start).count();
#endif
}

__attribute__((weak))
uint64_t get_timestamp_ns() {
    // Fallback to microsecond precision
    return get_timestamp_us() * 1000;
}

__attribute__((weak))
void sleep_us(uint32_t microseconds) {
#ifdef ESP_PLATFORM
    vTaskDelay(pdMS_TO_TICKS(microseconds / 1000));
#elif defined(__ARM_ARCH)
    // ARM-specific sleep implementation would go here
    // For now, use busy wait
    delay_us(microseconds);
#else
    // Generic sleep
    if (microseconds >= 1000) {
        std::this_thread::sleep_for(std::chrono::microseconds(microseconds));
    } else {
        delay_us(microseconds);
    }
#endif
}

__attribute__((weak))
void delay_us(uint32_t microseconds) {
    // Simple busy-wait delay (very approximate)
    volatile uint32_t cycles = microseconds * 100; // Rough calibration needed
    for (volatile uint32_t i = 0; i < cycles; i++) {
        __asm__ volatile ("nop");
    }
}

// =============================================================================
// MEMORY MANAGEMENT - Weak implementations
// =============================================================================

__attribute__((weak))
void* allocate_scratch(size_t size, const MemoryAttributes& attributes) {
    if (!platform_initialized || size == 0) {
        return nullptr;
    }
    
    // Align size to requested alignment
    size_t aligned_size = (size + attributes.alignment - 1) & ~(attributes.alignment - 1);
    
    // Check if we have enough space in scratch pool
    if (scratch_pool_offset + aligned_size > SCRATCH_POOL_SIZE) {
        log_error("Scratch pool exhausted");
        return nullptr;
    }
    
    // Align pointer to requested alignment
    uintptr_t current_ptr = reinterpret_cast<uintptr_t>(&scratch_pool[scratch_pool_offset]);
    uintptr_t aligned_ptr = (current_ptr + attributes.alignment - 1) & ~(attributes.alignment - 1);
    size_t alignment_offset = aligned_ptr - current_ptr;
    
    // Check if alignment fits
    if (scratch_pool_offset + alignment_offset + size > SCRATCH_POOL_SIZE) {
        log_error("Cannot align scratch allocation");
        return nullptr;
    }
    
    void* result = reinterpret_cast<void*>(aligned_ptr);
    scratch_pool_offset += alignment_offset + aligned_size;
    
    if (attributes.zero_init) {
        memset(result, 0, size);
    }
    
    return result;
}

__attribute__((weak))
void free_scratch(void* ptr) {
    // Simple pool allocator doesn't support individual free
    // Real implementation would track allocations
    (void)ptr;
}

__attribute__((weak))
void* allocate_persistent(size_t size, const MemoryAttributes& attributes) {
    if (!platform_initialized || size == 0) {
        return nullptr;
    }
    
    // Use system malloc for persistent allocations in default implementation
    void* ptr = nullptr;
    
    if (attributes.alignment > sizeof(void*)) {
        // Use aligned allocation if available
        #if defined(_WIN32)
        ptr = _aligned_malloc(size, attributes.alignment);
        #elif defined(__GLIBC__) && __GLIBC__ >= 2 && __GLIBC_MINOR__ >= 16
        ptr = aligned_alloc(attributes.alignment, size);
        #else
        // Fallback: over-allocate and align manually
        void* raw_ptr = malloc(size + attributes.alignment);
        if (raw_ptr) {
            uintptr_t aligned = (reinterpret_cast<uintptr_t>(raw_ptr) + attributes.alignment) 
                              & ~(attributes.alignment - 1);
            ptr = reinterpret_cast<void*>(aligned);
        }
        #endif
    } else {
        ptr = malloc(size);
    }
    
    if (ptr && attributes.zero_init) {
        memset(ptr, 0, size);
    }
    
    return ptr;
}

__attribute__((weak))
void free_persistent(void* ptr) {
    if (ptr) {
        free(ptr);
    }
}

__attribute__((weak))
size_t get_available_scratch_memory() {
    return SCRATCH_POOL_SIZE - scratch_pool_offset;
}

__attribute__((weak))
size_t get_available_persistent_memory() {
    // Cannot easily determine available heap space generically
    return SIZE_MAX; // Assume unlimited for default implementation
}

// =============================================================================
// CACHE MANAGEMENT - Weak implementations
// =============================================================================

__attribute__((weak))
void flush_cache(const void* ptr, size_t size) {
#ifdef __ARM_ARCH
    // ARM cache flush operations would go here
    // Example: __builtin___clear_cache((char*)ptr, (char*)ptr + size);
    (void)ptr;
    (void)size;
#elif defined(ESP_PLATFORM)
    // ESP32 cache operations would be implemented here
    // esp_cache_msync((void*)ptr, size, ESP_CACHE_MSYNC_FLAG_DIR_C2M);
    (void)ptr;
    (void)size;
#else
    // No cache operations needed on generic platforms
    (void)ptr;
    (void)size;
#endif
}

__attribute__((weak))
void invalidate_cache(const void* ptr, size_t size) {
#ifdef __ARM_ARCH
    // ARM cache invalidate operations would go here
    (void)ptr;
    (void)size;
#elif defined(ESP_PLATFORM)
    // ESP32 cache invalidate would be implemented here
    // esp_cache_msync((void*)ptr, size, ESP_CACHE_MSYNC_FLAG_DIR_M2C);
    (void)ptr;
    (void)size;
#else
    // No cache operations needed on generic platforms
    (void)ptr;
    (void)size;
#endif
}

__attribute__((weak))
void prefetch_data(const void* ptr, size_t size) {
#ifdef __ARM_ARCH
    // ARM prefetch instructions would go here
    // Example: __builtin_prefetch(ptr, 0, 3);
    for (size_t i = 0; i < size; i += platform_caps.cache_line_size) {
        __builtin_prefetch(static_cast<const char*>(ptr) + i, 0, 3);
    }
#else
    // Generic prefetch hint
    (void)ptr;
    (void)size;
#endif
}

__attribute__((weak))
void clean_invalidate_cache(const void* ptr, size_t size) {
    flush_cache(ptr, size);
    invalidate_cache(ptr, size);
}

// =============================================================================
// DMA OPERATIONS - Weak implementations
// =============================================================================

__attribute__((weak))
bool dma_transfer_blocking(void* dst, const void* src, size_t size) {
    if (!platform_caps.has_dma) {
        // Fallback to memory copy
        memcpy(dst, src, size);
        return true;
    }
    
    // Platform-specific DMA implementation would go here
    log_debug("DMA transfer not implemented, using memcpy fallback");
    memcpy(dst, src, size);
    return true;
}

__attribute__((weak))
DmaHandle* dma_transfer_async(void* dst, const void* src, size_t size) {
    if (!platform_caps.has_dma) {
        // No async support in fallback
        return nullptr;
    }
    
    // Platform-specific async DMA would be implemented here
    log_debug("Async DMA not implemented");
    return nullptr;
}

__attribute__((weak))
bool dma_wait_completion(DmaHandle* handle, uint32_t timeout_ms) {
    if (!handle) {
        return false;
    }
    
    // Platform-specific DMA wait implementation would go here
    log_debug("DMA wait not implemented");
    return false;
}

__attribute__((weak))
bool is_dma_accessible(const void* ptr) {
    if (!platform_caps.has_dma) {
        return false;
    }
    
    // Platform-specific DMA accessibility check would go here
    // For fallback, assume all memory is accessible
    (void)ptr;
    return true;
}

// =============================================================================
// LOGGING AND DEBUGGING - Weak implementations
// =============================================================================

__attribute__((weak))
void log_message(LogLevel level, const char* message) {
    if (!message) {
        return;
    }
    
    const char* prefix = "";
    if (static_cast<size_t>(level) < sizeof(log_level_prefixes) / sizeof(log_level_prefixes[0])) {
        prefix = log_level_prefixes[static_cast<size_t>(level)];
    }
    
#ifdef ESP_PLATFORM
    esp_log_level_t esp_level = ESP_LOG_INFO;
    switch (level) {
        case LogLevel::DEBUG:   esp_level = ESP_LOG_DEBUG; break;
        case LogLevel::INFO:    esp_level = ESP_LOG_INFO; break;
        case LogLevel::WARNING: esp_level = ESP_LOG_WARN; break;
        case LogLevel::ERROR:   esp_level = ESP_LOG_ERROR; break;
        case LogLevel::CRITICAL: esp_level = ESP_LOG_ERROR; break;
    }
    ESP_LOG_LEVEL(esp_level, "CMX_PLATFORM", "%s", message);
#else
    // Generic logging to stdout/stderr
    FILE* output = (level >= LogLevel::WARNING) ? stderr : stdout;
    fprintf(output, "CMX %s%s\n", prefix, message);
    fflush(output);
#endif
}

// =============================================================================
// POWER MANAGEMENT - Weak implementations
// =============================================================================

__attribute__((weak))
void enter_low_power_mode() {
#ifdef ESP_PLATFORM
    // ESP32 light sleep implementation would go here
    log_debug("Entering low power mode (ESP32)");
#elif defined(__ARM_ARCH)
    // ARM WFI (Wait For Interrupt) instruction
    __asm__ volatile ("wfi");
#else
    log_debug("Low power mode not implemented");
#endif
}

__attribute__((weak))
void exit_low_power_mode() {
    // Usually handled automatically by interrupt/event
    log_debug("Exiting low power mode");
}

__attribute__((weak))
bool set_cpu_frequency(uint32_t frequency_mhz) {
#ifdef ESP_PLATFORM
    // ESP32 frequency scaling would be implemented here
    log_debug("CPU frequency scaling not implemented (ESP32)");
    return false;
#elif defined(__ARM_ARCH)
    // ARM frequency scaling would be implemented here
    log_debug("CPU frequency scaling not implemented (ARM)");
    return false;
#else
    log_debug("CPU frequency scaling not available");
    return false;
#endif
}

// =============================================================================
// INTERRUPT MANAGEMENT - Weak implementations
// =============================================================================

__attribute__((weak))
uint32_t disable_interrupts() {
#ifdef __ARM_ARCH
    uint32_t primask;
    __asm__ volatile ("mrs %0, primask" : "=r" (primask));
    __asm__ volatile ("cpsid i" ::: "memory");
    return primask;
#elif defined(__riscv)
    // RISC-V interrupt disable would go here
    return 0;
#else
    // Generic platforms may not support interrupt control
    return 0;
#endif
}

__attribute__((weak))
void restore_interrupts(uint32_t state) {
#ifdef __ARM_ARCH
    __asm__ volatile ("msr primask, %0" :: "r" (state) : "memory");
#elif defined(__riscv)
    // RISC-V interrupt restore would go here
    (void)state;
#else
    // Generic platforms may not support interrupt control
    (void)state;
#endif
}

// =============================================================================
// ATOMIC OPERATIONS - Weak implementations
// =============================================================================

__attribute__((weak))
uint32_t atomic_load_32(const volatile uint32_t* ptr) {
#ifdef __ARM_ARCH
    uint32_t result;
    __asm__ volatile ("ldr %0, %1" : "=r" (result) : "m" (*ptr) : "memory");
    return result;
#elif defined(__riscv)
    // RISC-V atomic load would go here
    return *ptr;
#else
    // Generic atomic operations
    return __atomic_load_n(ptr, __ATOMIC_SEQ_CST);
#endif
}

__attribute__((weak))
void atomic_store_32(volatile uint32_t* ptr, uint32_t value) {
#ifdef __ARM_ARCH
    __asm__ volatile ("str %1, %0" : "=m" (*ptr) : "r" (value) : "memory");
#elif defined(__riscv)
    // RISC-V atomic store would go here
    *ptr = value;
#else
    // Generic atomic operations
    __atomic_store_n(ptr, value, __ATOMIC_SEQ_CST);
#endif
}

__attribute__((weak))
bool atomic_compare_swap_32(volatile uint32_t* ptr, uint32_t expected, uint32_t desired) {
#ifdef __ARM_ARCH
    uint32_t result;
    uint32_t temp;
    
    __asm__ volatile (
        "1: ldrex %0, %2\n"
        "   cmp %0, %3\n"
        "   bne 2f\n"
        "   strex %1, %4, %2\n"
        "   cmp %1, #0\n"
        "   bne 1b\n"
        "2:"
        : "=&r" (result), "=&r" (temp), "+m" (*ptr)
        : "r" (expected), "r" (desired)
        : "cc", "memory"
    );
    
    return result == expected;
#elif defined(__riscv)
    // RISC-V atomic compare and swap would go here
    return __atomic_compare_exchange_n(ptr, &expected, desired, false, 
                                       __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);
#else
    // Generic atomic operations
    return __atomic_compare_exchange_n(ptr, &expected, desired, false, 
                                       __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);
#endif
}

// =============================================================================
// UTILITY FUNCTIONS
// =============================================================================

namespace {
    /**
     * @brief Reset scratch pool (internal utility)
     */
    void reset_scratch_pool() {
        scratch_pool_offset = 0;
    }
    
    /**
     * @brief Get scratch pool usage statistics
     */
    void get_scratch_stats(size_t& used, size_t& total, float& utilization) {
        used = scratch_pool_offset;
        total = SCRATCH_POOL_SIZE;
        utilization = (total > 0) ? static_cast<float>(used) / total * 100.0f : 0.0f;
    }
}

// =============================================================================
// PLATFORM-SPECIFIC INITIALIZATION HOOKS
// =============================================================================

namespace platform_hooks {
    /**
     * @brief Platform-specific early initialization
     * Called before main platform initialization
     */
    __attribute__((weak))
    void early_init() {
        // Platform-specific early setup can override this
    }
    
    /**
     * @brief Platform-specific late initialization  
     * Called after main platform initialization
     */
    __attribute__((weak))
    void late_init() {
        // Platform-specific late setup can override this
    }
    
    /**
     * @brief Platform-specific shutdown preparation
     * Called before platform deinitialization
     */
    __attribute__((weak))
    void pre_shutdown() {
        // Platform-specific shutdown preparation can override this
    }
}

// =============================================================================
// DEBUG AND DIAGNOSTICS
// =============================================================================

#ifdef CMX_PLATFORM_DEBUG
/**
 * @brief Print platform diagnostics (debug builds only)
 */
void print_platform_diagnostics() {
    if (!platform_initialized) {
        log_warning("Platform not initialized");
        return;
    }
    
    char buffer[256];
    
    snprintf(buffer, sizeof(buffer), "=== CMX Platform Diagnostics ===");
    log_info(buffer);
    
    snprintf(buffer, sizeof(buffer), "DMA Support: %s", 
             platform_caps.has_dma ? "Yes" : "No");
    log_info(buffer);
    
    snprintf(buffer, sizeof(buffer), "Cache Support: %s", 
             platform_caps.has_cache ? "Yes" : "No");
    log_info(buffer);
    
    snprintf(buffer, sizeof(buffer), "FPU Support: %s", 
             platform_caps.has_fpu ? "Yes" : "No");
    log_info(buffer);
    
    snprintf(buffer, sizeof(buffer), "Vector Unit: %s", 
             platform_caps.has_vector_unit ? "Yes" : "No");
    log_info(buffer);
    
    snprintf(buffer, sizeof(buffer), "Cache Line Size: %u bytes", 
             platform_caps.cache_line_size);
    log_info(buffer);
    
    snprintf(buffer, sizeof(buffer), "Timer Resolution: %u us", 
             platform_caps.timer_resolution_us);
    log_info(buffer);
    
    size_t used, total;
    float utilization;
    get_scratch_stats(used, total, utilization);
    snprintf(buffer, sizeof(buffer), "Scratch Pool: %zu/%zu bytes (%.1f%%)", 
             used, total, utilization);
    log_info(buffer);
    
    snprintf(buffer, sizeof(buffer), "================================");
    log_info(buffer);
}
#endif

} // namespace platform
} // namespace cmx

// =============================================================================
// C INTERFACE FOR COMPATIBILITY
// =============================================================================

extern "C" {
    // C-style interface functions for integration with C code
    
    int cmx_platform_init(const void* config) {
        return static_cast<int>(cmx::platform::init_platform(config));
    }
    
    int cmx_platform_deinit(void) {
        return cmx::platform::deinit_platform() ? 0 : -1;
    }
    
    int cmx_platform_is_initialized(void) {
        return cmx::platform::is_platform_initialized() ? 1 : 0;
    }
    
    uint64_t cmx_platform_get_timestamp_us(void) {
        return cmx::platform::get_timestamp_us();
    }
    
    void* cmx_platform_allocate_scratch(size_t size) {
        return cmx::platform::allocate_scratch(size);
    }
    
    void cmx_platform_free_scratch(void* ptr) {
        cmx::platform::free_scratch(ptr);
    }
    
    void cmx_platform_log_debug(const char* message) {
        cmx::platform::log_debug(message);
    }
    
    void cmx_platform_flush_cache(const void* ptr, size_t size) {
        cmx::platform::flush_cache(ptr, size);
    }
}