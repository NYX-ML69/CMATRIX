#pragma once

#include <cstdint>
#include <cstddef>

namespace cmx {
namespace platform {

/**
 * @brief Platform initialization result codes
 */
enum class InitResult : uint8_t {
    SUCCESS = 0,        // Platform initialized successfully
    ALREADY_INIT,       // Platform already initialized
    HARDWARE_ERROR,     // Hardware initialization failed
    MEMORY_ERROR,       // Memory allocation failed
    INVALID_CONFIG      // Invalid configuration parameters
};

/**
 * @brief Log levels for platform logging
 */
enum class LogLevel : uint8_t {
    DEBUG = 0,
    INFO,
    WARNING,
    ERROR,
    CRITICAL
};

/**
 * @brief Memory allocation attributes
 */
struct MemoryAttributes {
    bool dma_capable = false;       // Memory must be DMA accessible
    bool cache_aligned = true;      // Align to cache line boundaries
    bool zero_init = false;         // Zero-initialize allocated memory
    uint32_t alignment = 32;        // Memory alignment in bytes
    uint32_t timeout_ms = 0;        // Allocation timeout (0 = no timeout)
};

/**
 * @brief Platform-specific opaque handle types
 * These are forward declarations - actual definitions are platform-specific
 */
struct PlatformHandle;
struct TimerHandle;
struct DmaHandle;

/**
 * @brief Platform capabilities structure
 */
struct PlatformCapabilities {
    bool has_dma;                   // DMA controller available
    bool has_cache;                 // Cache memory available
    bool has_fpu;                   // Floating point unit available
    bool has_dsp;                   // DSP instructions available
    bool has_vector_unit;           // Vector processing unit available
    uint32_t cache_line_size;       // Cache line size in bytes
    uint32_t timer_resolution_us;   // Timer resolution in microseconds
    size_t max_dma_transfer;        // Maximum DMA transfer size
    size_t fast_memory_size;        // Size of fastest memory region
    size_t total_memory_size;       // Total available memory
};

// =============================================================================
// PLATFORM INITIALIZATION
// =============================================================================

/**
 * @brief Initialize the platform abstraction layer
 * 
 * This function must be called before any other platform functions.
 * It initializes hardware-specific components like timers, DMA controllers,
 * memory management units, and other platform-specific resources.
 * 
 * @param config Platform-specific configuration data (can be nullptr)
 * @return InitResult indicating success or failure reason
 */
InitResult init_platform(const void* config = nullptr);

/**
 * @brief Deinitialize the platform abstraction layer
 * 
 * Cleans up all platform resources and resets hardware to a safe state.
 * Should be called during system shutdown or before switching contexts.
 * 
 * @return true if deinitialization was successful, false otherwise
 */
bool deinit_platform();

/**
 * @brief Check if platform is initialized
 * 
 * @return true if platform is initialized and ready to use
 */
bool is_platform_initialized();

/**
 * @brief Get platform capabilities
 * 
 * Returns information about hardware capabilities available on this platform.
 * Used by the runtime to optimize operations based on available features.
 * 
 * @return PlatformCapabilities structure with hardware information
 */
PlatformCapabilities get_platform_capabilities();

// =============================================================================
// TIMING AND PROFILING
// =============================================================================

/**
 * @brief Get high-resolution timestamp in microseconds
 * 
 * Provides monotonic time reference for profiling and scheduling.
 * Resolution should be at least 1 microsecond where possible.
 * 
 * @return Current timestamp in microseconds since platform initialization
 */
uint64_t get_timestamp_us();

/**
 * @brief Get timestamp in nanoseconds (if available)
 * 
 * Higher resolution timing for fine-grained profiling.
 * Falls back to microsecond precision if nanosecond timing unavailable.
 * 
 * @return Current timestamp in nanoseconds
 */
uint64_t get_timestamp_ns();

/**
 * @brief Sleep for specified microseconds
 * 
 * Puts the current execution context to sleep for the specified duration.
 * Actual sleep time may be longer due to scheduling granularity.
 * 
 * @param microseconds Duration to sleep in microseconds
 */
void sleep_us(uint32_t microseconds);

/**
 * @brief Busy-wait delay for specified microseconds
 * 
 * Active delay without yielding execution. Use for short, precise delays
 * where sleeping would introduce too much overhead.
 * 
 * @param microseconds Duration to delay in microseconds
 */
void delay_us(uint32_t microseconds);

// =============================================================================
// MEMORY MANAGEMENT
// =============================================================================

/**
 * @brief Allocate scratch/temporary memory for computations
 * 
 * Allocates fast, temporary memory for intermediate computations.
 * Memory may come from stack, static pool, or heap depending on platform.
 * 
 * @param size Size in bytes to allocate
 * @param attributes Memory allocation attributes
 * @return Pointer to allocated memory, nullptr if allocation failed
 */
void* allocate_scratch(size_t size, const MemoryAttributes& attributes = {});

/**
 * @brief Free scratch memory allocated by allocate_scratch
 * 
 * @param ptr Pointer to memory to free (must be from allocate_scratch)
 */
void free_scratch(void* ptr);

/**
 * @brief Allocate persistent memory for model weights and buffers
 * 
 * Allocates memory that persists across inference calls.
 * Typically allocated once during model loading.
 * 
 * @param size Size in bytes to allocate
 * @param attributes Memory allocation attributes
 * @return Pointer to allocated memory, nullptr if allocation failed
 */
void* allocate_persistent(size_t size, const MemoryAttributes& attributes = {});

/**
 * @brief Free persistent memory
 * 
 * @param ptr Pointer to memory to free (must be from allocate_persistent)
 */
void free_persistent(void* ptr);

/**
 * @brief Get available scratch memory
 * 
 * @return Available scratch memory in bytes
 */
size_t get_available_scratch_memory();

/**
 * @brief Get available persistent memory
 * 
 * @return Available persistent memory in bytes
 */
size_t get_available_persistent_memory();

// =============================================================================
// CACHE MANAGEMENT
// =============================================================================

/**
 * @brief Flush cache lines for specified memory range
 * 
 * Ensures data in cache is written back to main memory.
 * Required before DMA transfers or sharing data between cores.
 * 
 * @param ptr Pointer to memory range to flush
 * @param size Size of memory range in bytes
 */
void flush_cache(const void* ptr, size_t size);

/**
 * @brief Invalidate cache lines for specified memory range
 * 
 * Marks cache lines as invalid, forcing reload from main memory.
 * Required after DMA transfers or when data is modified by other cores.
 * 
 * @param ptr Pointer to memory range to invalidate
 * @param size Size of memory range in bytes
 */
void invalidate_cache(const void* ptr, size_t size);

/**
 * @brief Prefetch data into cache
 * 
 * Hint to the cache controller to load data in preparation for access.
 * May improve performance for predictable memory access patterns.
 * 
 * @param ptr Pointer to memory to prefetch
 * @param size Size of memory range to prefetch
 */
void prefetch_data(const void* ptr, size_t size);

/**
 * @brief Clean and invalidate cache for memory range
 * 
 * Combines flush and invalidate operations for cache coherency.
 * 
 * @param ptr Pointer to memory range
 * @param size Size of memory range in bytes
 */
void clean_invalidate_cache(const void* ptr, size_t size);

// =============================================================================
// DMA OPERATIONS
// =============================================================================

/**
 * @brief Perform DMA memory transfer (blocking)
 * 
 * Transfers data using DMA controller and waits for completion.
 * Source and destination must be in DMA-accessible memory regions.
 * 
 * @param dst Destination pointer (must be DMA accessible)
 * @param src Source pointer (must be DMA accessible)
 * @param size Number of bytes to transfer
 * @return true if transfer completed successfully, false otherwise
 */
bool dma_transfer_blocking(void* dst, const void* src, size_t size);

/**
 * @brief Start asynchronous DMA transfer
 * 
 * Initiates DMA transfer and returns immediately.
 * Use dma_wait_completion to wait for transfer to finish.
 * 
 * @param dst Destination pointer (must be DMA accessible)
 * @param src Source pointer (must be DMA accessible)
 * @param size Number of bytes to transfer
 * @return DMA handle for tracking transfer, nullptr if failed to start
 */
DmaHandle* dma_transfer_async(void* dst, const void* src, size_t size);

/**
 * @brief Wait for DMA transfer completion
 * 
 * @param handle DMA handle from dma_transfer_async
 * @param timeout_ms Maximum time to wait in milliseconds (0 = infinite)
 * @return true if transfer completed successfully, false on timeout/error
 */
bool dma_wait_completion(DmaHandle* handle, uint32_t timeout_ms = 0);

/**
 * @brief Check if memory address is DMA accessible
 * 
 * @param ptr Memory address to check
 * @return true if address can be used for DMA operations
 */
bool is_dma_accessible(const void* ptr);

// =============================================================================
// LOGGING AND DEBUGGING
// =============================================================================

/**
 * @brief Platform logging function
 * 
 * Outputs debug information through platform-specific mechanism
 * (UART, RTT, semihosting, etc.). Thread-safe where possible.
 * 
 * @param level Log level for the message
 * @param message Null-terminated string to log
 */
void log_message(LogLevel level, const char* message);

/**
 * @brief Debug logging (convenience function)
 * 
 * @param message Debug message to log
 */
inline void log_debug(const char* message) {
    log_message(LogLevel::DEBUG, message);
}

/**
 * @brief Info logging (convenience function)
 * 
 * @param message Info message to log
 */
inline void log_info(const char* message) {
    log_message(LogLevel::INFO, message);
}

/**
 * @brief Warning logging (convenience function)
 * 
 * @param message Warning message to log
 */
inline void log_warning(const char* message) {
    log_message(LogLevel::WARNING, message);
}

/**
 * @brief Error logging (convenience function)
 * 
 * @param message Error message to log
 */
inline void log_error(const char* message) {
    log_message(LogLevel::ERROR, message);
}

// =============================================================================
// POWER MANAGEMENT
// =============================================================================

/**
 * @brief Enter low-power mode
 * 
 * Reduces power consumption while maintaining ability to wake up.
 * Implementation depends on platform capabilities.
 */
void enter_low_power_mode();

/**
 * @brief Exit low-power mode
 * 
 * Returns to full performance mode.
 */
void exit_low_power_mode();

/**
 * @brief Set CPU frequency
 * 
 * Adjusts CPU clock frequency for power/performance trade-off.
 * 
 * @param frequency_mhz Desired frequency in MHz
 * @return true if frequency was set successfully
 */
bool set_cpu_frequency(uint32_t frequency_mhz);

// =============================================================================
// INTERRUPT MANAGEMENT
// =============================================================================

/**
 * @brief Disable interrupts globally
 * 
 * @return Previous interrupt state (for restore_interrupts)
 */
uint32_t disable_interrupts();

/**
 * @brief Restore interrupt state
 * 
 * @param state Previous interrupt state from disable_interrupts
 */
void restore_interrupts(uint32_t state);

/**
 * @brief Critical section RAII wrapper
 */
class CriticalSection {
public:
    CriticalSection() : saved_state_(disable_interrupts()) {}
    ~CriticalSection() { restore_interrupts(saved_state_); }
    
private:
    uint32_t saved_state_;
    // Prevent copying
    CriticalSection(const CriticalSection&) = delete;
    CriticalSection& operator=(const CriticalSection&) = delete;
};

// =============================================================================
// ATOMIC OPERATIONS
// =============================================================================

/**
 * @brief Atomic load (32-bit)
 * 
 * @param ptr Pointer to 32-bit value
 * @return Current value
 */
uint32_t atomic_load_32(const volatile uint32_t* ptr);

/**
 * @brief Atomic store (32-bit)
 * 
 * @param ptr Pointer to 32-bit value
 * @param value Value to store
 */
void atomic_store_32(volatile uint32_t* ptr, uint32_t value);

/**
 * @brief Atomic compare and swap (32-bit)
 * 
 * @param ptr Pointer to 32-bit value
 * @param expected Expected current value
 * @param desired Desired new value
 * @return true if swap occurred, false otherwise
 */
bool atomic_compare_swap_32(volatile uint32_t* ptr, uint32_t expected, uint32_t desired);

} // namespace platform
} // namespace cmx