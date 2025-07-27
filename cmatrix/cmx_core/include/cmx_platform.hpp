#pragma once

#include "cmx_platform_abstraction.hpp"
#include "cmx_error.hpp"
#include "cmx_types.hpp"

namespace cmx {

/**
 * @brief Platform types enumeration
 */
enum class cmx_platform_type {
    CMX_PLATFORM_CPU = 0,
    CMX_PLATFORM_CUDA,
    CMX_PLATFORM_OPENCL,
    CMX_PLATFORM_METAL,
    CMX_PLATFORM_VULKAN,
    CMX_PLATFORM_CUSTOM
};

/**
 * @brief Platform capabilities structure
 */
struct cmx_platform_caps {
    bool supports_fp16;
    bool supports_fp64;
    bool supports_int8;
    bool supports_unified_memory;
    bool supports_async_execution;
    uint32_t max_threads;
    uint64_t total_memory;
    uint64_t available_memory;
    uint32_t compute_units;
    float peak_gflops;
};

/**
 * @brief Platform information structure
 */
struct cmx_platform_info {
    cmx_platform_type type;
    const char* name;
    const char* vendor;
    const char* version;
    cmx_platform_caps capabilities;
    void* platform_specific;
};

/**
 * @brief Timer handle for performance measurement
 */
typedef void* cmx_timer_handle;

/**
 * @brief Memory allocation handle
 */
typedef void* cmx_memory_handle;

/**
 * @brief DMA transfer handle
 */
typedef void* cmx_dma_handle;

// Platform Initialization and Management
/**
 * @brief Initialize platform subsystem
 * @return Status code indicating success or failure
 */
cmx_status cmx_platform_init();

/**
 * @brief Shutdown platform subsystem
 * @return Status code indicating success or failure
 */
cmx_status cmx_platform_shutdown();

/**
 * @brief Initialize specific platform
 * @param platform_type Type of platform to initialize
 * @return Status code indicating success or failure
 */
cmx_status cmx_platform_init_device(cmx_platform_type platform_type);

/**
 * @brief Shutdown specific platform
 * @param platform_type Type of platform to shutdown
 * @return Status code indicating success or failure
 */
cmx_status cmx_platform_shutdown_device(cmx_platform_type platform_type);

/**
 * @brief Get available platforms
 * @param platforms Array to store platform information
 * @param max_count Maximum number of platforms to retrieve
 * @param actual_count Pointer to store actual number of platforms
 * @return Status code indicating success or failure
 */
cmx_status cmx_get_available_platforms(cmx_platform_info* platforms, 
                                       uint32_t max_count, uint32_t* actual_count);

/**
 * @brief Set active platform
 * @param platform_type Platform type to set as active
 * @return Status code indicating success or failure
 */
cmx_status cmx_set_active_platform(cmx_platform_type platform_type);

/**
 * @brief Get active platform
 * @param platform_type Pointer to store active platform type
 * @return Status code indicating success or failure
 */
cmx_status cmx_get_active_platform(cmx_platform_type* platform_type);

// Memory Management
/**
 * @brief Allocate platform-specific memory
 * @param size Size in bytes to allocate
 * @param alignment Memory alignment requirement
 * @param handle Pointer to store memory handle
 * @return Status code indicating success or failure
 */
cmx_status cmx_platform_malloc(size_t size, size_t alignment, cmx_memory_handle* handle);

/**
 * @brief Free platform-specific memory
 * @param handle Memory handle to free
 * @return Status code indicating success or failure
 */
cmx_status cmx_platform_free(cmx_memory_handle handle);

/**
 * @brief Get memory address from handle
 * @param handle Memory handle
 * @param ptr Pointer to store memory address
 * @return Status code indicating success or failure
 */
cmx_status cmx_platform_get_ptr(cmx_memory_handle handle, void** ptr);

/**
 * @brief Copy memory between host and device
 * @param dst Destination memory handle
 * @param src Source memory handle
 * @param size Number of bytes to copy
 * @param direction Copy direction (host to device, device to host, etc.)
 * @return Status code indicating success or failure
 */
cmx_status cmx_platform_memcpy(cmx_memory_handle dst, cmx_memory_handle src, 
                               size_t size, cmx_memory_copy_direction direction);

// Timer Functions
/**
 * @brief Create a platform timer
 * @param timer Pointer to store timer handle
 * @return Status code indicating success or failure
 */
cmx_status cmx_platform_timer_create(cmx_timer_handle* timer);

/**
 * @brief Destroy a platform timer
 * @param timer Timer handle to destroy
 * @return Status code indicating success or failure
 */
cmx_status cmx_platform_timer_destroy(cmx_timer_handle timer);

/**
 * @brief Start timer measurement
 * @param timer Timer handle
 * @return Status code indicating success or failure
 */
cmx_status cmx_platform_timer_start(cmx_timer_handle timer);

/**
 * @brief Stop timer measurement
 * @param timer Timer handle
 * @return Status code indicating success or failure
 */
cmx_status cmx_platform_timer_stop(cmx_timer_handle timer);

/**
 * @brief Get elapsed time in milliseconds
 * @param timer Timer handle
 * @param elapsed_ms Pointer to store elapsed time
 * @return Status code indicating success or failure
 */
cmx_status cmx_platform_timer_get_elapsed(cmx_timer_handle timer, float* elapsed_ms);

// DMA Functions
/**
 * @brief Create DMA transfer handle
 * @param dma Pointer to store DMA handle
 * @return Status code indicating success or failure
 */
cmx_status cmx_platform_dma_create(cmx_dma_handle* dma);

/**
 * @brief Destroy DMA transfer handle
 * @param dma DMA handle to destroy
 * @return Status code indicating success or failure
 */
cmx_status cmx_platform_dma_destroy(cmx_dma_handle dma);

/**
 * @brief Start asynchronous DMA transfer
 * @param dma DMA handle
 * @param dst Destination memory handle
 * @param src Source memory handle
 * @param size Number of bytes to transfer
 * @return Status code indicating success or failure
 */
cmx_status cmx_platform_dma_transfer_async(cmx_dma_handle dma, cmx_memory_handle dst,
                                           cmx_memory_handle src, size_t size);

/**
 * @brief Wait for DMA transfer completion
 * @param dma DMA handle
 * @param timeout_ms Timeout in milliseconds (0 for infinite)
 * @return Status code indicating success or failure
 */
cmx_status cmx_platform_dma_wait(cmx_dma_handle dma, uint32_t timeout_ms);

/**
 * @brief Check if DMA transfer is complete
 * @param dma DMA handle
 * @param is_complete Pointer to store completion status
 * @return Status code indicating success or failure
 */
cmx_status cmx_platform_dma_is_complete(cmx_dma_handle dma, bool* is_complete);

// Synchronization
/**
 * @brief Synchronize with platform device
 * @return Status code indicating success or failure
 */
cmx_status cmx_platform_synchronize();

/**
 * @brief Create platform-specific event
 * @param event Pointer to store event handle
 * @return Status code indicating success or failure
 */
cmx_status cmx_platform_create_event(void** event);

/**
 * @brief Destroy platform-specific event
 * @param event Event handle to destroy
 * @return Status code indicating success or failure
 */
cmx_status cmx_platform_destroy_event(void* event);

/**
 * @brief Record an event
 * @param event Event handle
 * @return Status code indicating success or failure
 */
cmx_status cmx_platform_record_event(void* event);

/**
 * @brief Wait for an event
 * @param event Event handle
 * @return Status code indicating success or failure
 */
cmx_status cmx_platform_wait_event(void* event);

} // namespace cmx