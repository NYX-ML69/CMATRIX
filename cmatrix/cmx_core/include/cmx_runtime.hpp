#pragma once

#include "cmx_config.hpp"
#include "cmx_error.hpp"
#include "cmx_profile.hpp"
#include "cmx_types.hpp"

namespace cmx {

/**
 * @brief Runtime configuration structure
 */
struct cmx_runtime_config {
    uint32_t num_threads;
    uint64_t memory_pool_size;
    bool enable_profiling;
    bool enable_optimization;
    uint32_t optimization_level;
    const char* cache_directory;
    const char* log_level;
    bool enable_debug_mode;
};

/**
 * @brief Runtime statistics structure
 */
struct cmx_runtime_stats {
    uint64_t total_operations;
    uint64_t total_execution_time_us;
    uint64_t peak_memory_usage;
    uint64_t current_memory_usage;
    uint32_t active_threads;
    uint32_t cache_hits;
    uint32_t cache_misses;
};

/**
 * @brief Runtime version information
 */
struct cmx_version_info {
    uint32_t major;
    uint32_t minor;
    uint32_t patch;
    const char* build_date;
    const char* commit_hash;
    const char* build_type;  // Debug, Release, RelWithDebInfo
};

// Core Runtime Functions
/**
 * @brief Initialize the CMatrix runtime
 * @return Status code indicating success or failure
 */
cmx_status cmx_init();

/**
 * @brief Initialize runtime with custom configuration
 * @param config Runtime configuration parameters
 * @return Status code indicating success or failure
 */
cmx_status cmx_init_with_config(const cmx_runtime_config& config);

/**
 * @brief Shutdown the CMatrix runtime
 * @return Status code indicating success or failure
 */
cmx_status cmx_shutdown();

/**
 * @brief Check if runtime is initialized
 * @return true if runtime is initialized, false otherwise
 */
bool cmx_is_initialized();

// Version and Information
/**
 * @brief Get CMatrix version string
 * @return Version string in format "major.minor.patch"
 */
const char* cmx_get_version();

/**
 * @brief Get detailed version information
 * @param version_info Pointer to structure to fill with version details
 * @return Status code indicating success or failure
 */
cmx_status cmx_get_version_info(cmx_version_info* version_info);

/**
 * @brief Get build information string
 * @return Build information string
 */
const char* cmx_get_build_info();

/**
 * @brief Get supported features as a comma-separated string
 * @return Features string (e.g., "CUDA,OpenCL,AVX2,FP16")
 */
const char* cmx_get_features();

// Runtime Configuration
/**
 * @brief Set runtime configuration
 * @param config Configuration structure
 * @return Status code indicating success or failure
 */
cmx_status cmx_set_config(const cmx_runtime_config& config);

/**
 * @brief Get current runtime configuration
 * @param config Pointer to structure to fill with current config
 * @return Status code indicating success or failure
 */
cmx_status cmx_get_config(cmx_runtime_config* config);

/**
 * @brief Set number of worker threads
 * @param num_threads Number of threads to use (0 for auto-detect)
 * @return Status code indicating success or failure
 */
cmx_status cmx_set_num_threads(uint32_t num_threads);

/**
 * @brief Get current number of worker threads
 * @param num_threads Pointer to store current thread count
 * @return Status code indicating success or failure
 */
cmx_status cmx_get_num_threads(uint32_t* num_threads);

/**
 * @brief Set memory pool size
 * @param pool_size Size of memory pool in bytes
 * @return Status code indicating success or failure
 */
cmx_status cmx_set_memory_pool_size(uint64_t pool_size);

// Runtime Statistics and Monitoring
/**
 * @brief Get runtime statistics
 * @param stats Pointer to structure to fill with statistics
 * @return Status code indicating success or failure
 */
cmx_status cmx_get_runtime_stats(cmx_runtime_stats* stats);

/**
 * @brief Reset runtime statistics
 * @return Status code indicating success or failure
 */
cmx_status cmx_reset_runtime_stats();

/**
 * @brief Enable or disable profiling
 * @param enable true to enable profiling, false to disable
 * @return Status code indicating success or failure
 */
cmx_status cmx_set_profiling_enabled(bool enable);

/**
 * @brief Check if profiling is enabled
 * @return true if profiling is enabled, false otherwise
 */
bool cmx_is_profiling_enabled();

// Memory Management
/**
 * @brief Get current memory usage
 * @param current_bytes Pointer to store current memory usage
 * @param peak_bytes Pointer to store peak memory usage
 * @return Status code indicating success or failure
 */
cmx_status cmx_get_memory_usage(uint64_t* current_bytes, uint64_t* peak_bytes);

/**
 * @brief Trigger garbage collection
 * @return Status code indicating success or failure
 */
cmx_status cmx_garbage_collect();

/**
 * @brief Set memory limit
 * @param limit_bytes Maximum memory usage in bytes
 * @return Status code indicating success or failure
 */
cmx_status cmx_set_memory_limit(uint64_t limit_bytes);

// Logging and Debugging
/**
 * @brief Set log level
 * @param level Log level string ("DEBUG", "INFO", "WARN", "ERROR")
 * @return Status code indicating success or failure
 */
cmx_status cmx_set_log_level(const char* level);

/**
 * @brief Get current log level
 * @return Current log level string
 */
const char* cmx_get_log_level();

/**
 * @brief Enable or disable debug mode
 * @param enable true to enable debug mode, false to disable
 * @return Status code indicating success or failure
 */
cmx_status cmx_set_debug_mode(bool enable);

/**
 * @brief Check if debug mode is enabled
 * @return true if debug mode is enabled, false otherwise
 */
bool cmx_is_debug_mode_enabled();

// Cache Management
/**
 * @brief Set cache directory for compiled kernels and optimized graphs
 * @param cache_dir Path to cache directory
 * @return Status code indicating success or failure
 */
cmx_status cmx_set_cache_directory(const char* cache_dir);

/**
 * @brief Get current cache directory
 * @return Path to current cache directory
 */
const char* cmx_get_cache_directory();

/**
 * @brief Clear all cached data
 * @return Status code indicating success or failure
 */
cmx_status cmx_clear_cache();

/**
 * @brief Get cache statistics
 * @param cache_hits Pointer to store number of cache hits
 * @param cache_misses Pointer to store number of cache misses
 * @param cache_size_bytes Pointer to store cache size in bytes
 * @return Status code indicating success or failure
 */
cmx_status cmx_get_cache_stats(uint32_t* cache_hits, uint32_t* cache_misses, uint64_t* cache_size_bytes);

// Error Handling
/**
 * @brief Get last error message
 * @return Pointer to last error message string
 */
const char* cmx_get_last_error();

/**
 * @brief Clear last error
 * @return Status code indicating success or failure
 */
cmx_status cmx_clear_last_error();

/**
 * @brief Set custom error handler
 * @param handler Function pointer to custom error handler
 * @return Status code indicating success or failure
 */
cmx_status cmx_set_error_handler(cmx_error_handler handler);

// Runtime Events and Callbacks
/**
 * @brief Register runtime event callback
 * @param event_type Type of event to listen for
 * @param callback Function to call on event
 * @param user_data User data to pass to callback
 * @return Status code indicating success or failure
 */
cmx_status cmx_register_event_callback(cmx_runtime_event_type event_type, 
                                       cmx_event_callback callback, void* user_data);

/**
 * @brief Unregister runtime event callback
 * @param event_type Type of event to stop listening for
 * @param callback Function to unregister
 * @return Status code indicating success or failure
 */
cmx_status cmx_unregister_event_callback(cmx_runtime_event_type event_type, 
                                         cmx_event_callback callback);

// Hardware Detection and Optimization
/**
 * @brief Auto-detect optimal runtime settings for current hardware
 * @param config Pointer to configuration structure to fill
 * @return Status code indicating success or failure
 */
cmx_status cmx_auto_detect_config(cmx_runtime_config* config);

/**
 * @brief Get hardware information
 * @param hw_info Pointer to structure to fill with hardware information
 * @return Status code indicating success or failure
 */
cmx_status cmx_get_hardware_info(cmx_hardware_info* hw_info);

/**
 * @brief Run runtime benchmark to determine optimal settings
 * @param benchmark_duration_ms Duration of benchmark in milliseconds
 * @param optimal_config Pointer to store optimal configuration
 * @return Status code indicating success or failure
 */
cmx_status cmx_benchmark_runtime(uint32_t benchmark_duration_ms, 
                                 cmx_runtime_config* optimal_config);

} // namespace cmx