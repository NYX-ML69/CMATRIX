#pragma once

#include "cmx_api.hpp"

/**
 * @file cmx_runtime_api.hpp
 * @brief Runtime engine lifecycle and control APIs
 * 
 * Provides functions to initialize, configure, and manage
 * the CMatrix runtime engine. Controls global runtime state
 * and resource management for embedded environments.
 */

namespace cmx {

/**
 * @brief Runtime configuration structure
 */
struct cmx_runtime_config {
    size_t memory_pool_size;      ///< Size of memory pool in bytes
    uint32_t max_models;          ///< Maximum number of concurrent models
    uint32_t thread_count;        ///< Number of worker threads (0 = auto)
    bool enable_optimization;     ///< Enable runtime optimizations
    bool enable_profiling;        ///< Enable performance profiling
    const char* cache_directory;  ///< Directory for runtime cache files
};

/**
 * @brief Runtime statistics structure
 */
struct cmx_runtime_stats {
    size_t memory_used;           ///< Current memory usage in bytes
    size_t memory_peak;           ///< Peak memory usage in bytes
    uint32_t models_loaded;       ///< Number of currently loaded models
    uint64_t total_executions;    ///< Total number of model executions
    double average_exec_time_ms;  ///< Average execution time in milliseconds
};

/**
 * @brief Initialize the CMatrix runtime with default configuration
 * @return Status code indicating initialization result
 */
cmx_status cmx_init();

/**
 * @brief Initialize the CMatrix runtime with custom configuration
 * @param config Pointer to configuration structure
 * @return Status code indicating initialization result
 */
cmx_status cmx_init_with_config(const cmx_runtime_config* config);

/**
 * @brief Shutdown the CMatrix runtime and release all resources
 * @return Status code indicating shutdown result
 */
cmx_status cmx_shutdown();

/**
 * @brief Check if the runtime is initialized
 * @return true if runtime is initialized, false otherwise
 */
bool cmx_is_initialized();

/**
 * @brief Get CMatrix runtime version string
 * @return Version string (e.g., "1.0.0")
 */
const char* cmx_get_version();

/**
 * @brief Get CMatrix runtime build information
 * @return Build info string containing compiler, date, etc.
 */
const char* cmx_get_build_info();

/**
 * @brief Get current runtime configuration
 * @param config Pointer to configuration structure to populate
 * @return Status code indicating success or failure
 */
cmx_status cmx_get_runtime_config(cmx_runtime_config* config);

/**
 * @brief Update runtime configuration (limited subset)
 * @param config Pointer to new configuration
 * @return Status code indicating success or failure
 */
cmx_status cmx_update_runtime_config(const cmx_runtime_config* config);

/**
 * @brief Get runtime statistics and performance metrics
 * @param stats Pointer to statistics structure to populate
 * @return Status code indicating success or failure
 */
cmx_status cmx_get_runtime_stats(cmx_runtime_stats* stats);

/**
 * @brief Reset runtime statistics counters
 * @return Status code indicating success or failure
 */
cmx_status cmx_reset_runtime_stats();

/**
 * @brief Force garbage collection of unused resources
 * @return Status code indicating success or failure
 */
cmx_status cmx_garbage_collect();

/**
 * @brief Set log level for runtime debugging
 * @param level Log level (0=off, 1=error, 2=warn, 3=info, 4=debug)
 * @return Status code indicating success or failure
 */
cmx_status cmx_set_log_level(uint32_t level);

/**
 * @brief Get last error message from runtime
 * @return Error message string, or nullptr if no error
 */
const char* cmx_get_last_error();

/**
 * @brief Clear the last error message
 */
void cmx_clear_last_error();

} // namespace cmx