// cmx_runtime_config.hpp
#pragma once

#include <cstddef>
#include <cstdint>

namespace cmx {
namespace runtime {

/**
 * @brief Central configuration constants for the CMX runtime system
 * 
 * All memory sizes, limits, and system parameters are defined here.
 * Can be overridden at compile time for different deployment targets.
 */
struct RuntimeConfig {
    // Memory pool configuration
    static constexpr size_t DEFAULT_MEMORY_POOL_SIZE = 1024 * 1024;  // 1MB
    static constexpr size_t DEFAULT_TENSOR_POOL_SIZE = 512 * 1024;   // 512KB
    static constexpr size_t DEFAULT_TEMP_BUFFER_SIZE = 256 * 1024;   // 256KB
    
    // Scheduler configuration
    static constexpr size_t MAX_TASKS = 64;
    static constexpr size_t MAX_TASK_DEPENDENCIES = 8;
    static constexpr uint32_t DEFAULT_TASK_PRIORITY = 5;
    static constexpr uint32_t MAX_PRIORITY = 10;
    
    // Tensor configuration
    static constexpr size_t MAX_TENSOR_DIMS = 8;
    static constexpr size_t MAX_TENSOR_NAME_LENGTH = 32;
    
    // Profiler configuration
    static constexpr size_t MAX_PROFILER_ENTRIES = 256;
    static constexpr bool PROFILING_ENABLED = true;
    
    // Alignment requirements
    static constexpr size_t MEMORY_ALIGNMENT = 64;  // 64-byte alignment for SIMD
    static constexpr size_t TENSOR_ALIGNMENT = 32;
};

// Compile-time overrides
#ifndef CMX_MEMORY_POOL_SIZE
#define CMX_MEMORY_POOL_SIZE RuntimeConfig::DEFAULT_MEMORY_POOL_SIZE
#endif

#ifndef CMX_PROFILING_ENABLED
#define CMX_PROFILING_ENABLED RuntimeConfig::PROFILING_ENABLED
#endif

/**
 * @brief Get the global runtime configuration
 */
const RuntimeConfig& GetRuntimeConfig();

/**
 * @brief Initialize runtime configuration with custom values
 * @param config Custom configuration parameters
 */
void InitializeRuntimeConfig(const RuntimeConfig& config);

} // namespace runtime
} // namespace cmx