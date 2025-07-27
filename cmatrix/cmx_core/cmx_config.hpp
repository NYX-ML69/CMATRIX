#pragma once

#include "cmx_types.hpp"

namespace cmx {

// Compile-time configuration flags
#ifndef CMX_DEBUG
#define CMX_DEBUG 0
#endif

#ifndef CMX_PROFILING
#define CMX_PROFILING 0
#endif

#ifndef CMX_BOUNDS_CHECK
#define CMX_BOUNDS_CHECK CMX_DEBUG
#endif

#ifndef CMX_MAX_THREADS
#define CMX_MAX_THREADS 16
#endif

#ifndef CMX_DEFAULT_ALIGNMENT
#define CMX_DEFAULT_ALIGNMENT 32
#endif

// Runtime configuration structure
struct cmx_runtime_config {
    bool debug_enabled;
    bool profiling_enabled;
    bool bounds_check_enabled;
    cmx_u32 num_threads;
    cmx_size memory_alignment;
    cmx_device default_device;
    cmx_size memory_pool_size;
    bool use_memory_pool;
    
    // Constructor with defaults
    cmx_runtime_config();
};

// Global configuration management
class CmxConfig {
private:
    static cmx_runtime_config config_;
    static bool initialized_;

public:
    // Initialization
    static void initialize();
    static void initialize(const cmx_runtime_config& custom_config);
    static bool is_initialized() { return initialized_; }
    
    // Debug settings
    static bool is_debug_enabled() { return config_.debug_enabled; }
    static void enable_debug(bool flag) { config_.debug_enabled = flag; }
    
    // Profiling settings
    static bool is_profiling_enabled() { return config_.profiling_enabled; }
    static void enable_profiling(bool flag) { config_.profiling_enabled = flag; }
    
    // Bounds checking
    static bool is_bounds_check_enabled() { return config_.bounds_check_enabled; }
    static void enable_bounds_check(bool flag) { config_.bounds_check_enabled = flag; }
    
    // Threading settings
    static cmx_u32 get_num_threads() { return config_.num_threads; }
    static void set_num_threads(cmx_u32 threads);
    
    // Memory settings
    static cmx_size get_memory_alignment() { return config_.memory_alignment; }
    static void set_memory_alignment(cmx_size alignment);
    
    static cmx_size get_memory_pool_size() { return config_.memory_pool_size; }
    static void set_memory_pool_size(cmx_size size) { config_.memory_pool_size = size; }
    
    static bool use_memory_pool() { return config_.use_memory_pool; }
    static void enable_memory_pool(bool flag) { config_.use_memory_pool = flag; }
    
    // Device settings
    static cmx_device get_default_device() { return config_.default_device; }
    static void set_default_device(cmx_device device) { config_.default_device = device; }
    
    // Configuration access
    static const cmx_runtime_config& get_config() { return config_; }
    static void set_config(const cmx_runtime_config& new_config);
    
    // Environment variable loading
    static void load_from_environment();
    
    // Reset to defaults
    static void reset_to_defaults();
};

// Convenience functions
bool is_debug_enabled();
void enable_debug(bool flag);
bool is_profiling_enabled();
void enable_profiling(bool flag);
void set_num_threads(cmx_u32 threads);
cmx_u32 get_num_threads();
void set_default_device(cmx_device device);
cmx_device get_default_device();

// RAII configuration scope
class CmxConfigScope {
private:
    cmx_runtime_config saved_config_;
    
public:
    explicit CmxConfigScope(const cmx_runtime_config& temp_config);
    ~CmxConfigScope();
    
    // Non-copyable, non-movable
    CmxConfigScope(const CmxConfigScope&) = delete;
    CmxConfigScope& operator=(const CmxConfigScope&) = delete;
    CmxConfigScope(CmxConfigScope&&) = delete;
    CmxConfigScope& operator=(CmxConfigScope&&) = delete;
};

} // namespace cmx