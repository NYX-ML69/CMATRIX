#include "cmx_config.hpp"
#include <cstdlib>
#include <algorithm>
#include <thread>

namespace cmx {

// Default configuration constructor
cmx_runtime_config::cmx_runtime_config()
    : debug_enabled(CMX_DEBUG != 0)
    , profiling_enabled(CMX_PROFILING != 0)
    , bounds_check_enabled(CMX_BOUNDS_CHECK != 0)
    , num_threads(std::min(static_cast<cmx_u32>(std::thread::hardware_concurrency()), 
                          static_cast<cmx_u32>(CMX_MAX_THREADS)))
    , memory_alignment(CMX_DEFAULT_ALIGNMENT)
    , default_device(cmx_device::CPU)
    , memory_pool_size(64 * 1024 * 1024)  // 64MB default
    , use_memory_pool(false)
{
    if (num_threads == 0) num_threads = 1;  // Fallback if hardware_concurrency fails
}

// Static member definitions
cmx_runtime_config CmxConfig::config_;
bool CmxConfig::initialized_ = false;

void CmxConfig::initialize() {
    if (!initialized_) {
        config_ = cmx_runtime_config();  // Use default constructor
        load_from_environment();
        initialized_ = true;
    }
}

void CmxConfig::initialize(const cmx_runtime_config& custom_config) {
    config_ = custom_config;
    initialized_ = true;
}

void CmxConfig::set_num_threads(cmx_u32 threads) {
    config_.num_threads = std::min(threads, static_cast<cmx_u32>(CMX_MAX_THREADS));
    if (config_.num_threads == 0) config_.num_threads = 1;
}

void CmxConfig::set_memory_alignment(cmx_size alignment) {
    // Ensure alignment is power of 2 and at least sizeof(void*)
    if (alignment < sizeof(void*)) alignment = sizeof(void*);
    
    // Round up to next power of 2
    cmx_size power = 1;
    while (power < alignment) power <<= 1;
    
    config_.memory_alignment = power;
}

void CmxConfig::set_config(const cmx_runtime_config& new_config) {
    config_ = new_config;
    
    // Validate and fix invalid settings
    set_num_threads(config_.num_threads);
    set_memory_alignment(config_.memory_alignment);
}

void CmxConfig::load_from_environment() {
    // Load debug setting
    if (const char* debug_env = std::getenv("CMX_DEBUG")) {
        config_.debug_enabled = (std::atoi(debug_env) != 0);
    }
    
    // Load profiling setting  
    if (const char* profile_env = std::getenv("CMX_PROFILING")) {
        config_.profiling_enabled = (std::atoi(profile_env) != 0);
    }
    
    // Load thread count
    if (const char* threads_env = std::getenv("CMX_THREADS")) {
        int threads = std::atoi(threads_env);
        if (threads > 0) {
            set_num_threads(static_cast<cmx_u32>(threads));
        }
    }
    
    // Load memory pool size
    if (const char* pool_env = std::getenv("CMX_MEMORY_POOL_SIZE")) {
        long long pool_size = std::atoll(pool_env);
        if (pool_size > 0) {
            config_.memory_pool_size = static_cast<cmx_size>(pool_size);
        }
    }
    
    // Load memory pool usage
    if (const char* use_pool_env = std::getenv("CMX_USE_MEMORY_POOL")) {
        config_.use_memory_pool = (std::atoi(use_pool_env) != 0);
    }
    
    // Load bounds checking
    if (const char* bounds_env = std::getenv("CMX_BOUNDS_CHECK")) {
        config_.bounds_check_enabled = (std::atoi(bounds_env) != 0);
    }
}

void CmxConfig::reset_to_defaults() {
    config_ = cmx_runtime_config();
}

// Convenience functions
bool is_debug_enabled() {
    if (!CmxConfig::is_initialized()) CmxConfig::initialize();
    return CmxConfig::is_debug_enabled();
}

void enable_debug(bool flag) {
    if (!CmxConfig::is_initialized()) CmxConfig::initialize();
    CmxConfig::enable_debug(flag);
}

bool is_profiling_enabled() {
    if (!CmxConfig::is_initialized()) CmxConfig::initialize();
    return CmxConfig::is_profiling_enabled();
}

void enable_profiling(bool flag) {
    if (!CmxConfig::is_initialized()) CmxConfig::initialize();
    CmxConfig::enable_profiling(flag);
}

void set_num_threads(cmx_u32 threads) {
    if (!CmxConfig::is_initialized()) CmxConfig::initialize();
    CmxConfig::set_num_threads(threads);
}

cmx_u32 get_num_threads() {
    if (!CmxConfig::is_initialized()) CmxConfig::initialize();
    return CmxConfig::get_num_threads();
}

void set_default_device(cmx_device device) {
    if (!CmxConfig::is_initialized()) CmxConfig::initialize();
    CmxConfig::set_default_device(device);
}

cmx_device get_default_device() {
    if (!CmxConfig::is_initialized()) CmxConfig::initialize();
    return CmxConfig::get_default_device();
}

// RAII configuration scope implementation
CmxConfigScope::CmxConfigScope(const cmx_runtime_config& temp_config) {
    if (!CmxConfig::is_initialized()) CmxConfig::initialize();
    saved_config_ = CmxConfig::get_config();
    CmxConfig::set_config(temp_config);
}

CmxConfigScope::~CmxConfigScope() {
    CmxConfig::set_config(saved_config_);
}

} // namespace cmx