#pragma once

#include "cmx_types.hpp"
#include "cmx_config.hpp"
#include <string>
#include <ostream>

namespace cmx {

// Debug levels
enum class cmx_debug_level : cmx_u8 {
    TRACE = 0,
    DEBUG = 1,
    INFO = 2,
    WARN = 3,
    ERROR = 4,
    FATAL = 5
};

// Debug output handler function type
using cmx_debug_handler_t = void(*)(cmx_debug_level level, const char* file, 
                                   cmx_u32 line, const char* function, 
                                   const std::string& message);

// Debug manager class
class CmxDebugManager {
private:
    static cmx_debug_handler_t debug_handler_;
    static cmx_debug_level min_level_;
    static bool include_timestamp_;
    static bool include_thread_id_;
    static std::ostream* output_stream_;
    
public:
    // Debug handler management
    static void set_debug_handler(cmx_debug_handler_t handler);
    static cmx_debug_handler_t get_debug_handler() { return debug_handler_; }
    static void reset_debug_handler();
    
    // Debug level management
    static void set_min_debug_level(cmx_debug_level level) { min_level_ = level; }
    static cmx_debug_level get_min_debug_level() { return min_level_; }
    
    // Output formatting options
    static void set_include_timestamp(bool include) { include_timestamp_ = include; }
    static bool get_include_timestamp() { return include_timestamp_; }
    
    static void set_include_thread_id(bool include) { include_thread_id_ = include; }
    static bool get_include_thread_id() { return include_thread_id_; }
    
    static void set_output_stream(std::ostream* stream) { output_stream_ = stream; }
    static std::ostream* get_output_stream() { return output_stream_; }
    
    // Debug logging
    static void log(cmx_debug_level level, const char* file, cmx_u32 line, 
                   const char* function, const std::string& message);
    
    // Convenience logging methods
    static void trace(const char* file, cmx_u32 line, const char* function, 
                     const std::string& message);
    static void debug(const char* file, cmx_u32 line, const char* function, 
                     const std::string& message);
    static void info(const char* file, cmx_u32 line, const char* function, 
                    const std::string& message);
    static void warn(const char* file, cmx_u32 line, const char* function, 
                    const std::string& message);
    static void error(const char* file, cmx_u32 line, const char* function, 
                     const std::string& message);
    static void fatal(const char* file, cmx_u32 line, const char* function, 
                     const std::string& message);
    
    // Default debug handler
    static void default_debug_handler(cmx_debug_level level, const char* file, 
                                     cmx_u32 line, const char* function, 
                                     const std::string& message);
};

// Utility functions
const char* cmx_debug_level_to_string(cmx_debug_level level);
const char* cmx_debug_level_to_color_code(cmx_debug_level level);
std::string cmx_get_timestamp_string();
std::string cmx_get_thread_id_string();

#if CMX_DEBUG

// Debug logging macros (enabled)
#define CMX_LOG_TRACE(msg) \
    cmx::CmxDebugManager::trace(__FILE__, __LINE__, __FUNCTION__, msg)

#define CMX_LOG_DEBUG(msg) \
    cmx::CmxDebugManager::debug(__FILE__, __LINE__, __FUNCTION__, msg)

#define CMX_LOG_INFO(msg) \
    cmx::CmxDebugManager::info(__FILE__, __LINE__, __FUNCTION__, msg)

#define CMX_LOG_WARN(msg) \
    cmx::CmxDebugManager::warn(__FILE__, __LINE__, __FUNCTION__, msg)

#define CMX_LOG_ERROR(msg) \
    cmx::CmxDebugManager::error(__FILE__, __LINE__, __FUNCTION__, msg)

#define CMX_LOG_FATAL(msg) \
    cmx::CmxDebugManager::fatal(__FILE__, __LINE__, __FUNCTION__, msg)

#define CMX_LOG(level, msg) \
    cmx::CmxDebugManager::log(level, __FILE__, __LINE__, __FUNCTION__, msg)

// Debug variable printing
#define CMX_LOG_VAR(var) \
    CMX_LOG_DEBUG(std::string(#var " = ") + std::to_string(var))

#define CMX_LOG_PTR(ptr) \
    CMX_LOG_DEBUG(std::string(#ptr " = ") + \
                  (ptr ? std::to_string(reinterpret_cast<std::uintptr_t>(ptr)) : "nullptr"))

// Debug function entry/exit tracking
#define CMX_LOG_ENTER() \
    CMX_LOG_TRACE("Entering function")

#define CMX_LOG_EXIT() \
    CMX_LOG_TRACE("Exiting function")

#define CMX_LOG_RETURN(val) \
    do { \
        CMX_LOG_TRACE(std::string("Returning: ") + std::to_string(val)); \
        return val; \
    } while(0)

// Conditional debug logging
#define CMX_LOG_IF(condition, level, msg) \
    do { \
        if (condition) { \
            CMX_LOG(level, msg); \
        } \
    } while(0)

// Debug timing helpers
class CmxDebugTimer {
private:
    std::chrono::high_resolution_clock::time_point start_time_;
    std::string name_;
    
public:
    explicit CmxDebugTimer(const std::string& name);
    ~CmxDebugTimer();
    
    void reset();
    cmx_f64 elapsed_ms() const;
};

#define CMX_DEBUG_TIMER(name) cmx::CmxDebugTimer _debug_timer_(name)

#else // CMX_DEBUG disabled

// Debug logging macros (disabled - no-ops)
#define CMX_LOG_TRACE(msg) do {} while(0)
#define CMX_LOG_DEBUG(msg) do {} while(0)
#define CMX_LOG_INFO(msg) do {} while(0)
#define CMX_LOG_WARN(msg) do {} while(0)
#define CMX_LOG_ERROR(msg) do {} while(0)
#define CMX_LOG_FATAL(msg) do {} while(0)
#define CMX_LOG(level, msg) do {} while(0)
#define CMX_LOG_VAR(var) do {} while(0)
#define CMX_LOG_PTR(ptr) do {} while(0)
#define CMX_LOG_ENTER() do {} while(0)
#define CMX_LOG_EXIT() do {} while(0)
#define CMX_LOG_RETURN(val) return val
#define CMX_LOG_IF(condition, level, msg) do {} while(0)

// Stub timer class when debugging is disabled
class CmxDebugTimer {
public:
    explicit CmxDebugTimer(const std::string&) {}
    void reset() {}
    cmx_f64 elapsed_ms() const { return 0.0; }
};

#define CMX_DEBUG_TIMER(name) do {} while(0)

#endif // CMX_DEBUG

// Always-available debug functions (not conditional on CMX_DEBUG)
void cmx_debug_log(const char* file, cmx_u32 line, const char* function, 
                  const std::string& message);
void cmx_debug_set_level(cmx_debug_level level);
void cmx_debug_enable_colors(bool enable);

// Memory debugging helpers (always available)
class CmxMemoryTracker {
private:
    static cmx_size total_allocated_;
    static cmx_size peak_allocated_;
    static cmx_size allocation_count_;
    
public:
    static void track_allocation(cmx_size size);
    static void track_deallocation(cmx_size size);
    static cmx_size get_total_allocated() { return total_allocated_; }
    static cmx_size get_peak_allocated() { return peak_allocated_; }
    static cmx_size get_allocation_count() { return allocation_count_; }
    static void reset_stats();
    static void print_stats();
};

// Memory debugging macros
#define CMX_TRACK_ALLOC(size) cmx::CmxMemoryTracker::track_allocation(size)
#define CMX_TRACK_FREE(size) cmx::CmxMemoryTracker::track_deallocation(size)

} // namespace cmx