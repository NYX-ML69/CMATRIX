#pragma once

#include "cmx_types.hpp"
#include "cmx_config.hpp"
#include <chrono>
#include <unordered_map>
#include <string>
#include <vector>

namespace cmx {

#if CMX_PROFILING

// High-resolution timer type
using cmx_time_point = std::chrono::high_resolution_clock::time_point;
using cmx_duration = std::chrono::nanoseconds;

// Profiling statistics for a section
struct cmx_profile_stats {
    std::string section_name;
    cmx_u64 call_count;
    cmx_duration total_time;
    cmx_duration min_time;
    cmx_duration max_time;
    cmx_duration avg_time;
    
    cmx_profile_stats() 
        : call_count(0)
        , total_time(0)
        , min_time(cmx_duration::max())
        , max_time(cmx_duration::min())
        , avg_time(0) {}
        
    void update(cmx_duration elapsed);
    void reset();
};

// Thread-local profiler state
struct cmx_profiler_state {
    std::unordered_map<std::string, cmx_profile_stats> stats;
    std::vector<std::pair<std::string, cmx_time_point>> call_stack;
    bool enabled;
    
    cmx_profiler_state() : enabled(false) {}
};

// Main profiler class
class CmxProfiler {
private:
    static thread_local cmx_profiler_state state_;
    static bool global_enabled_;
    
public:
    // Global enable/disable
    static void enable(bool flag = true) { global_enabled_ = flag; }
    static void disable() { global_enabled_ = false; }
    static bool is_enabled() { return global_enabled_; }
    
    // Thread-local enable/disable
    static void enable_local(bool flag = true) { state_.enabled = flag; }
    static void disable_local() { state_.enabled = false; }
    static bool is_local_enabled() { return state_.enabled && global_enabled_; }
    
    // Profiling operations
    static void begin_section(const char* section_name);
    static void end_section(const char* section_name);
    
    // Statistics access
    static const cmx_profile_stats* get_stats(const char* section_name);
    static std::vector<cmx_profile_stats> get_all_stats();
    static void clear_stats();
    
    // Reporting
    static void print_report();
    static std::string generate_report();
    
    // Internal state access
    static cmx_profiler_state& get_state() { return state_; }
};

// RAII profiling scope guard
class CmxProfileScope {
private:
    const char* section_name_;
    bool active_;
    
public:
    explicit CmxProfileScope(const char* section_name);
    ~CmxProfileScope();
    
    // Non-copyable, non-movable
    CmxProfileScope(const CmxProfileScope&) = delete;
    CmxProfileScope& operator=(const CmxProfileScope&) = delete;
    CmxProfileScope(CmxProfileScope&&) = delete;
    CmxProfileScope& operator=(CmxProfileScope&&) = delete;
};

// C-style API functions
void cmx_profiler_enable(bool flag);
void cmx_profiler_begin(const char* section);
void cmx_profiler_end(const char* section);
void cmx_profiler_report();
void cmx_profiler_clear();

// Utility macros for easier profiling
#define CMX_PROFILE_SCOPE(name) cmx::CmxProfileScope _prof_scope_(name)
#define CMX_PROFILE_FUNCTION() CMX_PROFILE_SCOPE(__FUNCTION__)
#define CMX_PROFILE_BEGIN(name) cmx::CmxProfiler::begin_section(name)
#define CMX_PROFILE_END(name) cmx::CmxProfiler::end_section(name)

#else // CMX_PROFILING disabled

// Stub implementations when profiling is disabled
class CmxProfiler {
public:
    static void enable(bool = true) {}
    static void disable() {}
    static bool is_enabled() { return false; }
    static void enable_local(bool = true) {}
    static void disable_local() {}
    static bool is_local_enabled() { return false; }
    static void begin_section(const char*) {}
    static void end_section(const char*) {}
    static void clear_stats() {}
    static void print_report() {}
    static std::string generate_report() { return "Profiling disabled"; }
};

class CmxProfileScope {
public:
    explicit CmxProfileScope(const char*) {}
};

// Stub C-style API
inline void cmx_profiler_enable(bool) {}
inline void cmx_profiler_begin(const char*) {}
inline void cmx_profiler_end(const char*) {}
inline void cmx_profiler_report() {}
inline void cmx_profiler_clear() {}

// No-op macros when profiling is disabled
#define CMX_PROFILE_SCOPE(name) do {} while(0)
#define CMX_PROFILE_FUNCTION() do {} while(0)
#define CMX_PROFILE_BEGIN(name) do {} while(0)
#define CMX_PROFILE_END(name) do {} while(0)

#endif // CMX_PROFILING

} // namespace cmx