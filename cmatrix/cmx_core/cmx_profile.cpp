#include "cmx_profile.hpp"
#include <iostream>
#include <iomanip>
#include <sstream>
#include <algorithm>

namespace cmx {

#if CMX_PROFILING

// Static member definitions
thread_local cmx_profiler_state CmxProfiler::state_;
bool CmxProfiler::global_enabled_ = false;

void cmx_profile_stats::update(cmx_duration elapsed) {
    call_count++;
    total_time += elapsed;
    
    if (elapsed < min_time) min_time = elapsed;
    if (elapsed > max_time) max_time = elapsed;
    
    avg_time = total_time / call_count;
}

void cmx_profile_stats::reset() {
    call_count = 0;
    total_time = cmx_duration(0);
    min_time = cmx_duration::max();
    max_time = cmx_duration::min();
    avg_time = cmx_duration(0);
}

void CmxProfiler::begin_section(const char* section_name) {
    if (!is_local_enabled()) return;
    
    auto now = std::chrono::high_resolution_clock::now();
    state_.call_stack.emplace_back(section_name, now);
}

void CmxProfiler::end_section(const char* section_name) {
    if (!is_local_enabled()) return;
    
    auto now = std::chrono::high_resolution_clock::now();
    
    // Find matching begin in call stack
    for (auto it = state_.call_stack.rbegin(); it != state_.call_stack.rend(); ++it) {
        if (it->first == section_name) {
            // Calculate elapsed time
            auto elapsed = std::chrono::duration_cast<cmx_duration>(now - it->second);
            
            // Update statistics
            auto& stats = state_.stats[section_name];
            if (stats.section_name.empty()) {
                stats.section_name = section_name;
            }
            stats.update(elapsed);
            
            // Remove from call stack
            state_.call_stack.erase(std::next(it).base());
            return;
        }
    }
    
    // Section not found in call stack - this is an error but we'll handle it gracefully
    // Could be due to mismatched begin/end calls
}

const cmx_profile_stats* CmxProfiler::get_stats(const char* section_name) {
    auto it = state_.stats.find(section_name);
    return (it != state_.stats.end()) ? &it->second : nullptr;
}

std::vector<cmx_profile_stats> CmxProfiler::get_all_stats() {
    std::vector<cmx_profile_stats> result;
    result.reserve(state_.stats.size());
    
    for (const auto& pair : state_.stats) {
        result.push_back(pair.second);
    }
    
    // Sort by total time descending
    std::sort(result.begin(), result.end(), 
        [](const cmx_profile_stats& a, const cmx_profile_stats& b) {
            return a.total_time > b.total_time;
        });
    
    return result;
}

void CmxProfiler::clear_stats() {
    state_.stats.clear();
    state_.call_stack.clear();
}

std::string CmxProfiler::generate_report() {
    if (!global_enabled_) {
        return "Profiling is disabled globally";
    }
    
    if (state_.stats.empty()) {
        return "No profiling data available";
    }
    
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(3);
    
    oss << "\n=== CMatrix Profiling Report ===\n";
    oss << std::setw(25) << "Section" 
        << std::setw(10) << "Calls"
        << std::setw(12) << "Total (ms)"
        << std::setw(12) << "Avg (ms)"
        << std::setw(12) << "Min (ms)"
        << std::setw(12) << "Max (ms)" << "\n";
    oss << std::string(83, '-') << "\n";
    
    auto stats = get_all_stats();
    for (const auto& stat : stats) {
        double total_ms = stat.total_time.count() / 1e6;
        double avg_ms = stat.avg_time.count() / 1e6;
        double min_ms = (stat.min_time == cmx_duration::max()) ? 0.0 : stat.min_time.count() / 1e6;
        double max_ms = (stat.max_time == cmx_duration::min()) ? 0.0 : stat.max_time.count() / 1e6;
        
        oss << std::setw(25) << stat.section_name
            << std::setw(10) << stat.call_count
            << std::setw(12) << total_ms
            << std::setw(12) << avg_ms
            << std::setw(12) << min_ms
            << std::setw(12) << max_ms << "\n";
    }
    
    oss << std::string(83, '-') << "\n";
    return oss.str();
}

void CmxProfiler::print_report() {
    std::cout << generate_report() << std::endl;
}

// RAII scope implementation
CmxProfileScope::CmxProfileScope(const char* section_name) 
    : section_name_(section_name)
    , active_(CmxProfiler::is_local_enabled())
{
    if (active_) {
        CmxProfiler::begin_section(section_name_);
    }
}

CmxProfileScope::~CmxProfileScope() {
    if (active_) {
        CmxProfiler::end_section(section_name_);
    }
}

// C-style API implementations
void cmx_profiler_enable(bool flag) {
    CmxProfiler::enable(flag);
    CmxProfiler::enable_local(flag);
}

void cmx_profiler_begin(const char* section) {
    CmxProfiler::begin_section(section);
}

void cmx_profiler_end(const char* section) {
    CmxProfiler::end_section(section);
}

void cmx_profiler_report() {
    CmxProfiler::print_report();
}

void cmx_profiler_clear() {
    CmxProfiler::clear_stats();
}

#endif // CMX_PROFILING

} // namespace cmx