#include "cmx_profiler.hpp"
#include <algorithm>
#include <cstdio>

namespace cmx {
namespace runtime {

void ProfileReport::sort_by_total_time() {
    std::sort(entries, entries + entry_count, 
              [](const ProfileEntry& a, const ProfileEntry& b) {
                  return a.total_time_us > b.total_time_us;
              });
}

void ProfileReport::sort_by_average_time() {
    std::sort(entries, entries + entry_count,
              [](const ProfileEntry& a, const ProfileEntry& b) {
                  return a.get_average_time_us() > b.get_average_time_us();
              });
}

void ProfileReport::sort_by_call_count() {
    std::sort(entries, entries + entry_count,
              [](const ProfileEntry& a, const ProfileEntry& b) {
                  return a.call_count > b.call_count;
              });
}

CMXProfiler::CMXProfiler(bool enabled) 
    : entry_count_(0), is_enabled_(enabled) {
    session_start_time_ = HighResTimer::now();
}

CMXProfiler::~CMXProfiler() {
    // Nothing to cleanup - using stack allocation
}

void CMXProfiler::set_enabled(bool enabled) {
    std::lock_guard<std::mutex> lock(profile_mutex_);
    is_enabled_ = enabled;
}

void CMXProfiler::start(const char* label) {
#if CMX_PROFILING_ENABLED
    if (!is_enabled_ || !label) {
        return;
    }
    
    std::lock_guard<std::mutex> lock(profile_mutex_);
    
    ProfileEntry* entry = find_entry(label);
    if (!entry) {
        entry = create_entry(label);
    }
    
    if (entry) {
        entry->start();
    }
#endif
}

void CMXProfiler::stop(const char* label) {
#if CMX_PROFILING_ENABLED
    if (!is_enabled_ || !label) {
        return;
    }
    
    std::lock_guard<std::mutex> lock(profile_mutex_);
    
    ProfileEntry* entry = find_entry(label);
    if (entry) {
        entry->stop();
    }
#endif
}

bool CMXProfiler::get_report(ProfileReport& report) const {
    if (!is_enabled_) {
        return false;
    }
    
    std::lock_guard<std::mutex> lock(profile_mutex_);
    
    report = ProfileReport();
    
    // Copy all entries to report
    for (size_t i = 0; i < entry_count_; ++i) {
        if (!report.add_entry(entries_[i])) {
            break; // Report is full
        }
    }
    
    // Calculate total profiling time
    auto current_time = HighResTimer::now();
    report.total_profiling_time_us = HighResTimer::duration_us(session_start_time_, current_time);
    report.report_start_time = session_start_time_;
    
    return true;
}

void CMXProfiler::reset() {
    std::lock_guard<std::mutex> lock(profile_mutex_);
    
    // Reset all entries
    for (size_t i = 0; i < entry_count_; ++i) {
        entries_[i].reset();
    }
    
    entry_count_ = 0;
    session_start_time_ = HighResTimer::now();
}

uint64_t CMXProfiler::get_session_time_us() const {
    auto current_time = HighResTimer::now();
    return HighResTimer::duration_us(session_start_time_, current_time);
}

const ProfileEntry* CMXProfiler::get_entry(const char* label) const {
    if (!label) {
        return nullptr;
    }
    
    std::lock_guard<std::mutex> lock(profile_mutex_);
    
    for (size_t i = 0; i < entry_count_; ++i) {
        if (std::strcmp(entries_[i].label, label) == 0) {
            return &entries_[i];
        }
    }
    
    return nullptr;
}

size_t CMXProfiler::print_report(char* buffer, size_t buffer_size, bool sort_by_total_time) const {
    if (!buffer || buffer_size == 0) {
        return 0;
    }
    
    ProfileReport report;
    if (!get_report(report)) {
        return 0;
    }
    
    // Sort report
    if (sort_by_total_time) {
        report.sort_by_total_time();
    } else {
        report.sort_by_average_time();
    }
    
    size_t written = 0;
    
    // Header
    written += std::snprintf(buffer + written, buffer_size - written,
                            "=== CMX Profiler Report ===\n");
    written += std::snprintf(buffer + written, buffer_size - written,
                            "Session Time: %llu us\n", 
                            static_cast<unsigned long long>(report.total_profiling_time_us));
    written += std::snprintf(buffer + written, buffer_size - written,
                            "Active Entries: %zu\n\n", report.entry_count);
    
    // Column headers
    written += std::snprintf(buffer + written, buffer_size - written,
                            "%-32s %10s %10s %10s %10s %10s\n",
                            "Operation", "Calls", "Total(us)", "Avg(us)", "Min(us)", "Max(us)");
    written += std::snprintf(buffer + written, buffer_size - written,
                            "%-32s %10s %10s %10s %10s %10s\n",
                            "--------------------------------", 
                            "----------", "----------", "----------", "----------", "----------");
    
    // Entries
    for (size_t i = 0; i < report.entry_count && written < buffer_size - 1; ++i) {
        const ProfileEntry& entry = report.entries[i];
        
        if (entry.call_count > 0) {
            written += std::snprintf(buffer + written, buffer_size - written,
                                    "%-32s %10u %10llu %10llu %10llu %10llu\n",
                                    entry.label,
                                    entry.call_count,
                                    static_cast<unsigned long long>(entry.total_time_us),
                                    static_cast<unsigned long long>(entry.get_average_time_us()),
                                    static_cast<unsigned long long>(entry.min_time_us),
                                    static_cast<unsigned long long>(entry.max_time_us));
        }
    }
    
    // Footer
    written += std::snprintf(buffer + written, buffer_size - written, "\n");
    
    // Ensure null termination
    if (written < buffer_size) {
        buffer[written] = '\0';
    } else {
        buffer[buffer_size - 1] = '\0';
        written = buffer_size - 1;
    }
    
    return written;
}

ProfileEntry* CMXProfiler::find_entry(const char* label) {
    if (!label) {
        return nullptr;
    }
    
    for (size_t i = 0; i < entry_count_; ++i) {
        if (std::strcmp(entries_[i].label, label) == 0) {
            return &entries_[i];
        }
    }
    
    return nullptr;
}

ProfileEntry* CMXProfiler::create_entry(const char* label) {
    if (!label || entry_count_ >= MAX_PROFILE_ENTRIES) {
        return nullptr;
    }
    
    ProfileEntry* entry = &entries_[entry_count_++];
    entry->set_label(label);
    
    return entry;
}

} // namespace runtime
} // namespace cmx