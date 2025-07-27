#pragma once

#include <chrono>
#include <cstdint>
#include <cstring>
#include <mutex>

// Compile-time flag to enable/disable profiling
#ifndef CMX_PROFILING_ENABLED
#define CMX_PROFILING_ENABLED 1
#endif

// Convenience macros for profiling
#if CMX_PROFILING_ENABLED
#define CMX_PROFILE_START(profiler, label) (profiler).start(label)
#define CMX_PROFILE_STOP(profiler, label) (profiler).stop(label)
#define CMX_PROFILE_SCOPE(profiler, label) cmx::runtime::ProfileScope _prof_scope(profiler, label)
#else
#define CMX_PROFILE_START(profiler, label) do {} while(0)
#define CMX_PROFILE_STOP(profiler, label) do {} while(0)
#define CMX_PROFILE_SCOPE(profiler, label) do {} while(0)
#endif

namespace cmx {
namespace runtime {

/**
 * @brief High-resolution timer for profiling
 */
class HighResTimer {
private:
    using Clock = std::chrono::high_resolution_clock;
    using TimePoint = Clock::time_point;
    
public:
    /**
     * @brief Get current time point
     * @return Current time point
     */
    static TimePoint now() {
        return Clock::now();
    }
    
    /**
     * @brief Calculate duration in microseconds
     * @param start Start time point
     * @param end End time point
     * @return Duration in microseconds
     */
    static uint64_t duration_us(const TimePoint& start, const TimePoint& end) {
        return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    }
    
    /**
     * @brief Calculate duration in nanoseconds
     * @param start Start time point
     * @param end End time point
     * @return Duration in nanoseconds
     */
    static uint64_t duration_ns(const TimePoint& start, const TimePoint& end) {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    }
};

/**
 * @brief Profile entry for storing timing information
 */
struct ProfileEntry {
    static constexpr size_t MAX_LABEL_LENGTH = 64;
    
    char label[MAX_LABEL_LENGTH];           ///< Operation label
    uint64_t total_time_us;                 ///< Total execution time in microseconds
    uint64_t min_time_us;                   ///< Minimum execution time in microseconds
    uint64_t max_time_us;                   ///< Maximum execution time in microseconds
    uint32_t call_count;                    ///< Number of times this operation was called
    bool is_active;                         ///< Whether this entry is currently being timed
    HighResTimer::TimePoint start_time;     ///< Start time for current measurement
    
    /**
     * @brief Default constructor
     */
    ProfileEntry() : total_time_us(0), min_time_us(UINT64_MAX), max_time_us(0), 
                     call_count(0), is_active(false) {
        std::memset(label, 0, MAX_LABEL_LENGTH);
    }
    
    /**
     * @brief Constructor with label
     * @param op_label Operation label
     */
    explicit ProfileEntry(const char* op_label) : ProfileEntry() {
        set_label(op_label);
    }
    
    /**
     * @brief Set operation label
     * @param op_label Operation label (will be truncated if too long)
     */
    void set_label(const char* op_label) {
        if (op_label) {
            std::strncpy(label, op_label, MAX_LABEL_LENGTH - 1);
            label[MAX_LABEL_LENGTH - 1] = '\0';
        }
    }
    
    /**
     * @brief Start timing this operation
     */
    void start() {
        if (!is_active) {
            start_time = HighResTimer::now();
            is_active = true;
        }
    }
    
    /**
     * @brief Stop timing this operation
     */
    void stop() {
        if (is_active) {
            auto end_time = HighResTimer::now();
            uint64_t duration = HighResTimer::duration_us(start_time, end_time);
            
            total_time_us += duration;
            min_time_us = (duration < min_time_us) ? duration : min_time_us;
            max_time_us = (duration > max_time_us) ? duration : max_time_us;
            call_count++;
            is_active = false;
        }
    }
    
    /**
     * @brief Get average execution time
     * @return Average time in microseconds
     */
    uint64_t get_average_time_us() const {
        return (call_count > 0) ? (total_time_us / call_count) : 0;
    }
    
    /**
     * @brief Reset all timing data
     */
    void reset() {
        total_time_us = 0;
        min_time_us = UINT64_MAX;
        max_time_us = 0;
        call_count = 0;
        is_active = false;
    }
};

/**
 * @brief Profiling report structure
 */
struct ProfileReport {
    static constexpr size_t MAX_ENTRIES = 128;
    
    ProfileEntry entries[MAX_ENTRIES];      ///< Array of profile entries
    size_t entry_count;                     ///< Number of valid entries
    uint64_t total_profiling_time_us;       ///< Total time spent profiling
    HighResTimer::TimePoint report_start_time; ///< When profiling session started
    
    /**
     * @brief Default constructor
     */
    ProfileReport() : entry_count(0), total_profiling_time_us(0) {
        report_start_time = HighResTimer::now();
    }
    
    /**
     * @brief Add an entry to the report
     * @param entry Profile entry to add
     * @return true if added successfully, false if report is full
     */
    bool add_entry(const ProfileEntry& entry) {
        if (entry_count < MAX_ENTRIES) {
            entries[entry_count++] = entry;
            return true;
        }
        return false;
    }
    
    /**
     * @brief Sort entries by total time (descending)
     */
    void sort_by_total_time();
    
    /**
     * @brief Sort entries by average time (descending)
     */
    void sort_by_average_time();
    
    /**
     * @brief Sort entries by call count (descending)
     */
    void sort_by_call_count();
};

/**
 * @brief Runtime profiling tool for performance analysis
 * 
 * This profiler provides:
 * - Per-operation timing with start/stop API
 * - Statistical analysis (min, max, average, total time)
 * - Call count tracking
 * - Thread-safe operation
 * - Minimal overhead when disabled
 */
class CMXProfiler {
private:
    static constexpr size_t MAX_PROFILE_ENTRIES = 128;
    
    ProfileEntry entries_[MAX_PROFILE_ENTRIES]; ///< Profile entries storage
    size_t entry_count_;                        ///< Number of active entries
    bool is_enabled_;                           ///< Whether profiling is enabled
    std::mutex profile_mutex_;                  ///< Mutex for thread safety
    HighResTimer::TimePoint session_start_time_; ///< Session start time
    
    /**
     * @brief Find profile entry by label
     * @param label Operation label
     * @return Pointer to entry if found, nullptr otherwise
     */
    ProfileEntry* find_entry(const char* label);
    
    /**
     * @brief Create new profile entry
     * @param label Operation label
     * @return Pointer to new entry if created, nullptr if storage is full
     */
    ProfileEntry* create_entry(const char* label);

public:
    /**
     * @brief Constructor
     * @param enabled Whether profiling is initially enabled
     */
    explicit CMXProfiler(bool enabled = true);
    
    /**
     * @brief Destructor
     */
    ~CMXProfiler();
    
    /**
     * @brief Enable or disable profiling
     * @param enabled Whether to enable profiling
     */
    void set_enabled(bool enabled);
    
    /**
     * @brief Check if profiling is enabled
     * @return true if profiling is enabled
     */
    bool is_enabled() const { return is_enabled_; }
    
    /**
     * @brief Start timing an operation
     * @param label Operation label
     */
    void start(const char* label);
    
    /**
     * @brief Stop timing an operation
     * @param label Operation label
     */
    void stop(const char* label);
    
    /**
     * @brief Get current profiling report
     * @param report Output report structure
     * @return true if report generated successfully
     */
    bool get_report(ProfileReport& report) const;
    
    /**
     * @brief Reset all profiling data
     */
    void reset();
    
    /**
     * @brief Get total session time
     * @return Total time since profiler was created/reset in microseconds
     */
    uint64_t get_session_time_us() const;
    
    /**
     * @brief Get number of active profile entries
     * @return Number of operations being profiled
     */
    size_t get_entry_count() const { return entry_count_; }
    
    /**
     * @brief Check if profiler has capacity for more entries
     * @return true if more entries can be added
     */
    bool has_capacity() const { return entry_count_ < MAX_PROFILE_ENTRIES; }
    
    /**
     * @brief Get entry by label
     * @param label Operation label
     * @return Pointer to entry if found, nullptr otherwise
     */
    const ProfileEntry* get_entry(const char* label) const;
    
    /**
     * @brief Print report to string buffer
     * @param buffer Output buffer
     * @param buffer_size Size of output buffer
     * @param sort_by_total_time If true, sort by total time; otherwise by average time
     * @return Number of characters written (excluding null terminator)
     */
    size_t print_report(char* buffer, size_t buffer_size, bool sort_by_total_time = true) const;
};

/**
 * @brief RAII scope profiler for automatic start/stop
 */
class ProfileScope {
private:
    CMXProfiler& profiler_;
    const char* label_;
    
public:
    /**
     * @brief Constructor - starts profiling
     * @param profiler Reference to profiler instance
     * @param label Operation label
     */
    ProfileScope(CMXProfiler& profiler, const char* label) 
        : profiler_(profiler), label_(label) {
        profiler_.start(label_);
    }
    
    /**
     * @brief Destructor - stops profiling
     */
    ~ProfileScope() {
        profiler_.stop(label_);
    }
    
    // Disable copy and move
    ProfileScope(const ProfileScope&) = delete;
    ProfileScope& operator=(const ProfileScope&) = delete;
    ProfileScope(ProfileScope&&) = delete;
    ProfileScope& operator=(ProfileScope&&) = delete;
};

} // namespace runtime
} // namespace cmx