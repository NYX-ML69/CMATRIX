#include "cmx_debug.hpp"
#include <iostream>
#include <iomanip>
#include <sstream>
#include <chrono>
#include <thread>
#include <mutex>

namespace cmx {

// Static member definitions
cmx_debug_handler_t CmxDebugManager::debug_handler_ = nullptr;
cmx_debug_level CmxDebugManager::min_level_ = cmx_debug_level::INFO;
bool CmxDebugManager::include_timestamp_ = true;
bool CmxDebugManager::include_thread_id_ = false;
std::ostream* CmxDebugManager::output_stream_ = &std::cout;

// Thread-safe logging mutex
static std::mutex debug_mutex;

void CmxDebugManager::set_debug_handler(cmx_debug_handler_t handler) {
    std::lock_guard<std::mutex> lock(debug_mutex);
    debug_handler_ = handler;
}

void CmxDebugManager::reset_debug_handler() {
    std::lock_guard<std::mutex> lock(debug_mutex);
    debug_handler_ = nullptr;
}

void CmxDebugManager::log(cmx_debug_level level, const char* file, cmx_u32 line, 
                         const char* function, const std::string& message) {
    // Check if debug is enabled and level meets threshold
    if (!is_debug_enabled() || level < min_level_) {
        return;
    }
    
    std::lock_guard<std::mutex> lock(debug_mutex);
    
    if (debug_handler_) {
        debug_handler_(level, file, line, function, message);
    } else {
        default_debug_handler(level, file, line, function, message);
    }
}

void CmxDebugManager::trace(const char* file, cmx_u32 line, const char* function, 
                               const std::string& message) {
    log(cmx_debug_level::TRACE, file, line, function, message);
}

void CmxDebugManager::debug(const char* file, cmx_u32 line, const char* function, 
                           const std::string& message) {
    log(cmx_debug_level::DEBUG, file, line, function, message);
}

void CmxDebugManager::info(const char* file, cmx_u32 line, const char* function, 
                          const std::string& message) {
    log(cmx_debug_level::INFO, file, line, function, message);
}

void CmxDebugManager::warn(const char* file, cmx_u32 line, const char* function, 
                          const std::string& message) {
    log(cmx_debug_level::WARN, file, line, function, message);
}

void CmxDebugManager::error(const char* file, cmx_u32 line, const char* function, 
                           const std::string& message) {
    log(cmx_debug_level::ERROR, file, line, function, message);
}

void CmxDebugManager::fatal(const char* file, cmx_u32 line, const char* function, 
                           const std::string& message) {
    log(cmx_debug_level::FATAL, file, line, function, message);
}

void CmxDebugManager::default_debug_handler(cmx_debug_level level, const char* file, 
                                           cmx_u32 line, const char* function, 
                                           const std::string& message) {
    if (!output_stream_) return;
    
    std::ostringstream oss;
    
    // Add timestamp if enabled
    if (include_timestamp_) {
        oss << "[" << cmx_get_timestamp_string() << "] ";
    }
    
    // Add thread ID if enabled
    if (include_thread_id_) {
        oss << "[" << cmx_get_thread_id_string() << "] ";
    }
    
    // Add level with color coding
    oss << "[" << cmx_debug_level_to_string(level) << "] ";
    
    // Add message
    oss << message;
    
    // Add location info for higher levels
    if (level >= cmx_debug_level::WARN && file && function) {
        oss << " (at " << function << " in " << file << ":" << line << ")";
    }
    
    *output_stream_ << oss.str() << std::endl;
    output_stream_->flush();
}

// Utility functions
const char* cmx_debug_level_to_string(cmx_debug_level level) {
    switch (level) {
        case cmx_debug_level::TRACE: return "TRACE";
        case cmx_debug_level::DEBUG: return "DEBUG";
        case cmx_debug_level::INFO: return "INFO";
        case cmx_debug_level::WARN: return "WARN";
        case cmx_debug_level::ERROR: return "ERROR";
        case cmx_debug_level::FATAL: return "FATAL";
        default: return "UNKNOWN";
    }
}

const char* cmx_debug_level_to_color_code(cmx_debug_level level) {
    switch (level) {
        case cmx_debug_level::TRACE: return "\033[37m";  // White
        case cmx_debug_level::DEBUG: return "\033[36m";  // Cyan
        case cmx_debug_level::INFO: return "\033[32m";   // Green
        case cmx_debug_level::WARN: return "\033[33m";   // Yellow
        case cmx_debug_level::ERROR: return "\033[31m";  // Red
        case cmx_debug_level::FATAL: return "\033[35m";  // Magenta
        default: return "\033[0m";                       // Reset
    }
}

std::string cmx_get_timestamp_string() {
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()) % 1000;
    
    std::ostringstream oss;
    oss << std::put_time(std::localtime(&time_t), "%H:%M:%S");
    oss << "." << std::setfill('0') << std::setw(3) << ms.count();
    
    return oss.str();
}

std::string cmx_get_thread_id_string() {
    std::ostringstream oss;
    oss << std::this_thread::get_id();
    return oss.str();
}

#if CMX_DEBUG
// Debug timer implementation
CmxDebugTimer::CmxDebugTimer(const std::string& name) 
    : name_(name) {
    start_time_ = std::chrono::high_resolution_clock::now();
    CMX_LOG_DEBUG("Timer '" + name_ + "' started");
}

CmxDebugTimer::~CmxDebugTimer() {
    auto elapsed = elapsed_ms();
    CMX_LOG_DEBUG("Timer '" + name_ + "' finished: " + std::to_string(elapsed) + " ms");
}

void CmxDebugTimer::reset() {
    start_time_ = std::chrono::high_resolution_clock::now();
}

cmx_f64 CmxDebugTimer::elapsed_ms() const {
    auto now = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(now - start_time_);
    return duration.count() / 1e6;  // Convert to milliseconds
}
#endif

// Always-available debug functions
void cmx_debug_log(const char* file, cmx_u32 line, const char* function, 
                  const std::string& message) {
    CmxDebugManager::info(file, line, function, message);
}

void cmx_debug_set_level(cmx_debug_level level) {
    CmxDebugManager::set_min_debug_level(level);
}

void cmx_debug_enable_colors(bool enable) {
    // This could be implemented to enable/disable color output
    // For now, it's a placeholder
    (void)enable;
}

// Memory tracker implementation
cmx_size CmxMemoryTracker::total_allocated_ = 0;
cmx_size CmxMemoryTracker::peak_allocated_ = 0;
cmx_size CmxMemoryTracker::allocation_count_ = 0;

static std::mutex memory_mutex;

void CmxMemoryTracker::track_allocation(cmx_size size) {
    std::lock_guard<std::mutex> lock(memory_mutex);
    total_allocated_ += size;
    allocation_count_++;
    
    if (total_allocated_ > peak_allocated_) {
        peak_allocated_ = total_allocated_;
    }
}

void CmxMemoryTracker::track_deallocation(cmx_size size) {
    std::lock_guard<std::mutex> lock(memory_mutex);
    if (total_allocated_ >= size) {
        total_allocated_ -= size;
    }
}

void CmxMemoryTracker::reset_stats() {
    std::lock_guard<std::mutex> lock(memory_mutex);
    total_allocated_ = 0;
    peak_allocated_ = 0;
    allocation_count_ = 0;
}

void CmxMemoryTracker::print_stats() {
    std::lock_guard<std::mutex> lock(memory_mutex);
    
    std::cout << "\n=== CMatrix Memory Statistics ===\n";
    std::cout << "Current allocated: " << total_allocated_ << " bytes\n";
    std::cout << "Peak allocated: " << peak_allocated_ << " bytes\n";
    std::cout << "Total allocations: " << allocation_count_ << "\n";
    std::cout << "================================\n" << std::endl;
}

} // namespace cmx