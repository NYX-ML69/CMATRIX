#include "cmx_error.hpp"
#include <iostream>
#include <sstream>
#include <cstdlib>

namespace cmx {

// Static member definitions
cmx_error_handler_t CmxErrorManager::error_handler_ = nullptr;
bool CmxErrorManager::abort_on_error_ = false;
cmx_error_level CmxErrorManager::min_level_ = cmx_error_level::WARNING;

// CmxException implementation
CmxException::CmxException(const char* file, cmx_u32 line, const char* function,
                          const std::string& message, cmx_status status, 
                          cmx_error_level level)
    : context_(file, line, function, message, status, level) {}

const char* CmxException::what() const noexcept {
    if (what_str_.empty()) {
        std::ostringstream oss;
        oss << "[" << cmx_error_level_to_string(context_.level) << "] "
            << cmx_status_to_string(context_.status) << ": " 
            << context_.message;
        
        if (context_.file && context_.function) {
            oss << " (at " << context_.function << " in " << context_.file 
                << ":" << context_.line << ")";
        }
        
        what_str_ = oss.str();
    }
    return what_str_.c_str();
}

// CmxErrorManager implementation
void CmxErrorManager::set_error_handler(cmx_error_handler_t handler) {
    error_handler_ = handler;
}

void CmxErrorManager::reset_error_handler() {
    error_handler_ = nullptr;
}

void CmxErrorManager::report_error(const char* file, cmx_u32 line, const char* function,
                                  const std::string& message, cmx_status status, 
                                  cmx_error_level level) {
    // Check minimum error level
    if (level < min_level_) {
        return;
    }
    
    // Create error context
    cmx_error_context context(file, line, function, message, status, level);
    
    // Call custom error handler if set
    if (error_handler_) {
        error_handler_(context);
    } else {
        default_error_handler(context);
    }
    
    // Abort on fatal errors or if configured to do so
    if (level == cmx_error_level::FATAL || abort_on_error_) {
        std::abort();
    }
}

void CmxErrorManager::default_error_handler(const cmx_error_context& context) {
    std::ostream& output = (context.level >= cmx_error_level::ERROR) ? std::cerr : std::cout;
    
    output << "[CMatrix " << cmx_error_level_to_string(context.level) << "] "
           << cmx_status_to_string(context.status) << ": " 
           << context.message;
    
    if (context.file && context.function) {
        output << "\n  at " << context.function << " (" << context.file 
               << ":" << context.line << ")";
    }
    
    output << std::endl;
}

// Status code to string conversion
const char* cmx_status_to_string(cmx_status status) {
    switch (status) {
        case cmx_status::OK: return "OK";
        
        // Memory errors
        case cmx_status::ERROR_ALLOC: return "ALLOC_ERROR";
        case cmx_status::ERROR_OUT_OF_MEMORY: return "OUT_OF_MEMORY";
        case cmx_status::ERROR_INVALID_POINTER: return "INVALID_POINTER";
        case cmx_status::ERROR_ALIGNMENT: return "ALIGNMENT_ERROR";
        
        // Shape and dimension errors
        case cmx_status::ERROR_SHAPE: return "SHAPE_ERROR";
        case cmx_status::ERROR_DIMENSION_MISMATCH: return "DIMENSION_MISMATCH";
        case cmx_status::ERROR_INVALID_DIMENSION: return "INVALID_DIMENSION";
        case cmx_status::ERROR_BROADCAST_ERROR: return "BROADCAST_ERROR";
        
        // Type errors
        case cmx_status::ERROR_TYPE: return "TYPE_ERROR";
        case cmx_status::ERROR_TYPE_MISMATCH: return "TYPE_MISMATCH";
        case cmx_status::ERROR_UNSUPPORTED_TYPE: return "UNSUPPORTED_TYPE";
        case cmx_status::ERROR_TYPE_CONVERSION: return "TYPE_CONVERSION_ERROR";
        
        // Computation errors
        case cmx_status::ERROR_COMPUTE: return "COMPUTE_ERROR";
        case cmx_status::ERROR_DIVISION_BY_ZERO: return "DIVISION_BY_ZERO";
        case cmx_status::ERROR_NUMERICAL_OVERFLOW: return "NUMERICAL_OVERFLOW";
        case cmx_status::ERROR_NUMERICAL_UNDERFLOW: return "NUMERICAL_UNDERFLOW";
        case cmx_status::ERROR_CONVERGENCE_FAILED: return "CONVERGENCE_FAILED";
        
        // Device and runtime errors
        case cmx_status::ERROR_DEVICE: return "DEVICE_ERROR";
        case cmx_status::ERROR_DEVICE_NOT_AVAILABLE: return "DEVICE_NOT_AVAILABLE";
        case cmx_status::ERROR_DEVICE_OUT_OF_MEMORY: return "DEVICE_OUT_OF_MEMORY";
        case cmx_status::ERROR_KERNEL_LAUNCH_FAILED: return "KERNEL_LAUNCH_FAILED";
        case cmx_status::ERROR_SYNCHRONIZATION: return "SYNCHRONIZATION_ERROR";
        
        // I/O and file errors
        case cmx_status::ERROR_IO: return "IO_ERROR";
        case cmx_status::ERROR_FILE_NOT_FOUND: return "FILE_NOT_FOUND";
        case cmx_status::ERROR_FILE_READ: return "FILE_READ_ERROR";
        case cmx_status::ERROR_FILE_WRITE: return "FILE_WRITE_ERROR";
        case cmx_status::ERROR_INVALID_FORMAT: return "INVALID_FORMAT";
        
        // Configuration and parameter errors
        case cmx_status::ERROR_CONFIG: return "CONFIG_ERROR";
        case cmx_status::ERROR_INVALID_PARAMETER: return "INVALID_PARAMETER";
        case cmx_status::ERROR_NOT_INITIALIZED: return "NOT_INITIALIZED";
        case cmx_status::ERROR_ALREADY_INITIALIZED: return "ALREADY_INITIALIZED";
        case cmx_status::ERROR_NOT_SUPPORTED: return "NOT_SUPPORTED";
        
        case cmx_status::ERROR_UNKNOWN:
        default: return "UNKNOWN_ERROR";
    }
}

const char* cmx_error_level_to_string(cmx_error_level level) {
    switch (level) {
        case cmx_error_level::INFO: return "INFO";
        case cmx_error_level::WARNING: return "WARNING";
        case cmx_error_level::ERROR: return "ERROR";
        case cmx_error_level::FATAL: return "FATAL";
        default: return "UNKNOWN";
    }
}

// C-style API implementations
void cmx_set_error_handler(cmx_error_handler_t handler) {
    CmxErrorManager::set_error_handler(handler);
}

void cmx_assert(bool condition, const char* message) {
    if (!condition) {
        CmxErrorManager::report_error(__FILE__, __LINE__, __FUNCTION__,
                                     std::string("Assertion failed: ") + message,
                                     cmx_status::ERROR_UNKNOWN, 
                                     cmx_error_level::FATAL);
    }
}

const char* cmx_get_error_string(cmx_status status) {
    return cmx_status_to_string(status);
}

} // namespace cmx