#pragma once

#include "cmx_types.hpp"
#include <string>
#include <exception>

namespace cmx {

// Error status codes
enum class cmx_status : cmx_u32 {
    OK = 0,
    
    // Memory errors
    ERROR_ALLOC = 1,
    ERROR_OUT_OF_MEMORY = 2,
    ERROR_INVALID_POINTER = 3,
    ERROR_ALIGNMENT = 4,
    
    // Shape and dimension errors
    ERROR_SHAPE = 10,
    ERROR_DIMENSION_MISMATCH = 11,
    ERROR_INVALID_DIMENSION = 12,
    ERROR_BROADCAST_ERROR = 13,
    
    // Type errors
    ERROR_TYPE = 20,
    ERROR_TYPE_MISMATCH = 21,
    ERROR_UNSUPPORTED_TYPE = 22,
    ERROR_TYPE_CONVERSION = 23,
    
    // Computation errors
    ERROR_COMPUTE = 30,
    ERROR_DIVISION_BY_ZERO = 31,
    ERROR_NUMERICAL_OVERFLOW = 32,
    ERROR_NUMERICAL_UNDERFLOW = 33,
    ERROR_CONVERGENCE_FAILED = 34,
    
    // Device and runtime errors
    ERROR_DEVICE = 40,
    ERROR_DEVICE_NOT_AVAILABLE = 41,
    ERROR_DEVICE_OUT_OF_MEMORY = 42,
    ERROR_KERNEL_LAUNCH_FAILED = 43,
    ERROR_SYNCHRONIZATION = 44,
    
    // I/O and file errors
    ERROR_IO = 50,
    ERROR_FILE_NOT_FOUND = 51,
    ERROR_FILE_READ = 52,
    ERROR_FILE_WRITE = 53,
    ERROR_INVALID_FORMAT = 54,
    
    // Configuration and parameter errors
    ERROR_CONFIG = 60,
    ERROR_INVALID_PARAMETER = 61,
    ERROR_NOT_INITIALIZED = 62,
    ERROR_ALREADY_INITIALIZED = 63,
    ERROR_NOT_SUPPORTED = 64,
    
    // Generic errors
    ERROR_UNKNOWN = 255
};

// Error severity levels
enum class cmx_error_level : cmx_u8 {
    INFO = 0,
    WARNING = 1,
    ERROR = 2,
    FATAL = 3
};

// Error context information
struct cmx_error_context {
    const char* file;
    cmx_u32 line;
    const char* function;
    std::string message;
    cmx_status status;
    cmx_error_level level;
    
    cmx_error_context(const char* f, cmx_u32 l, const char* func, 
                     const std::string& msg, cmx_status s, cmx_error_level lvl)
        : file(f), line(l), function(func), message(msg), status(s), level(lvl) {}
};

// Exception class for CMatrix errors
class CmxException : public std::exception {
private:
    cmx_error_context context_;
    mutable std::string what_str_;
    
public:
    CmxException(const char* file, cmx_u32 line, const char* function,
                const std::string& message, cmx_status status, 
                cmx_error_level level = cmx_error_level::ERROR);
    
    const char* what() const noexcept override;
    
    // Accessors
    cmx_status status() const { return context_.status; }
    cmx_error_level level() const { return context_.level; }
    const cmx_error_context& context() const { return context_; }
    const std::string& message() const { return context_.message; }
};

// Error handler function type
using cmx_error_handler_t = void(*)(const cmx_error_context& context);

// Error management class
class CmxErrorManager {
private:
    static cmx_error_handler_t error_handler_;
    static bool abort_on_error_;
    static cmx_error_level min_level_;
    
public:
    // Error handler management
    static void set_error_handler(cmx_error_handler_t handler);
    static cmx_error_handler_t get_error_handler() { return error_handler_; }
    static void reset_error_handler();
    
    // Error reporting behavior
    static void set_abort_on_error(bool abort) { abort_on_error_ = abort; }
    static bool get_abort_on_error() { return abort_on_error_; }
    
    static void set_min_error_level(cmx_error_level level) { min_level_ = level; }
    static cmx_error_level get_min_error_level() { return min_level_; }
    
    // Error reporting
    static void report_error(const char* file, cmx_u32 line, const char* function,
                           const std::string& message, cmx_status status, 
                           cmx_error_level level = cmx_error_level::ERROR);
    
    // Default error handler
    static void default_error_handler(const cmx_error_context& context);
};

// Utility functions
const char* cmx_status_to_string(cmx_status status);
const char* cmx_error_level_to_string(cmx_error_level level);

// Error reporting macros
#define CMX_ERROR(status, message) \
    do { \
        cmx::CmxErrorManager::report_error(__FILE__, __LINE__, __FUNCTION__, \
                                          message, status, cmx::cmx_error_level::ERROR); \
    } while(0)

#define CMX_WARNING(status, message) \
    do { \
        cmx::CmxErrorManager::report_error(__FILE__, __LINE__, __FUNCTION__, \
                                          message, status, cmx::cmx_error_level::WARNING); \
    } while(0)

#define CMX_FATAL(status, message) \
    do { \
        cmx::CmxErrorManager::report_error(__FILE__, __LINE__, __FUNCTION__, \
                                          message, status, cmx::cmx_error_level::FATAL); \
    } while(0)

#define CMX_INFO(status, message) \
    do { \
        cmx::CmxErrorManager::report_error(__FILE__, __LINE__, __FUNCTION__, \
                                          message, status, cmx::cmx_error_level::INFO); \
    } while(0)

// Exception throwing macros
#define CMX_THROW(status, message) \
    throw cmx::CmxException(__FILE__, __LINE__, __FUNCTION__, message, status)

#define CMX_THROW_IF(condition, status, message) \
    do { \
        if (condition) { \
            CMX_THROW(status, message); \
        } \
    } while(0)

// Assertion macros
#ifdef CMX_DEBUG
#define CMX_ASSERT(condition, message) \
    do { \
        if (!(condition)) { \
            CMX_FATAL(cmx::cmx_status::ERROR_UNKNOWN, \
                     std::string("Assertion failed: ") + #condition + " - " + message); \
        } \
    } while(0)
#else
#define CMX_ASSERT(condition, message) do {} while(0)
#endif

// Bounds checking macros
#if CMX_BOUNDS_CHECK
#define CMX_CHECK_BOUNDS(index, size, message) \
    do { \
        if ((index) >= (size)) { \
            CMX_THROW(cmx::cmx_status::ERROR_INVALID_PARAMETER, \
                     std::string("Bounds check failed: ") + message); \
        } \
    } while(0)
#else
#define CMX_CHECK_BOUNDS(index, size, message) do {} while(0)
#endif

// Status checking macros
#define CMX_CHECK_STATUS(status) \
    do { \
        if ((status) != cmx::cmx_status::OK) { \
            CMX_THROW(status, "Operation failed"); \
        } \
    } while(0)

#define CMX_RETURN_IF_ERROR(status) \
    do { \
        if ((status) != cmx::cmx_status::OK) { \
            return status; \
        } \
    } while(0)

// C-style error handling functions
void cmx_set_error_handler(cmx_error_handler_t handler);
void cmx_assert(bool condition, const char* message);
const char* cmx_get_error_string(cmx_status status);

} // namespace cmx