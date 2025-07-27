#pragma once

#include <cstddef>
#include <cstdint>

/**
 * @file cmx_api.hpp
 * @brief Core API definitions for CMatrix runtime
 * 
 * Contains fundamental types, enums, status codes, and handles
 * used across all CMatrix APIs. This provides the stable
 * foundation for the public interface.
 */

namespace cmx {

/**
 * @brief Status codes returned by CMatrix API functions
 */
enum class cmx_status {
    OK = 0,                    ///< Operation completed successfully
    ERROR = 1,                 ///< General error occurred
    INVALID_MODEL = 2,         ///< Model data is invalid or corrupted
    INVALID_HANDLE = 3,        ///< Model handle is null or invalid
    MEMORY_ERROR = 4,          ///< Memory allocation failed
    IO_ERROR = 5,              ///< File I/O operation failed
    NOT_INITIALIZED = 6,       ///< Runtime not initialized
    ALREADY_INITIALIZED = 7,   ///< Runtime already initialized
    UNSUPPORTED_VERSION = 8,   ///< Model version not supported
    RUNTIME_ERROR = 9          ///< Runtime execution error
};

/**
 * @brief Opaque handle to a loaded model
 */
using cmx_model_handle = void*;

/**
 * @brief Invalid model handle constant
 */
constexpr cmx_model_handle CMX_INVALID_HANDLE = nullptr;

/**
 * @brief Model metadata structure
 */
struct cmx_model_info {
    const char* name;          ///< Model name
    const char* version;       ///< Model version string
    uint32_t input_count;      ///< Number of input tensors
    uint32_t output_count;     ///< Number of output tensors
    size_t memory_required;    ///< Required memory in bytes
};

/**
 * @brief Tensor descriptor
 */
struct cmx_tensor_desc {
    const char* name;          ///< Tensor name
    uint32_t* shape;           ///< Tensor dimensions
    uint32_t rank;             ///< Number of dimensions
    size_t element_size;       ///< Size of each element in bytes
    size_t total_size;         ///< Total tensor size in bytes
};

/**
 * @brief Convert status code to human-readable string
 * @param status Status code to convert
 * @return String representation of the status
 */
const char* cmx_status_to_string(cmx_status status);

/**
 * @brief Check if status indicates success
 * @param status Status code to check
 * @return true if status is OK, false otherwise
 */
inline bool cmx_is_success(cmx_status status) {
    return status == cmx_status::OK;
}

/**
 * @brief Check if handle is valid
 * @param handle Model handle to check
 * @return true if handle is valid, false otherwise
 */
inline bool cmx_is_valid_handle(cmx_model_handle handle) {
    return handle != CMX_INVALID_HANDLE;
}

} // namespace cmx