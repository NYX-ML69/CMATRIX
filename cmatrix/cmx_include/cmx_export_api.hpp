#pragma once

#include "cmx_api.hpp"

/**
 * @file cmx_export_api.hpp
 * @brief Model export and serialization APIs
 * 
 * Provides functions to export compiled models, runtime graphs,
 * and optimization data from the C++ runtime layer. Enables
 * model persistence and cross-platform deployment.
 */

namespace cmx {

/**
 * @brief Export format enumeration
 */
enum class cmx_export_format {
    BINARY = 0,           ///< Native binary format
    JSON = 1,             ///< JSON representation
    PROTOBUF = 2,         ///< Protocol buffer format
    ONNX = 3              ///< ONNX format (if supported)
};

/**
 * @brief Export options structure
 */
struct cmx_export_options {
    cmx_export_format format;     ///< Output format
    bool include_weights;         ///< Include model weights
    bool include_metadata;        ///< Include model metadata
    bool optimize_for_size;       ///< Optimize for smaller file size
    bool include_profiling_data;  ///< Include profiling information
    const char* encryption_key;   ///< Optional encryption key
    uint32_t compression_level;   ///< Compression level (0-9, 0=none)
};

/**
 * @brief Export progress callback function type
 * @param progress Progress percentage (0.0 to 1.0)
 * @param user_data User-provided data pointer
 */
typedef void (*cmx_export_progress_callback)(float progress, void* user_data);

/**
 * @brief Export a model to file with default options
 * @param handle Model handle to export
 * @param file_path Output file path
 * @return true on success, false on failure
 */
bool cmx_export_model(cmx_model_handle handle, const char* file_path);

/**
 * @brief Export a model to file with custom options
 * @param handle Model handle to export
 * @param file_path Output file path
 * @param options Export options structure
 * @return Status code indicating export result
 */
cmx_status cmx_export_model_with_options(cmx_model_handle handle, 
                                       const char* file_path, 
                                       const cmx_export_options* options);

/**
 * @brief Export a model to memory buffer
 * @param handle Model handle to export
 * @param buffer Pointer to buffer pointer (allocated by function)
 * @param size Pointer to size variable (set by function)
 * @param options Export options structure (optional)
 * @return Status code indicating export result
 */
cmx_status cmx_export_model_to_buffer(cmx_model_handle handle,
                                    void** buffer,
                                    size_t* size,
                                    const cmx_export_options* options);

/**
 * @brief Export a model with progress callback
 * @param handle Model handle to export
 * @param file_path Output file path
 * @param options Export options structure
 * @param callback Progress callback function
 * @param user_data User data passed to callback
 * @return Status code indicating export result
 */
cmx_status cmx_export_model_with_progress(cmx_model_handle handle,
                                        const char* file_path,
                                        const cmx_export_options* options,
                                        cmx_export_progress_callback callback,
                                        void* user_data);

/**
 * @brief Free buffer allocated by cmx_export_model_to_buffer
 * @param buffer Buffer pointer to free
 */
void cmx_free_export_buffer(void* buffer);

/**
 * @brief Get default export options for a format
 * @param format Target export format
 * @param options Pointer to options structure to populate
 * @return Status code indicating success or failure
 */
cmx_status cmx_get_default_export_options(cmx_export_format format, cmx_export_options* options);

/**
 * @brief Validate export options for compatibility
 * @param handle Model handle
 * @param options Export options to validate
 * @return Status code indicating validation result
 */
cmx_status cmx_validate_export_options(cmx_model_handle handle, const cmx_export_options* options);

/**
 * @brief Get estimated export file size
 * @param handle Model handle
 * @param options Export options
 * @param estimated_size Pointer to size estimate (in bytes)
 * @return Status code indicating success or failure
 */
cmx_status cmx_estimate_export_size(cmx_model_handle handle, 
                                  const cmx_export_options* options,
                                  size_t* estimated_size);

/**
 * @brief Export model runtime graph (for debugging/visualization)
 * @param handle Model handle
 * @param file_path Output file path
 * @param format Graph format (JSON or DOT)
 * @return Status code indicating export result
 */
cmx_status cmx_export_runtime_graph(cmx_model_handle handle,
                                  const char* file_path,
                                  const char* format);

/**
 * @brief Export model profiling data
 * @param handle Model handle
 * @param file_path Output file path
 * @return Status code indicating export result
 */
cmx_status cmx_export_profiling_data(cmx_model_handle handle, const char* file_path);

} // namespace cmx