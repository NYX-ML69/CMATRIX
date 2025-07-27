#pragma once

#include "cmx_api.hpp"

/**
 * @file cmx_model_loader.hpp
 * @brief Model loading and management APIs
 * 
 * Provides functions to load, deserialize, validate, and manage
 * CMatrix models at runtime. Handles model lifecycle from
 * binary data to executable runtime representation.
 */

namespace cmx {

/**
 * @brief Load a model from binary data
 * @param data Pointer to model binary data
 * @param size Size of the binary data in bytes
 * @return Model handle on success, CMX_INVALID_HANDLE on failure
 */
cmx_model_handle cmx_load_model(const void* data, size_t size);

/**
 * @brief Load a model from file path
 * @param file_path Path to the model file
 * @return Model handle on success, CMX_INVALID_HANDLE on failure
 */
cmx_model_handle cmx_load_model_from_file(const char* file_path);

/**
 * @brief Free a loaded model and release its resources
 * @param handle Model handle to free
 * @return Status code indicating success or failure
 */
cmx_status cmx_free_model(cmx_model_handle handle);

/**
 * @brief Get model information and metadata
 * @param handle Model handle
 * @param info Pointer to info structure to populate
 * @return Status code indicating success or failure
 */
cmx_status cmx_get_model_info(cmx_model_handle handle, cmx_model_info* info);

/**
 * @brief Get input tensor descriptor by index
 * @param handle Model handle
 * @param index Input tensor index (0-based)
 * @param desc Pointer to descriptor structure to populate
 * @return Status code indicating success or failure
 */
cmx_status cmx_get_input_desc(cmx_model_handle handle, uint32_t index, cmx_tensor_desc* desc);

/**
 * @brief Get output tensor descriptor by index
 * @param handle Model handle
 * @param index Output tensor index (0-based)
 * @param desc Pointer to descriptor structure to populate
 * @return Status code indicating success or failure
 */
cmx_status cmx_get_output_desc(cmx_model_handle handle, uint32_t index, cmx_tensor_desc* desc);

/**
 * @brief Validate model integrity and compatibility
 * @param handle Model handle
 * @return Status code indicating validation result
 */
cmx_status cmx_validate_model(cmx_model_handle handle);

/**
 * @brief Execute model inference
 * @param handle Model handle
 * @param inputs Array of input tensor data pointers
 * @param outputs Array of output tensor data pointers
 * @return Status code indicating execution result
 */
cmx_status cmx_execute_model(cmx_model_handle handle, void** inputs, void** outputs);

/**
 * @brief Set input tensor data
 * @param handle Model handle
 * @param index Input tensor index
 * @param data Pointer to input data
 * @param size Size of input data in bytes
 * @return Status code indicating success or failure
 */
cmx_status cmx_set_input(cmx_model_handle handle, uint32_t index, const void* data, size_t size);

/**
 * @brief Get output tensor data
 * @param handle Model handle
 * @param index Output tensor index
 * @param data Pointer to buffer for output data
 * @param size Size of output buffer in bytes
 * @return Status code indicating success or failure
 */
cmx_status cmx_get_output(cmx_model_handle handle, uint32_t index, void* data, size_t size);

} // namespace cmx