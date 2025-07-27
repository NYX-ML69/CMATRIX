#pragma once

#include "cmx_math_utils.hpp"
#include "cmx_tensor_utils.hpp"
#include "cmx_quantization.hpp"
#include "cmx_padding.hpp"
#include "cmx_im2col.hpp"
#include "cmx_error.hpp"
#include "cmx_types.hpp"
#include <cstdint>

namespace cmx {

// Forward declarations
struct cmx_tensor;
struct cmx_quantization_params;

// Mathematical Utilities
/**
 * @brief Compute next power of 2
 * @param value Input value
 * @return Next power of 2 >= value
 */
uint32_t cmx_next_power_of_2(uint32_t value);

/**
 * @brief Check if value is power of 2
 * @param value Input value
 * @return true if value is power of 2, false otherwise
 */
bool cmx_is_power_of_2(uint32_t value);

/**
 * @brief Compute greatest common divisor
 * @param a First value
 * @param b Second value
 * @return Greatest common divisor of a and b
 */
uint32_t cmx_gcd(uint32_t a, uint32_t b);

/**
 * @brief Compute least common multiple
 * @param a First value
 * @param b Second value
 * @return Least common multiple of a and b
 */
uint32_t cmx_lcm(uint32_t a, uint32_t b);

/**
 * @brief Align value to specified alignment
 * @param value Value to align
 * @param alignment Alignment requirement (must be power of 2)
 * @return Aligned value
 */
size_t cmx_align(size_t value, size_t alignment);

/**
 * @brief Compute optimal tile size for given dimensions
 * @param width Width dimension
 * @param height Height dimension
 * @param max_tile_size Maximum tile size
 * @param tile_width Pointer to store optimal tile width
 * @param tile_height Pointer to store optimal tile height
 * @return Status code indicating success or failure
 */
cmx_status cmx_compute_optimal_tile_size(uint32_t width, uint32_t height, 
                                         uint32_t max_tile_size,
                                         uint32_t* tile_width, uint32_t* tile_height);

// Tensor Utilities
/**
 * @brief Calculate tensor size in bytes
 * @param tensor Tensor to calculate size for
 * @return Size in bytes
 */
size_t cmx_tensor_size_bytes(const cmx_tensor& tensor);

/**
 * @brief Calculate number of elements in tensor
 * @param tensor Tensor to calculate element count for
 * @return Number of elements
 */
uint64_t cmx_tensor_element_count(const cmx_tensor& tensor);

/**
 * @brief Check if tensor dimensions are valid
 * @param tensor Tensor to validate
 * @return true if dimensions are valid, false otherwise
 */
bool cmx_tensor_is_valid(const cmx_tensor& tensor);

/**
 * @brief Check if two tensors have compatible shapes for broadcasting
 * @param tensor_a First tensor
 * @param tensor_b Second tensor
 * @return true if tensors are broadcastable, false otherwise
 */
bool cmx_tensors_broadcastable(const cmx_tensor& tensor_a, const cmx_tensor& tensor_b);

/**
 * @brief Compute broadcast shape for two tensors
 * @param tensor_a First tensor
 * @param tensor_b Second tensor
 * @param result_shape Array to store result shape
 * @param max_dims Maximum number of dimensions in result
 * @param actual_dims Pointer to store actual number of dimensions
 * @return Status code indicating success or failure
 */
cmx_status cmx_compute_broadcast_shape(const cmx_tensor& tensor_a, const cmx_tensor& tensor_b,
                                       uint32_t* result_shape, uint32_t max_dims, 
                                       uint32_t* actual_dims);

/**
 * @brief Reshape tensor to new dimensions
 * @param tensor Tensor to reshape
 * @param new_shape Array of new dimensions
 * @param num_dims Number of new dimensions
 * @return Status code indicating success or failure
 */
cmx_status cmx_tensor_reshape(cmx_tensor& tensor, const uint32_t* new_shape, uint32_t num_dims);

/**
 * @brief Transpose tensor
 * @param input Input tensor
 * @param output Output tensor
 * @param axes Array of axis permutations
 * @return Status code indicating success or failure
 */
cmx_status cmx_tensor_transpose(const cmx_tensor& input, cmx_tensor& output, const uint32_t* axes);

/**
 * @brief Copy tensor data
 * @param dst Destination tensor
 * @param src Source tensor
 * @return Status code indicating success or failure
 */
cmx_status cmx_tensor_copy(cmx_tensor& dst, const cmx_tensor& src);

// Padding Utilities
/**
 * @brief Calculate padding sizes for convolution
 * @param input_size Input dimension size
 * @param kernel_size Kernel dimension size
 * @param stride Stride value
 * @param padding_type Type of padding (SAME, VALID)
 * @param pad_before Pointer to store padding before
 * @param pad_after Pointer to store padding after
 * @return Status code indicating success or failure
 */
cmx_status cmx_calculate_conv_padding(uint32_t input_size, uint32_t kernel_size, 
                                      uint32_t stride, cmx_padding_type padding_type,
                                      uint32_t* pad_before, uint32_t* pad_after);

/**
 * @brief Apply padding to tensor
 * @param input Input tensor
 * @param output Output tensor (must be pre-allocated)
 * @param padding Padding configuration
 * @return Status code indicating success or failure
 */
cmx_status cmx_apply_padding(const cmx_tensor& input, cmx_tensor& output, 
                             const cmx_padding_config& padding);

/**
 * @brief Remove padding from tensor
 * @param input Input tensor with padding
 * @param output Output tensor without padding
 * @param padding Padding configuration used
 * @return Status code indicating success or failure
 */
cmx_status cmx_remove_padding(const cmx_tensor& input, cmx_tensor& output,
                              const cmx_padding_config& padding);

// Im2Col Utilities
/**
 * @brief Convert image to column matrix for convolution
 * @param input Input tensor (NCHW format)
 * @param output Output column matrix
 * @param kernel_h Kernel height
 * @param kernel_w Kernel width
 * @param stride_h Stride height
 * @param stride_w Stride width
 * @param pad_h Padding height
 * @param pad_w Padding width
 * @param dilation_h Dilation height
 * @param dilation_w Dilation width
 * @return Status code indicating success or failure
 */
cmx_status cmx_im2col(const cmx_tensor& input, cmx_tensor& output,
                      uint32_t kernel_h, uint32_t kernel_w,
                      uint32_t stride_h, uint32_t stride_w,
                      uint32_t pad_h, uint32_t pad_w,
                      uint32_t dilation_h, uint32_t dilation_w);

/**
 * @brief Convert column matrix back to image format
 * @param input Input column matrix
 * @param output Output tensor (NCHW format)
 * @param channels Number of channels
 * @param height Output height
 * @param width Output width
 * @param kernel_h Kernel height
 * @param kernel_w Kernel width
 * @param stride_h Stride height
 * @param stride_w Stride width
 * @param pad_h Padding height
 * @param pad_w Padding width
 * @return Status code indicating success or failure
 */
cmx_status cmx_col2im(const cmx_tensor& input, cmx_tensor& output,
                      uint32_t channels, uint32_t height, uint32_t width,
                      uint32_t kernel_h, uint32_t kernel_w,
                      uint32_t stride_h, uint32_t stride_w,
                      uint32_t pad_h, uint32_t pad_w);

// Quantization Utilities
/**
 * @brief Quantize floating point tensor to 8-bit integers
 * @param input Input floating point tensor
 * @param output Output quantized tensor
 * @param params Quantization parameters
 * @return Status code indicating success or failure
 */
cmx_status cmx_quantize_tensor(const cmx_tensor& input, cmx_tensor& output,
                               const cmx_quantization_params& params);

/**
 * @brief Dequantize 8-bit integer tensor to floating point
 * @param input Input quantized tensor
 * @param output Output floating point tensor
 * @param params Quantization parameters
 * @return Status code indicating success or failure
 */
cmx_status cmx_dequantize_tensor(const cmx_tensor& input, cmx_tensor& output,
                                 const cmx_quantization_params& params);

/**
 * @brief Calculate quantization parameters from tensor data
 * @param tensor Input tensor to analyze
 * @param params Pointer to store calculated parameters
 * @return Status code indicating success or failure
 */
cmx_status cmx_calculate_quantization_params(const cmx_tensor& tensor,
                                             cmx_quantization_params* params);

/**
 * @brief Calibrate quantization parameters using calibration data
 * @param calibration_tensors Array of calibration tensors
 * @param num_tensors Number of calibration tensors
 * @param params Pointer to store calibrated parameters
 * @return Status code indicating success or failure
 */
cmx_status cmx_calibrate_quantization(const cmx_tensor* calibration_tensors, 
                                      uint32_t num_tensors,
                                      cmx_quantization_params* params);

// Data Format Conversion
/**
 * @brief Convert tensor from NCHW to NHWC format
 * @param input Input tensor in NCHW format
 * @param output Output tensor in NHWC format
 * @return Status code indicating success or failure
 */
cmx_status cmx_nchw_to_nhwc(const cmx_tensor& input, cmx_tensor& output);

/**
 * @brief Convert tensor from NHWC to NCHW format
 * @param input Input tensor in NHWC format
 * @param output Output tensor in NCHW format
 * @return Status code indicating success or failure
 */
cmx_status cmx_nhwc_to_nchw(const cmx_tensor& input, cmx_tensor& output);

/**
 * @brief Pack multiple channels into single tensor
 * @param inputs Array of single-channel tensors
 * @param num_inputs Number of input tensors
 * @param output Output multi-channel tensor
 * @return Status code indicating success or failure
 */
cmx_status cmx_pack_channels(const cmx_tensor* inputs, uint32_t num_inputs, cmx_tensor& output);

/**
 * @brief Unpack multi-channel tensor into separate channels
 * @param input Input multi-channel tensor
 * @param outputs Array to store single-channel tensors
 * @param num_outputs Number of output tensors
 * @return Status code indicating success or failure
 */
cmx_status cmx_unpack_channels(const cmx_tensor& input, cmx_tensor* outputs, uint32_t num_outputs);

// Memory Utilities
/**
 * @brief Allocate aligned memory
 * @param size Size in bytes to allocate
 * @param alignment Alignment requirement
 * @return Pointer to allocated memory, or nullptr on failure
 */
void* cmx_aligned_malloc(size_t size, size_t alignment);

/**
 * @brief Free aligned memory
 * @param ptr Pointer to memory to free
 */
void cmx_aligned_free(void* ptr);

/**
 * @brief Copy memory with optimal performance
 * @param dst Destination pointer
 * @param src Source pointer
 * @param size Number of bytes to copy
 * @return Status code indicating success or failure
 */
cmx_status cmx_fast_memcpy(void* dst, const void* src, size_t size);

/**
 * @brief Set memory to specified value with optimal performance
 * @param ptr Pointer to memory
 * @param value Value to set
 * @param size Number of bytes to set
 * @return Status code indicating success or failure
 */
cmx_status cmx_fast_memset(void* ptr, int value, size_t size);

// String and Hash Utilities
/**
 * @brief Compute hash of memory region
 * @param data Pointer to data
 * @param size Size of data in bytes
 * @return 64-bit hash value
 */
uint64_t cmx_compute_hash(const void* data, size_t size);

/**
 * @brief Compute hash of string
 * @param str String to hash
 * @return 64-bit hash value
 */
uint64_t cmx_string_hash(const char* str);

/**
 * @brief Safe string copy
 * @param dst Destination buffer
 * @param src Source string
 * @param dst_size Size of destination buffer
 * @return Status code indicating success or failure
 */
cmx_status cmx_safe_strcpy(char* dst, const char* src, size_t dst_size);

/**
 * @brief Format string with bounds checking
 * @param buffer Output buffer
 * @param buffer_size Size of output buffer
 * @param format Format string
 * @param ... Format arguments
 * @return Number of characters written, or -1 on error
 */
int cmx_safe_snprintf(char* buffer, size_t buffer_size, const char* format, ...);

} // namespace cmx