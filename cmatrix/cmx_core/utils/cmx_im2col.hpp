#pragma once

/**
 * @file cmx_im2col.hpp
 * @brief Image-to-column transformation utilities for efficient convolution operations
 * 
 * Implements im2col transformation to convert 2D spatial tensor blocks into columnar form
 * for efficient GEMM-based convolution operations. Optimized for embedded systems with
 * no dynamic memory allocation.
 */

namespace cmx {
namespace utils {

/**
 * @brief Performs image-to-column transformation for 2D convolution
 * 
 * Converts input tensor patches into column format for efficient matrix multiplication.
 * Supports stride, padding, and dilation parameters.
 * 
 * @param input Input tensor data (NHWC format)
 * @param output Pre-allocated output buffer for column data
 * @param channels Number of input channels
 * @param input_height Height of input tensor
 * @param input_width Width of input tensor
 * @param kernel_height Height of convolution kernel
 * @param kernel_width Width of convolution kernel
 * @param stride_h Vertical stride
 * @param stride_w Horizontal stride
 * @param pad_top Top padding
 * @param pad_left Left padding
 * @param dilation_h Vertical dilation (default: 1)
 * @param dilation_w Horizontal dilation (default: 1)
 * 
 * @note Output buffer must be pre-allocated with size:
 *       (output_height * output_width) * (channels * kernel_height * kernel_width)
 */
void im2col_2d(const float* input, 
               float* output,
               int channels,
               int input_height,
               int input_width,
               int kernel_height,
               int kernel_width,
               int stride_h,
               int stride_w,
               int pad_top,
               int pad_left,
               int dilation_h = 1,
               int dilation_w = 1);

/**
 * @brief Performs image-to-column transformation for 1D convolution
 * 
 * @param input Input tensor data
 * @param output Pre-allocated output buffer
 * @param channels Number of input channels
 * @param input_length Length of input tensor
 * @param kernel_size Size of convolution kernel
 * @param stride Stride value
 * @param pad_left Left padding
 * @param dilation Dilation factor (default: 1)
 */
void im2col_1d(const float* input,
               float* output,
               int channels,
               int input_length,
               int kernel_size,
               int stride,
               int pad_left,
               int dilation = 1);

/**
 * @brief Calculate output dimensions for 2D convolution
 * 
 * @param input_height Input height
 * @param input_width Input width
 * @param kernel_height Kernel height
 * @param kernel_width Kernel width
 * @param stride_h Vertical stride
 * @param stride_w Horizontal stride
 * @param pad_top Top padding
 * @param pad_bottom Bottom padding
 * @param pad_left Left padding
 * @param pad_right Right padding
 * @param dilation_h Vertical dilation
 * @param dilation_w Horizontal dilation
 * @param output_height Output height (returned)
 * @param output_width Output width (returned)
 */
void calculate_conv2d_output_dims(int input_height,
                                  int input_width,
                                  int kernel_height,
                                  int kernel_width,
                                  int stride_h,
                                  int stride_w,
                                  int pad_top,
                                  int pad_bottom,
                                  int pad_left,
                                  int pad_right,
                                  int dilation_h,
                                  int dilation_w,
                                  int& output_height,
                                  int& output_width);

/**
 * @brief Calculate output length for 1D convolution
 * 
 * @param input_length Input length
 * @param kernel_size Kernel size
 * @param stride Stride
 * @param pad_left Left padding
 * @param pad_right Right padding
 * @param dilation Dilation
 * @return Output length
 */
int calculate_conv1d_output_length(int input_length,
                                   int kernel_size,
                                   int stride,
                                   int pad_left,
                                   int pad_right,
                                   int dilation);

} // namespace utils
} // namespace cmx