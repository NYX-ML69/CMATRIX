#pragma once

/**
 * @file cmx_padding.hpp
 * @brief Tensor padding utilities for embedded ML inference
 * 
 * Provides functions for applying various padding schemes to tensors
 * with support for in-place operations and no dynamic memory allocation.
 */

namespace cmx {
namespace utils {

/**
 * @brief Padding types supported by the system
 */
enum class PaddingType {
    ZERO,      ///< Zero padding (default)
    CONSTANT,  ///< Constant value padding
    REFLECT,   ///< Reflection padding
    REPLICATE  ///< Edge replication padding
};

/**
 * @brief Apply 2D padding to a tensor
 * 
 * @param input Input tensor data (NHWC format)
 * @param output Output tensor with padding applied
 * @param channels Number of channels
 * @param input_height Input height
 * @param input_width Input width
 * @param pad_top Top padding
 * @param pad_bottom Bottom padding
 * @param pad_left Left padding
 * @param pad_right Right padding
 * @param padding_type Type of padding to apply
 * @param constant_value Value to use for constant padding (default: 0.0)
 * 
 * @note Output buffer must be pre-allocated with size:
 *       (input_height + pad_top + pad_bottom) * (input_width + pad_left + pad_right) * channels
 */
void pad_2d(const float* input,
            float* output,
            int channels,
            int input_height,
            int input_width,
            int pad_top,
            int pad_bottom,
            int pad_left,
            int pad_right,
            PaddingType padding_type = PaddingType::ZERO,
            float constant_value = 0.0f);

/**
 * @brief Apply 1D padding to a tensor
 * 
 * @param input Input tensor data
 * @param output Output tensor with padding applied
 * @param channels Number of channels
 * @param input_length Input length
 * @param pad_left Left padding
 * @param pad_right Right padding
 * @param padding_type Type of padding to apply
 * @param constant_value Value to use for constant padding (default: 0.0)
 * 
 * @note Output buffer must be pre-allocated with size:
 *       (input_length + pad_left + pad_right) * channels
 */
void pad_1d(const float* input,
            float* output,
            int channels,
            int input_length,
            int pad_left,
            int pad_right,
            PaddingType padding_type = PaddingType::ZERO,
            float constant_value = 0.0f);

/**
 * @brief Apply symmetric padding to 2D tensor
 * 
 * Convenience function for symmetric padding (same padding on all sides)
 * 
 * @param input Input tensor data
 * @param output Output tensor with padding applied
 * @param channels Number of channels
 * @param input_height Input height
 * @param input_width Input width
 * @param pad_size Padding size for all sides
 * @param padding_type Type of padding to apply
 * @param constant_value Value to use for constant padding (default: 0.0)
 */
void pad_2d_symmetric(const float* input,
                      float* output,
                      int channels,
                      int input_height,
                      int input_width,
                      int pad_size,
                      PaddingType padding_type = PaddingType::ZERO,
                      float constant_value = 0.0f);

/**
 * @brief Calculate output dimensions after padding
 * 
 * @param input_height Input height
 * @param input_width Input width
 * @param pad_top Top padding
 * @param pad_bottom Bottom padding
 * @param pad_left Left padding
 * @param pad_right Right padding
 * @param output_height Output height (returned)
 * @param output_width Output width (returned)
 */
void calculate_padded_dims(int input_height,
                          int input_width,
                          int pad_top,
                          int pad_bottom,
                          int pad_left,
                          int pad_right,
                          int& output_height,
                          int& output_width);

/**
 * @brief Remove padding from a tensor (crop operation)
 * 
 * @param input Input tensor data
 * @param output Output tensor with padding removed
 * @param channels Number of channels
 * @param input_height Input height
 * @param input_width Input width
 * @param crop_top Top crop amount
 * @param crop_bottom Bottom crop amount
 * @param crop_left Left crop amount
 * @param crop_right Right crop amount
 */
void crop_2d(const float* input,
             float* output,
             int channels,
             int input_height,
             int input_width,
             int crop_top,
             int crop_bottom,
             int crop_left,
             int crop_right);

} // namespace utils
} // namespace cmx