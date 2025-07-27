#include "cmx_im2col.hpp"

namespace cmx {
namespace utils {

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
               int dilation_h,
               int dilation_w) {
    
    int output_height = (input_height + 2 * pad_top - dilation_h * (kernel_height - 1) - 1) / stride_h + 1;
    int output_width = (input_width + 2 * pad_left - dilation_w * (kernel_width - 1) - 1) / stride_w + 1;
    
    int col_idx = 0;
    
    // Iterate through output spatial dimensions
    for (int out_y = 0; out_y < output_height; ++out_y) {
        for (int out_x = 0; out_x < output_width; ++out_x) {
            
            // Iterate through kernel dimensions
            for (int ky = 0; ky < kernel_height; ++ky) {
                for (int kx = 0; kx < kernel_width; ++kx) {
                    
                    // Calculate input coordinates
                    int in_y = out_y * stride_h - pad_top + ky * dilation_h;
                    int in_x = out_x * stride_w - pad_left + kx * dilation_w;
                    
                    // Iterate through channels
                    for (int c = 0; c < channels; ++c) {
                        if (in_y >= 0 && in_y < input_height && in_x >= 0 && in_x < input_width) {
                            // Valid input position - copy data
                            int input_idx = (in_y * input_width + in_x) * channels + c;
                            output[col_idx] = input[input_idx];
                        } else {
                            // Padding region - set to zero
                            output[col_idx] = 0.0f;
                        }
                        ++col_idx;
                    }
                }
            }
        }
    }
}

void im2col_1d(const float* input,
               float* output,
               int channels,
               int input_length,
               int kernel_size,
               int stride,
               int pad_left,
               int dilation) {
    
    int output_length = (input_length + 2 * pad_left - dilation * (kernel_size - 1) - 1) / stride + 1;
    
    int col_idx = 0;
    
    // Iterate through output positions
    for (int out_pos = 0; out_pos < output_length; ++out_pos) {
        
        // Iterate through kernel positions
        for (int k_pos = 0; k_pos < kernel_size; ++k_pos) {
            
            // Calculate input position
            int in_pos = out_pos * stride - pad_left + k_pos * dilation;
            
            // Iterate through channels
            for (int c = 0; c < channels; ++c) {
                if (in_pos >= 0 && in_pos < input_length) {
                    // Valid input position
                    int input_idx = in_pos * channels + c;
                    output[col_idx] = input[input_idx];
                } else {
                    // Padding region
                    output[col_idx] = 0.0f;
                }
                ++col_idx;
            }
        }
    }
}

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
                                  int& output_width) {
    
    int effective_kernel_h = dilation_h * (kernel_height - 1) + 1;
    int effective_kernel_w = dilation_w * (kernel_width - 1) + 1;
    
    output_height = (input_height + pad_top + pad_bottom - effective_kernel_h) / stride_h + 1;
    output_width = (input_width + pad_left + pad_right - effective_kernel_w) / stride_w + 1;
    
    // Ensure non-negative dimensions
    if (output_height < 0) output_height = 0;
    if (output_width < 0) output_width = 0;
}

int calculate_conv1d_output_length(int input_length,
                                   int kernel_size,
                                   int stride,
                                   int pad_left,
                                   int pad_right,
                                   int dilation) {
    
    int effective_kernel_size = dilation * (kernel_size - 1) + 1;
    int output_length = (input_length + pad_left + pad_right - effective_kernel_size) / stride + 1;
    
    return output_length > 0 ? output_length : 0;
}

} // namespace utils
} // namespace cmx