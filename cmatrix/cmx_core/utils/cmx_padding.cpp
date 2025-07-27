#include "cmx_padding.hpp"

namespace cmx {
namespace utils {

namespace {
    // Helper function to get padded value based on padding type
    float get_padded_value(const float* input,
                          int channels,
                          int input_height,
                          int input_width,
                          int y, int x, int c,
                          PaddingType padding_type,
                          float constant_value) {
        
        switch (padding_type) {
            case PaddingType::ZERO:
                return 0.0f;
                
            case PaddingType::CONSTANT:
                return constant_value;
                
            case PaddingType::REFLECT: {
                // Clamp coordinates to valid range using reflection
                int clamped_y = y;
                int clamped_x = x;
                
                if (y < 0) clamped_y = -y;
                else if (y >= input_height) clamped_y = 2 * input_height - y - 2;
                
                if (x < 0) clamped_x = -x;
                else if (x >= input_width) clamped_x = 2 * input_width - x - 2;
                
                // Clamp to bounds in case of extreme values
                clamped_y = (clamped_y < 0) ? 0 : (clamped_y >= input_height) ? input_height - 1 : clamped_y;
                clamped_x = (clamped_x < 0) ? 0 : (clamped_x >= input_width) ? input_width - 1 : clamped_x;
                
                int idx = (clamped_y * input_width + clamped_x) * channels + c;
                return input[idx];
            }
            
            case PaddingType::REPLICATE: {
                // Clamp coordinates to valid range
                int clamped_y = (y < 0) ? 0 : (y >= input_height) ? input_height - 1 : y;
                int clamped_x = (x < 0) ? 0 : (x >= input_width) ? input_width - 1 : x;
                
                int idx = (clamped_y * input_width + clamped_x) * channels + c;
                return input[idx];
            }
            
            default:
                return 0.0f;
        }
    }
}

void pad_2d(const float* input,
            float* output,
            int channels,
            int input_height,
            int input_width,
            int pad_top,
            int pad_bottom,
            int pad_left,
            int pad_right,
            PaddingType padding_type,
            float constant_value) {
    
    int output_height = input_height + pad_top + pad_bottom;
    int output_width = input_width + pad_left + pad_right;
    
    for (int out_y = 0; out_y < output_height; ++out_y) {
        for (int out_x = 0; out_x < output_width; ++out_x) {
            
            // Calculate input coordinates
            int in_y = out_y - pad_top;
            int in_x = out_x - pad_left;
            
            for (int c = 0; c < channels; ++c) {
                int output_idx = (out_y * output_width + out_x) * channels + c;
                
                if (in_y >= 0 && in_y < input_height && in_x >= 0 && in_x < input_width) {
                    // Inside input region - copy directly
                    int input_idx = (in_y * input_width + in_x) * channels + c;
                    output[output_idx] = input[input_idx];
                } else {
                    // Outside input region - apply padding
                    output[output_idx] = get_padded_value(input, channels, input_height, input_width,
                                                         in_y, in_x, c, padding_type, constant_value);
                }
            }
        }
    }
}

void pad_1d(const float* input,
            float* output,
            int channels,
            int input_length,
            int pad_left,
            int pad_right,
            PaddingType padding_type,
            float constant_value) {
    
    int output_length = input_length + pad_left + pad_right;
    
    for (int out_pos = 0; out_pos < output_length; ++out_pos) {
        
        int in_pos = out_pos - pad_left;
        
        for (int c = 0; c < channels; ++c) {
            int output_idx = out_pos * channels + c;
            
            if (in_pos >= 0 && in_pos < input_length) {
                // Inside input region
                int input_idx = in_pos * channels + c;
                output[output_idx] = input[input_idx];
            } else {
                // Outside input region - apply padding
                switch (padding_type) {
                    case PaddingType::ZERO:
                        output[output_idx] = 0.0f;
                        break;
                    case PaddingType::CONSTANT:
                        output[output_idx] = constant_value;
                        break;
                    case PaddingType::REFLECT: {
                        int clamped_pos = in_pos;
                        if (in_pos < 0) clamped_pos = -in_pos;
                        else if (in_pos >= input_length) clamped_pos = 2 * input_length - in_pos - 2;
                        
                        clamped_pos = (clamped_pos < 0) ? 0 : (clamped_pos >= input_length) ? input_length - 1 : clamped_pos;
                        int idx = clamped_pos * channels + c;
                        output[output_idx] = input[idx];
                        break;
                    }
                    case PaddingType::REPLICATE: {
                        int clamped_pos = (in_pos < 0) ? 0 : (in_pos >= input_length) ? input_length - 1 : in_pos;
                        int idx = clamped_pos * channels + c;
                        output[output_idx] = input[idx];
                        break;
                    }
                    default:
                        output[output_idx] = 0.0f;
                        break;
                }
            }
        }
    }
}

void pad_2d_symmetric(const float* input,
                      float* output,
                      int channels,
                      int input_height,
                      int input_width,
                      int pad_size,
                      PaddingType padding_type,
                      float constant_value) {
    
    pad_2d(input, output, channels, input_height, input_width,
           pad_size, pad_size, pad_size, pad_size, padding_type, constant_value);
}

void calculate_padded_dims(int input_height,
                          int input_width,
                          int pad_top,
                          int pad_bottom,
                          int pad_left,
                          int pad_right,
                          int& output_height,
                          int& output_width) {
    
    output_height = input_height + pad_top + pad_bottom;
    output_width = input_width + pad_left + pad_right;
}

void crop_2d(const float* input,
             float* output,
             int channels,
             int input_height,
             int input_width,
             int crop_top,
             int crop_bottom,
             int crop_left,
             int crop_right) {
    
    int output_height = input_height - crop_top - crop_bottom;
    int output_width = input_width - crop_left - crop_right;
    
    for (int out_y = 0; out_y < output_height; ++out_y) {
        for (int out_x = 0; out_x < output_width; ++out_x) {
            
            int in_y = out_y + crop_top;
            int in_x = out_x + crop_left;
            
            for (int c = 0; c < channels; ++c) {
                int input_idx = (in_y * input_width + in_x) * channels + c;
                int output_idx = (out_y * output_width + out_x) * channels + c;
                output[output_idx] = input[input_idx];
            }
        }
    }
}

} // namespace utils
} // namespace cmx