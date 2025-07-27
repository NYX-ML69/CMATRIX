#include "cmx_data_format.hpp"
#include <cstring>

namespace cmx {
namespace utils {

void calculate_strides(
    const TensorDims& dims, 
    DataFormat format, 
    int32_t strides[4]
) {
    // Initialize strides array
    for (int i = 0; i < 4; ++i) {
        strides[i] = 0;
    }
    
    switch (format) {
        case DataFormat::NHWC:
            strides[0] = dims.height * dims.width * dims.channel;  // batch_stride
            strides[1] = 1;                                        // channel_stride
            strides[2] = dims.width * dims.channel;               // height_stride
            strides[3] = dims.channel;                            // width_stride
            break;
            
        case DataFormat::NCHW:
            strides[0] = dims.channel * dims.height * dims.width; // batch_stride
            strides[1] = dims.height * dims.width;                // channel_stride
            strides[2] = dims.width;                              // height_stride
            strides[3] = 1;                                       // width_stride
            break;
            
        case DataFormat::CHW:
            strides[0] = 0;                                       // batch_stride (no batch)
            strides[1] = dims.height * dims.width;               // channel_stride
            strides[2] = dims.width;                             // height_stride
            strides[3] = 1;                                      // width_stride
            break;
            
        case DataFormat::HWC:
            strides[0] = 0;                                      // batch_stride (no batch)
            strides[1] = 1;                                      // channel_stride
            strides[2] = dims.width * dims.channel;             // height_stride
            strides[3] = dims.channel;                          // width_stride
            break;
            
        case DataFormat::NC:
            strides[0] = dims.channel;                          // batch_stride
            strides[1] = 1;                                     // channel_stride
            strides[2] = 0;                                     // height_stride (no height)
            strides[3] = 0;                                     // width_stride (no width)
            break;
            
        case DataFormat::C:
            strides[0] = 0;                                     // batch_stride (no batch)
            strides[1] = 1;                                     // channel_stride
            strides[2] = 0;                                     // height_stride (no height)
            strides[3] = 0;                                     // width_stride (no width)
            break;
            
        default:
            // UNKNOWN format - all strides remain 0
            break;
    }
}

int32_t get_tensor_index(
    int32_t batch, 
    int32_t channel, 
    int32_t height, 
    int32_t width,
    const TensorDims& dims,
    DataFormat format
) {
    int32_t strides[4];
    calculate_strides(dims, format, strides);
    
    return batch * strides[0] + channel * strides[1] + height * strides[2] + width * strides[3];
}

void convert_data_format(
    const float* input,
    float* output,
    const TensorDims& dims,
    DataFormat input_format,
    DataFormat output_format
) {
    if (input_format == output_format) {
        // Direct copy if formats are the same
        int32_t size = get_tensor_size(dims);
        for (int32_t i = 0; i < size; ++i) {
            output[i] = input[i];
        }
        return;
    }
    
    // Convert element by element
    for (int32_t b = 0; b < dims.batch; ++b) {
        for (int32_t c = 0; c < dims.channel; ++c) {
            for (int32_t h = 0; h < dims.height; ++h) {
                for (int32_t w = 0; w < dims.width; ++w) {
                    int32_t input_idx = get_tensor_index(b, c, h, w, dims, input_format);
                    int32_t output_idx = get_tensor_index(b, c, h, w, dims, output_format);
                    output[output_idx] = input[input_idx];
                }
            }
        }
    }
}

void convert_data_format_int8(
    const int8_t* input,
    int8_t* output,
    const TensorDims& dims,
    DataFormat input_format,
    DataFormat output_format
) {
    if (input_format == output_format) {
        // Direct copy if formats are the same
        int32_t size = get_tensor_size(dims);
        for (int32_t i = 0; i < size; ++i) {
            output[i] = input[i];
        }
        return;
    }
    
    // Convert element by element
    for (int32_t b = 0; b < dims.batch; ++b) {
        for (int32_t c = 0; c < dims.channel; ++c) {
            for (int32_t h = 0; h < dims.height; ++h) {
                for (int32_t w = 0; w < dims.width; ++w) {
                    int32_t input_idx = get_tensor_index(b, c, h, w, dims, input_format);
                    int32_t output_idx = get_tensor_index(b, c, h, w, dims, output_format);
                    output[output_idx] = input[input_idx];
                }
            }
        }
    }
}

const char* data_format_to_string(DataFormat format) {
    switch (format) {
        case DataFormat::NHWC:   return "NHWC";
        case DataFormat::NCHW:   return "NCHW";
        case DataFormat::CHW:    return "CHW";
        case DataFormat::HWC:    return "HWC";
        case DataFormat::NC:     return "NC";
        case DataFormat::C:      return "C";
        default:                 return "UNKNOWN";
    }
}

DataFormat string_to_data_format(const char* format_str) {
    if (strcmp(format_str, "NHWC") == 0) return DataFormat::NHWC;
    if (strcmp(format_str, "NCHW") == 0) return DataFormat::NCHW;
    if (strcmp(format_str, "CHW") == 0)  return DataFormat::CHW;
    if (strcmp(format_str, "HWC") == 0)  return DataFormat::HWC;
    if (strcmp(format_str, "NC") == 0)   return DataFormat::NC;
    if (strcmp(format_str, "C") == 0)    return DataFormat::C;
    return DataFormat::UNKNOWN;
}

bool is_valid_format(const TensorDims& dims, DataFormat format) {
    switch (format) {
        case DataFormat::NHWC:
        case DataFormat::NCHW:
            return dims.batch > 0 && dims.channel > 0 && dims.height > 0 && dims.width > 0;
            
        case DataFormat::CHW:
            return dims.channel > 0 && dims.height > 0 && dims.width > 0;
            
        case DataFormat::HWC:
            return dims.height > 0 && dims.width > 0 && dims.channel > 0;
            
        case DataFormat::NC:
            return dims.batch > 0 && dims.channel > 0;
            
        case DataFormat::C:
            return dims.channel > 0;
            
        default:
            return false;
    }
}

int32_t get_spatial_dims(DataFormat format) {
    switch (format) {
        case DataFormat::NHWC:
        case DataFormat::NCHW:
        case DataFormat::CHW:
        case DataFormat::HWC:
            return 2;  // Height and Width
            
        case DataFormat::NC:
        case DataFormat::C:
            return 0;  // No spatial dimensions
            
        default:
            return -1; // Unknown
    }
}

float calculate_memory_efficiency(const TensorDims& dims, DataFormat format) {
    // Simple heuristic: formats with better cache locality get higher scores
    // Channel-last formats (NHWC, HWC) are generally better for convolutions
    // Channel-first formats (NCHW, CHW) are better for element-wise operations
    
    int32_t total_elements = get_tensor_size(dims);
    if (total_elements == 0) return 0.0f;
    
    switch (format) {
        case DataFormat::NHWC:
        case DataFormat::HWC:
            // Good for convolutions - channels are contiguous
            return 1.0f;
            
        case DataFormat::NCHW:
        case DataFormat::CHW:
            // Good for element-wise operations - spatial locality
            return 0.8f;
            
        case DataFormat::NC:
        case DataFormat::C:
            // Linear formats - optimal for their use case
            return 1.0f;
            
        default:
            return 0.0f;
    }
}

bool are_formats_compatible(DataFormat format1, DataFormat format2) {
    // Check if formats have the same dimensionality
    int32_t dims1 = get_spatial_dims(format1);
    int32_t dims2 = get_spatial_dims(format2);
    
    if (dims1 != dims2) return false;
    
    // Both formats must be known
    if (format1 == DataFormat::UNKNOWN || format2 == DataFormat::UNKNOWN) {
        return false;
    }
    
    // Check batch dimension compatibility
    bool has_batch1 = (format1 == DataFormat::NHWC || format1 == DataFormat::NCHW || format1 == DataFormat::NC);
    bool has_batch2 = (format2 == DataFormat::NHWC || format2 == DataFormat::NCHW || format2 == DataFormat::NC);
    
    return has_batch1 == has_batch2;
}

} // namespace utils
} // namespace cmx