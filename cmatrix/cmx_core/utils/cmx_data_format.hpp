#pragma once

#include <cstdint>

namespace cmx {
namespace utils {

/**
 * @brief Data layout formats for tensor operations
 */
enum class DataFormat {
    NHWC,   ///< Batch, Height, Width, Channel (TensorFlow default)
    NCHW,   ///< Batch, Channel, Height, Width (PyTorch default)
    CHW,    ///< Channel, Height, Width (single image)
    HWC,    ///< Height, Width, Channel (single image)
    NC,     ///< Batch, Channel (fully connected)
    C,      ///< Channel only (1D tensor)
    UNKNOWN ///< Unknown or unsupported format
};

/**
 * @brief Tensor dimension structure
 */
struct TensorDims {
    int32_t batch;    ///< Batch dimension (N)
    int32_t channel;  ///< Channel dimension (C)
    int32_t height;   ///< Height dimension (H)
    int32_t width;    ///< Width dimension (W)
    
    TensorDims() : batch(1), channel(1), height(1), width(1) {}
    TensorDims(int32_t b, int32_t c, int32_t h, int32_t w) 
        : batch(b), channel(c), height(h), width(w) {}
};

/**
 * @brief Calculate total number of elements in tensor
 * @param dims Tensor dimensions
 * @return Total number of elements
 */
inline int32_t get_tensor_size(const TensorDims& dims) {
    return dims.batch * dims.channel * dims.height * dims.width;
}

/**
 * @brief Calculate stride for each dimension based on data format
 * @param dims Tensor dimensions
 * @param format Data layout format
 * @param strides Output array for strides [batch_stride, channel_stride, height_stride, width_stride]
 */
void calculate_strides(
    const TensorDims& dims, 
    DataFormat format, 
    int32_t strides[4]
);

/**
 * @brief Get linear index for tensor element access
 * @param batch Batch index
 * @param channel Channel index
 * @param height Height index
 * @param width Width index
 * @param dims Tensor dimensions
 * @param format Data layout format
 * @return Linear index in the tensor
 */
int32_t get_tensor_index(
    int32_t batch, 
    int32_t channel, 
    int32_t height, 
    int32_t width,
    const TensorDims& dims,
    DataFormat format
);

/**
 * @brief Convert tensor data from one format to another
 * @param input Input tensor data
 * @param output Output tensor data (pre-allocated)
 * @param dims Tensor dimensions
 * @param input_format Input data format
 * @param output_format Output data format
 */
void convert_data_format(
    const float* input,
    float* output,
    const TensorDims& dims,
    DataFormat input_format,
    DataFormat output_format
);

/**
 * @brief Convert tensor data from one format to another (int8 version)
 * @param input Input tensor data
 * @param output Output tensor data (pre-allocated)
 * @param dims Tensor dimensions
 * @param input_format Input data format
 * @param output_format Output data format
 */
void convert_data_format_int8(
    const int8_t* input,
    int8_t* output,
    const TensorDims& dims,
    DataFormat input_format,
    DataFormat output_format
);

/**
 * @brief Get string representation of data format
 * @param format Data format enum
 * @return String representation
 */
const char* data_format_to_string(DataFormat format);

/**
 * @brief Parse data format from string
 * @param format_str String representation
 * @return Data format enum
 */
DataFormat string_to_data_format(const char* format_str);

/**
 * @brief Check if data format is valid for given tensor dimensions
 * @param dims Tensor dimensions
 * @param format Data format
 * @return true if valid, false otherwise
 */
bool is_valid_format(const TensorDims& dims, DataFormat format);

/**
 * @brief Get the number of spatial dimensions for a format
 * @param format Data format
 * @return Number of spatial dimensions (0, 1, or 2)
 */
int32_t get_spatial_dims(DataFormat format);

/**
 * @brief Calculate memory layout efficiency score for a format
 * @param dims Tensor dimensions
 * @param format Data format
 * @return Efficiency score (higher is better for cache locality)
 */
float calculate_memory_efficiency(const TensorDims& dims, DataFormat format);

/**
 * @brief Helper function to get channel dimension based on format
 * @param dims Tensor dimensions
 * @param format Data format
 * @return Channel dimension value
 */
inline int32_t get_channel_dim(const TensorDims& dims, DataFormat format) {
    return dims.channel;
}

/**
 * @brief Helper function to get spatial dimensions based on format
 * @param dims Tensor dimensions
 * @param format Data format
 * @param height Output height
 * @param width Output width
 */
inline void get_spatial_dims(const TensorDims& dims, DataFormat format, int32_t& height, int32_t& width) {
    height = dims.height;
    width = dims.width;
}

/**
 * @brief Check if two data formats are compatible for direct conversion
 * @param format1 First format
 * @param format2 Second format
 * @return true if compatible, false otherwise
 */
bool are_formats_compatible(DataFormat format1, DataFormat format2);

} // namespace utils
} // namespace cmx