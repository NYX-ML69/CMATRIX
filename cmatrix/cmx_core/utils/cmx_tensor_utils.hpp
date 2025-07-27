#pragma once

/**
 * @file cmx_tensor_utils.hpp
 * @brief General tensor manipulation utilities for embedded ML inference
 * 
 * Provides lightweight functions for tensor operations like reshape, transpose,
 * flatten, and broadcasting helpers. Optimized for embedded systems with
 * no dynamic memory allocation.
 */

namespace cmx {
namespace utils {

/**
 * @brief Flatten a multi-dimensional tensor to 1D
 * 
 * @param input Input tensor data
 * @param output Output flattened tensor
 * @param total_elements Total number of elements to flatten
 * 
 * @note This is essentially a memory copy operation
 */
void flatten(const float* input, float* output, int total_elements);

/**
 * @brief Reshape tensor data (logical reshape - no data movement)
 * 
 * @param input Input tensor data
 * @param output Output tensor data
 * @param total_elements Total number of elements
 * 
 * @note This function validates that reshape is possible and copies data
 */
void reshape(const float* input, float* output, int total_elements);

/**
 * @brief Transpose a 2D matrix
 * 
 * @param input Input matrix data (row-major)
 * @param output Output transposed matrix
 * @param rows Number of rows in input
 * @param cols Number of columns in input
 */
void transpose_2d(const float* input, float* output, int rows, int cols);

/**
 * @brief Transpose a 3D tensor (permute dimensions)
 * 
 * @param input Input tensor data
 * @param output Output transposed tensor
 * @param dim0 First dimension size
 * @param dim1 Second dimension size
 * @param dim2 Third dimension size
 * @param axis0 New position for dimension 0
 * @param axis1 New position for dimension 1
 * @param axis2 New position for dimension 2
 */
void transpose_3d(const float* input, float* output,
                  int dim0, int dim1, int dim2,
                  int axis0, int axis1, int axis2);

/**
 * @brief Transpose a 4D tensor (NHWC <-> NCHW conversion)
 * 
 * @param input Input tensor data
 * @param output Output transposed tensor
 * @param n Batch size
 * @param h Height
 * @param w Width
 * @param c Channels
 * @param nhwc_to_nchw If true, convert NHWC to NCHW; otherwise NCHW to NHWC
 */
void transpose_4d_nhwc_nchw(const float* input, float* output,
                           int n, int h, int w, int c,
                           bool nhwc_to_nchw);

/**
 * @brief Copy tensor data with optional stride
 * 
 * @param input Input tensor data
 * @param output Output tensor data
 * @param elements Number of elements to copy
 * @param input_stride Input stride (default: 1)
 * @param output_stride Output stride (default: 1)
 */
void copy_with_stride(const float* input, float* output, int elements,
                     int input_stride = 1, int output_stride = 1);

/**
 * @brief Slice tensor along one dimension
 * 
 * @param input Input tensor data
 * @param output Output sliced tensor
 * @param total_size Total size of input tensor
 * @param slice_dim_size Size of dimension being sliced
 * @param slice_start Start index for slice
 * @param slice_length Length of slice
 * @param elements_per_slice Number of elements per slice unit
 */
void slice_1d(const float* input, float* output,
              int total_size, int slice_dim_size,
              int slice_start, int slice_length,
              int elements_per_slice);

/**
 * @brief Concatenate tensors along specified axis
 * 
 * @param inputs Array of input tensor pointers
 * @param output Output concatenated tensor
 * @param num_inputs Number of input tensors
 * @param tensor_sizes Array of sizes for each input tensor
 * @param axis_size Size of the concatenation axis for each tensor
 * @param elements_per_axis Number of elements per axis unit
 */
void concatenate(const float** inputs, float* output,
                int num_inputs, const int* tensor_sizes,
                const int* axis_sizes, int elements_per_axis);

/**
 * @brief Broadcast tensor to larger dimensions
 * 
 * @param input Input tensor data
 * @param output Output broadcast tensor
 * @param input_size Size of input tensor
 * @param output_size Size of output tensor
 * @param broadcast_factor Broadcast multiplication factor
 */
void broadcast_1d(const float* input, float* output,
                  int input_size, int output_size, int broadcast_factor);

/**
 * @brief Calculate linear index from multi-dimensional coordinates
 * 
 * @param coords Array of coordinates
 * @param dims Array of dimension sizes
 * @param num_dims Number of dimensions
 * @return Linear index
 */
int coords_to_index(const int* coords, const int* dims, int num_dims);

/**
 * @brief Calculate multi-dimensional coordinates from linear index
 * 
 * @param index Linear index
 * @param dims Array of dimension sizes
 * @param num_dims Number of dimensions
 * @param coords Output array of coordinates
 */
void index_to_coords(int index, const int* dims, int num_dims, int* coords);

/**
 * @brief Calculate total size from dimension array
 * 
 * @param dims Array of dimension sizes
 * @param num_dims Number of dimensions
 * @return Total number of elements
 */
int calculate_total_size(const int* dims, int num_dims);

/**
 * @brief Fill tensor with constant value
 * 
 * @param output Output tensor data
 * @param size Number of elements
 * @param value Value to fill with
 */
void fill_constant(float* output, int size, float value);

/**
 * @brief Copy tensor data
 * 
 * @param input Input tensor data
 * @param output Output tensor data
 * @param size Number of elements to copy
 */
void copy_tensor(const float* input, float* output, int size);

} // namespace utils
} // namespace cmx