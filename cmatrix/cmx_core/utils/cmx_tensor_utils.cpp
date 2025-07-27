#include "cmx_tensor_utils.hpp"

namespace cmx {
namespace utils {

void flatten(const float* input, float* output, int total_elements) {
    for (int i = 0; i < total_elements; ++i) {
        output[i] = input[i];
    }
}

void reshape(const float* input, float* output, int total_elements) {
    // Reshape is just a copy operation since we're not changing memory layout
    for (int i = 0; i < total_elements; ++i) {
        output[i] = input[i];
    }
}

void transpose_2d(const float* input, float* output, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            output[j * rows + i] = input[i * cols + j];
        }
    }
}

void transpose_3d(const float* input, float* output,
                  int dim0, int dim1, int dim2,
                  int axis0, int axis1, int axis2) {
    
    int dims[3] = {dim0, dim1, dim2};
    int new_dims[3] = {0, 0, 0};
    new_dims[axis0] = dim0;
    new_dims[axis1] = dim1;
    new_dims[axis2] = dim2;
    
    for (int i = 0; i < dim0; ++i) {
        for (int j = 0; j < dim1; ++j) {
            for (int k = 0; k < dim2; ++k) {
                int old_coords[3] = {i, j, k};
                int new_coords[3] = {0, 0, 0};
                
                new_coords[axis0] = old_coords[0];
                new_coords[axis1] = old_coords[1];
                new_coords[axis2] = old_coords[2];
                
                int old_idx = i * dim1 * dim2 + j * dim2 + k;
                int new_idx = new_coords[0] * new_dims[1] * new_dims[2] + 
                             new_coords[1] * new_dims[2] + new_coords[2];
                
                output[new_idx] = input[old_idx];
            }
        }
    }
}

void transpose_4d_nhwc_nchw(const float* input, float* output,
                           int n, int h, int w, int c,
                           bool nhwc_to_nchw) {
    
    if (nhwc_to_nchw) {
        // NHWC -> NCHW
        for (int batch = 0; batch < n; ++batch) {
            for (int channel = 0; channel < c; ++channel) {
                for (int height = 0; height < h; ++height) {
                    for (int width = 0; width < w; ++width) {
                        int nhwc_idx = batch * h * w * c + height * w * c + width * c + channel;
                        int nchw_idx = batch * c * h * w + channel * h * w + height * w + width;
                        output[nchw_idx] = input[nhwc_idx];
                    }
                }
            }
        }
    } else {
        // NCHW -> NHWC
        for (int batch = 0; batch < n; ++batch) {
            for (int channel = 0; channel < c; ++channel) {
                for (int height = 0; height < h; ++height) {
                    for (int width = 0; width < w; ++width) {
                        int nchw_idx = batch * c * h * w + channel * h * w + height * w + width;
                        int nhwc_idx = batch * h * w * c + height * w * c + width * c + channel;
                        output[nhwc_idx] = input[nchw_idx];
                    }
                }
            }
        }
    }
}

void copy_with_stride(const float* input, float* output, int elements,
                     int input_stride, int output_stride) {
    
    int in_idx = 0;
    int out_idx = 0;
    
    for (int i = 0; i < elements; ++i) {
        output[out_idx] = input[in_idx];
        in_idx += input_stride;
        out_idx += output_stride;
    }
}

void slice_1d(const float* input, float* output,
              int total_size, int slice_dim_size,
              int slice_start, int slice_length,
              int elements_per_slice) {
    
    int slices_before = total_size / (slice_dim_size * elements_per_slice);
    int output_idx = 0;
    
    for (int slice_group = 0; slice_group < slices_before; ++slice_group) {
        int start_idx = slice_group * slice_dim_size * elements_per_slice + 
                       slice_start * elements_per_slice;
        
        for (int i = 0; i < slice_length * elements_per_slice; ++i) {
            output[output_idx++] = input[start_idx + i];
        }
    }
}

void concatenate(const float** inputs, float* output,
                int num_inputs, const int* tensor_sizes,
                const int* axis_sizes, int elements_per_axis) {
    
    int output_idx = 0;
    
    for (int i = 0; i < num_inputs; ++i) {
        int current_size = axis_sizes[i] * elements_per_axis;
        for (int j = 0; j < current_size; ++j) {
            output[output_idx++] = inputs[i][j];
        }
    }
}

void broadcast_1d(const float* input, float* output,
                  int input_size, int output_size, int broadcast_factor) {
    
    for (int i = 0; i < output_size; ++i) {
        output[i] = input[i % input_size];
    }
}

int coords_to_index(const int* coords, const int* dims, int num_dims) {
    int index = 0;
    int stride = 1;
    
    for (int i = num_dims - 1; i >= 0; --i) {
        index += coords[i] * stride;
        stride *= dims[i];
    }
    
    return index;
}

void index_to_coords(int index, const int* dims, int num_dims, int* coords) {
    for (int i = num_dims - 1; i >= 0; --i) {
        coords[i] = index % dims[i];
        index /= dims[i];
    }
}

int calculate_total_size(const int* dims, int num_dims) {
    int total = 1;
    for (int i = 0; i < num_dims; ++i) {
        total *= dims[i];
    }
    return total;
}

void fill_constant(float* output, int size, float value) {
    for (int i = 0; i < size; ++i) {
        output[i] = value;
    }
}

void copy_tensor(const float* input, float* output, int size) {
    for (int i = 0; i < size; ++i) {
        output[i] = input[i];
    }
}

} // namespace utils
} // namespace cmx