#include "cmx_conv3d.hpp"
#include <cstring>
#include <algorithm>
#include <cmath>

namespace cmx {
namespace kernels {

CmxConv3D::CmxConv3D() : weights_(nullptr), bias_(nullptr), is_configured_(false) {}

CmxConv3D::~CmxConv3D() {}

bool CmxConv3D::configure(const Config& config, const float* weights, const float* bias) {
    if (!weights) {
        return false;
    }

    config_ = config;
    weights_ = weights;
    bias_ = bias;
    
    // Validate configuration
    if (config_.kernel_size[0] <= 0 || config_.kernel_size[1] <= 0 || config_.kernel_size[2] <= 0) {
        return false;
    }
    
    if (config_.strides[0] <= 0 || config_.strides[1] <= 0 || config_.strides[2] <= 0) {
        return false;
    }
    
    if (config_.input_channels <= 0 || config_.output_channels <= 0) {
        return false;
    }

    calculate_padding();
    is_configured_ = true;
    return true;
}

bool CmxConv3D::run(const float* input, const TensorShape& input_shape,
                    float* output, const TensorShape& output_shape) {
    if (!is_configured_ || !input || !output) {
        return false;
    }

    // Choose implementation based on kernel size and performance characteristics
    if (config_.kernel_size[0] * config_.kernel_size[1] * config_.kernel_size[2] <= 27) {
        conv3d_direct(input, input_shape, output, output_shape);
    } else {
        conv3d_im2col(input, input_shape, output, output_shape);
    }

    return true;
}

bool CmxConv3D::infer_shape(const TensorShape& input_shape, TensorShape& output_shape) {
    if (!is_configured_) {
        return false;
    }

    calculate_output_size(input_shape, output_shape);
    return true;
}

bool CmxConv3D::get_memory_requirements(const TensorShape& input_shape,
                                        size_t& weight_memory, size_t& workspace_memory) {
    if (!is_configured_) {
        return false;
    }

    // Weight memory
    weight_memory = config_.output_channels * config_.input_channels * 
                   config_.kernel_size[0] * config_.kernel_size[1] * config_.kernel_size[2] * sizeof(float);
    
    if (config_.use_bias) {
        weight_memory += config_.output_channels * sizeof(float);
    }

    // Workspace memory for im2col buffer
    workspace_memory = config_.input_channels * config_.kernel_size[0] * 
                      config_.kernel_size[1] * config_.kernel_size[2] * sizeof(float);

    return true;
}

void CmxConv3D::calculate_output_size(const TensorShape& input_shape, TensorShape& output_shape) {
    output_shape.batch = input_shape.batch;
    output_shape.channels = config_.output_channels;

    // Calculate output dimensions
    output_shape.depth = (input_shape.depth + config_.padding[0] + config_.padding[1] - 
                         config_.dilation[0] * (config_.kernel_size[0] - 1) - 1) / config_.strides[0] + 1;
    
    output_shape.height = (input_shape.height + config_.padding[2] + config_.padding[3] - 
                          config_.dilation[1] * (config_.kernel_size[1] - 1) - 1) / config_.strides[1] + 1;
    
    output_shape.width = (input_shape.width + config_.padding[4] + config_.padding[5] - 
                         config_.dilation[2] * (config_.kernel_size[2] - 1) - 1) / config_.strides[2] + 1;
}

void CmxConv3D::calculate_padding() {
    if (config_.padding_mode == PaddingMode::SAME) {
        // Calculate padding for SAME mode
        for (int i = 0; i < 3; ++i) {
            int32_t total_pad = (config_.kernel_size[i] - 1) * config_.dilation[i];
            config_.padding[i * 2] = total_pad / 2;
            config_.padding[i * 2 + 1] = total_pad - config_.padding[i * 2];
        }
    } else if (config_.padding_mode == PaddingMode::VALID) {
        // No padding for VALID mode
        std::fill(config_.padding.begin(), config_.padding.end(), 0);
    }
    // For EXPLICIT mode, padding is already set by user
}

void CmxConv3D::apply_activation(float* data, size_t size) {
    if (!config_.fused_activation) {
        return;
    }

    switch (config_.activation_type) {
        case 1: // ReLU
            for (size_t i = 0; i < size; ++i) {
                data[i] = std::max(0.0f, data[i]);
            }
            break;
        case 2: // ReLU6
            for (size_t i = 0; i < size; ++i) {
                data[i] = std::min(6.0f, std::max(0.0f, data[i]));
            }
            break;
        case 3: // Tanh
            for (size_t i = 0; i < size; ++i) {
                data[i] = std::tanh(data[i]);
            }
            break;
        default:
            break;
    }
}

void CmxConv3D::conv3d_direct(const float* input, const TensorShape& input_shape,
                              float* output, const TensorShape& output_shape) {
    const int32_t batch_size = input_shape.batch;
    const int32_t in_depth = input_shape.depth;
    const int32_t in_height = input_shape.height;
    const int32_t in_width = input_shape.width;
    const int32_t in_channels = input_shape.channels;
    
    const int32_t out_depth = output_shape.depth;
    const int32_t out_height = output_shape.height;
    const int32_t out_width = output_shape.width;
    const int32_t out_channels = output_shape.channels;

    // Direct convolution implementation
    for (int32_t b = 0; b < batch_size; ++b) {
        for (int32_t oc = 0; oc < out_channels; ++oc) {
            for (int32_t od = 0; od < out_depth; ++od) {
                for (int32_t oh = 0; oh < out_height; ++oh) {
                    for (int32_t ow = 0; ow < out_width; ++ow) {
                        float sum = 0.0f;
                        
                        // Apply bias if present
                        if (config_.use_bias && bias_) {
                            sum = bias_[oc];
                        }
                        
                        // Convolution operation
                        for (int32_t ic = 0; ic < in_channels; ++ic) {
                            for (int32_t kd = 0; kd < config_.kernel_size[0]; ++kd) {
                                for (int32_t kh = 0; kh < config_.kernel_size[1]; ++kh) {
                                    for (int32_t kw = 0; kw < config_.kernel_size[2]; ++kw) {
                                        int32_t id = od * config_.strides[0] + kd * config_.dilation[0] - config_.padding[0];
                                        int32_t ih = oh * config_.strides[1] + kh * config_.dilation[1] - config_.padding[2];
                                        int32_t iw = ow * config_.strides[2] + kw * config_.dilation[2] - config_.padding[4];
                                        
                                        if (id >= 0 && id < in_depth && ih >= 0 && ih < in_height && 
                                            iw >= 0 && iw < in_width) {
                                            
                                            int32_t input_idx = ((b * in_depth + id) * in_height + ih) * in_width * in_channels + 
                                                               iw * in_channels + ic;
                                            int32_t weight_idx = ((oc * in_channels + ic) * config_.kernel_size[0] + kd) * 
                                                               config_.kernel_size[1] * config_.kernel_size[2] + 
                                                               kh * config_.kernel_size[2] + kw;
                                            
                                            sum += input[input_idx] * weights_[weight_idx];
                                        }
                                    }
                                }
                            }
                        }
                        
                        int32_t output_idx = ((b * out_depth + od) * out_height + oh) * out_width * out_channels + 
                                           ow * out_channels + oc;
                        output[output_idx] = sum;
                    }
                }
            }
        }
    }
    
    // Apply activation if configured
    if (config_.fused_activation) {
        apply_activation(output, batch_size * out_depth * out_height * out_width * out_channels);
    }
}

void CmxConv3D::conv3d_im2col(const float* input, const TensorShape& input_shape,
                              float* output, const TensorShape& output_shape) {
    // Placeholder for im2col implementation
    // This would involve creating a column matrix from input patches
    // and performing matrix multiplication with weight matrix
    // For now, fall back to direct implementation
    conv3d_direct(input, input_shape, output, output_shape);
}

} // namespace kernels
} // namespace cmx