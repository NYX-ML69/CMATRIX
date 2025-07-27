#include "cmx_bias.hpp"
#include <cstring>
#include <algorithm>
#include <cmath>

namespace cmx {
namespace kernels {

CmxBias::CmxBias() : bias_data_(nullptr), is_configured_(false) {}

CmxBias::~CmxBias() {}

bool CmxBias::configure(const Config& config, const float* bias_data) {
    if (!bias_data) {
        return false;
    }

    config_ = config;
    bias_data_ = bias_data;
    
    // Validate configuration
    if (config_.channels <= 0) {
        return false;
    }

    is_configured_ = true;
    return true;
}

bool CmxBias::run(const float* input, const TensorShape& input_shape,
                  float* output, const TensorShape& output_shape) {
    if (!is_configured_ || !input || !output) {
        return false;
    }

    // Validate shapes
    if (input_shape.batch != output_shape.batch ||
        input_shape.height != output_shape.height ||
        input_shape.width != output_shape.width ||
        input_shape.channels != output_shape.channels) {
        return false;
    }

    // Apply bias based on mode and layout
    switch (config_.bias_mode) {
        case BiasMode::CHANNEL_WISE:
            if (config_.data_layout == DataLayout::NHWC) {
                apply_channel_wise_bias_nhwc(input, input_shape, output, output_shape);
            } else if (config_.data_layout == DataLayout::NCHW) {
                apply_channel_wise_bias_nchw(input, input_shape, output, output_shape);
            } else { // NC layout
                apply_channel_wise_bias_nhwc(input, input_shape, output, output_shape);
            }
            break;
        case BiasMode::ELEMENT_WISE:
            apply_element_wise_bias(input, input_shape, output, output_shape);
            break;
        default:
            return false;
    }

    return true;
}

bool CmxBias::infer_shape(const TensorShape& input_shape, TensorShape& output_shape) {
    if (!is_configured_) {
        return false;
    }

    // Output shape is same as input shape
    output_shape = input_shape;
    return true;
}

bool CmxBias::get_memory_requirements(const TensorShape& input_shape,
                                     size_t& weight_memory, size_t& workspace_memory) {
    if (!is_configured_) {
        return false;
    }

    // Weight memory for bias data
    if (config_.bias_mode == BiasMode::CHANNEL_WISE) {
        weight_memory = config_.channels * sizeof(float);
    } else {
        weight_memory = input_shape.batch * input_shape.height * 
                       input_shape.width * input_shape.channels * sizeof(float);
    }

    // No workspace memory needed
    workspace_memory = 0;

    return true;
}

void CmxBias::apply_activation(float* data, size_t size) {
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

void CmxBias::apply_channel_wise_bias_nhwc(const float* input, const TensorShape& input_shape,
                                          float* output, const TensorShape& output_shape) {
    const int32_t batch_size = input_shape.batch;
    const int32_t height = input_shape.height;
    const int32_t width = input_shape.width;
    const int32_t channels = input_shape.channels;
    
    const size_t spatial_size = height * width;
    const size_t total_size = batch_size * spatial_size * channels;

    // Apply bias channel-wise for NHWC layout
    for (int32_t b = 0; b < batch_size; ++b) {
        for (int32_t hw = 0; hw < spatial_size; ++hw) {
            for (int32_t c = 0; c < channels; ++c) {
                const size_t idx = ((b * spatial_size + hw) * channels) + c;
                output[idx] = input[idx] + bias_data_[c];
            }
        }
    }
    
    // Apply activation if configured
    if (config_.fused_activation) {
        apply_activation(output, total_size);
    }
}

void CmxBias::apply_channel_wise_bias_nchw(const float* input, const TensorShape& input_shape,
                                          float* output, const TensorShape& output_shape) {
    const int32_t batch_size = input_shape.batch;
    const int32_t height = input_shape.height;
    const int32_t width = input_shape.width;
    const int32_t channels = input_shape.channels;
    
    const size_t spatial_size = height * width;
    const size_t channel_size = spatial_size;
    const size_t total_size = batch_size * channels * spatial_size;

    // Apply bias channel-wise for NCHW layout
    for (int32_t b = 0; b < batch_size; ++b) {
        for (int32_t c = 0; c < channels; ++c) {
            const float bias_val = bias_data_[c];
            const size_t channel_offset = (b * channels + c) * spatial_size;
            
            for (int32_t hw = 0; hw < spatial_size; ++hw) {
                const size_t idx = channel_offset + hw;
                output[idx] = input[idx] + bias_val;
            }
        }
    }
    
    // Apply activation if configured
    if (config_.fused_activation) {
        apply_activation(output, total_size);
    }
}

void CmxBias::apply_element_wise_bias(const float* input, const TensorShape& input_shape,
                                     float* output, const TensorShape& output_shape) {
    const size_t total_size = input_shape.batch * input_shape.height * 
                             input_shape.width * input_shape.channels;

    // Element-wise bias addition
    for (size_t i = 0; i < total_size; ++i) {
        output[i] = input[i] + bias_data_[i];
    }
    
    // Apply activation if configured
    if (config_.fused_activation) {
        apply_activation(output, total_size);
    }
}

void CmxBias::apply_bias_vectorized(const float* input, float* output, size_t size, 
                                   const float* bias, size_t bias_size, size_t stride) {
    // Vectorized bias application with stride
    for (size_t i = 0; i < size; ++i) {
        const size_t bias_idx = (i / stride) % bias_size;
        output[i] = input[i] + bias[bias_idx];
    }
}

} // namespace kernels
} // namespace cmx