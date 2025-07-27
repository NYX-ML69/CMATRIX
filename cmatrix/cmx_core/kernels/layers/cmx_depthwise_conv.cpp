#include "cmx_depthwise_conv.hpp"
#include <cstring>
#include <algorithm>
#include <cmath>

namespace cmx {
namespace kernels {

CmxDepthwiseConv::CmxDepthwiseConv() : weights_(nullptr), bias_(nullptr), is_configured_(false) {}

CmxDepthwiseConv::~CmxDepthwiseConv() {}

bool CmxDepthwiseConv::configure(const Config& config, const float* weights, const float* bias) {
    if (!weights) {
        return false;
    }

    config_ = config;
    weights_ = weights;
    bias_ = bias;
    
    // Validate configuration
    if (config_.kernel_size[0] <= 0 || config_.kernel_size[1] <= 0) {
        return false;
    }
    
    if (config_.strides[0] <= 0 || config_.strides[1] <= 0) {
        return false;
    }
    
    if (config_.input_channels <= 0 || config_.depth_multiplier <= 0) {
        return false;
    }

    calculate_padding();
    is_configured_ = true;
    return true;
}

bool CmxDepthwiseConv::run(const float* input, const TensorShape& input_shape,
                          float* output, const TensorShape& output_shape) {
    if (!is_configured_ || !input || !output) {
        return false;
    }

    // Choose implementation based on hardware capabilities
    #ifdef __ARM_NEON
        depthwise_conv_simd(input, input_shape, output, output_shape);
    #else
        depthwise_conv_direct(input, input_shape, output, output_shape);
    #endif

    return true;
}

bool CmxDepthwiseConv::infer_shape(const TensorShape& input_shape, TensorShape& output_shape) {
    if (!is_configured_) {
        return false;
    }

    calculate_output_size(input_shape, output_shape);
    return true;
}

bool CmxDepthwiseConv::get_memory_requirements(const TensorShape& input_shape,
                                             size_t& weight_memory, size_t& workspace_memory) {
    if (!is_configured_) {
        return false;
    }

    // Weight memory for depthwise filters
    weight_memory = config_.input_channels * config_.depth_multiplier * 
                   config_.kernel_size[0] * config_.kernel_size[1] * sizeof(float);
    
    if (config_.use_bias) {
        weight_memory += config_.input_channels * config_.depth_multiplier * sizeof(float);
    }

    // Minimal workspace memory needed
    workspace_memory = 64; // Small buffer for intermediate calculations

    return true;
}

void CmxDepthwiseConv::calculate_output_size(const TensorShape& input_shape, TensorShape& output_shape) {
    output_shape.batch = input_shape.batch;
    output_shape.channels = config_.input_channels * config_.depth_multiplier;

    // Calculate output dimensions
    output_shape.height = (input_shape.height + config_.padding[0] + config_.padding[1] - 
                          config_.dilation[0] * (config_.kernel_size[0] - 1) - 1) / config_.strides[0] + 1;
    
    output_shape.width = (input_shape.width + config_.padding[2] + config_.padding[3] - 
                         config_.dilation[1] * (config_.kernel_size[1] - 1) - 1) / config_.strides[1] + 1;
}

void CmxDepthwiseConv::calculate_padding() {
    if (config_.padding_mode == PaddingMode::SAME) {
        // Calculate padding for SAME mode
        int32_t pad_h = (config_.kernel_size[0] - 1) * config_.dilation[0];
        int32_t pad_w = (config_.kernel_size[1] - 1) * config_.dilation[1];
        
        config_.padding[0] = pad_h / 2;       // top
        config_.padding[1] = pad_h - config_.padding[0];  // bottom
        config_.padding[2] = pad_w / 2;       // left
        config_.padding[3] = pad_w - config_.padding[2];  // right
    } else if (config_.padding_mode == PaddingMode::VALID) {
        // No padding for VALID mode
        std::fill(config_.padding.begin(), config_.padding.end(), 0);
    }
    // For EXPLICIT mode, padding is already set by user
}

void CmxDepthwiseConv::apply_activation(float* data, size_t size) {
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

void CmxDepthwiseConv::depthwise_conv_direct(const float* input, const TensorShape& input_shape,
                                           float* output, const TensorShape& output_shape) {
    const int32_t batch_size = input_shape.batch;
    const int32_t in_height = input_shape.height;
    const int32_t in_width = input_shape.width;
    const int32_t in_channels = input_shape.channels;
    
    const int32_t out_height = output_shape.height;
    const int32_t out_width = output_shape.width;
    const int32_t out_channels = output_shape.channels;

    // Process each batch
    for (int32_t b = 0; b < batch_size; ++b) {
        // Process each input channel
        for (int32_t ic = 0; ic < in_channels; ++ic) {
            // Process each depth multiplier
            for (int32_t dm = 0; dm < config_.depth_multiplier; ++dm) {
                int32_t oc = ic * config_.depth_multiplier + dm;
                
                // Process each output spatial location
                for (int32_t oh = 0; oh < out_height; ++oh) {
                    for (int32_t ow = 0; ow < out_width; ++ow) {
                        float sum = 0.0f;
                        
                        // Apply bias if present
                        if (config_.use_bias && bias_) {
                            sum = bias_[oc];
                        }
                        
                        // Apply depthwise filter
                        for (int32_t kh = 0; kh < config_.kernel_size[0]; ++kh) {
                            for (int32_t kw = 0; kw < config_.kernel_size[1]; ++kw) {
                                int32_t ih = oh * config_.strides[0] + kh * config_.dilation[0] - config_.padding[0];
                                int32_t iw = ow * config_.strides[1] + kw * config_.dilation[1] - config_.padding[2];
                                
                                if (ih >= 0 && ih < in_height && iw >= 0 && iw < in_width) {
                                    int32_t input_idx = ((b * in_height + ih) * in_width + iw) * in_channels + ic;
                                    int32_t weight_idx = (oc * config_.kernel_size[0] + kh) * config_.kernel_size[1] + kw;
                                    
                                    sum += input[input_idx] * weights_[weight_idx];
                                }
                            }
                        }
                        
                        int32_t output_idx = ((b * out_height + oh) * out_width + ow) * out_channels + oc;
                        output[output_idx] = sum;
                    }
                }
            }
        }
    }
    
    // Apply activation if configured
    if (config_.fused_activation) {
        apply_activation(output, batch_size * out_height * out_width * out_channels);
    }
}

void CmxDepthwiseConv::depthwise_conv_simd(const float* input, const TensorShape& input_shape,
                                         float* output, const TensorShape& output_shape) {
    // SIMD optimized implementation would go here
    // For now, fall back to direct implementation
    depthwise_conv_direct(input, input_shape, output, output_shape);
}

void CmxDepthwiseConv::process_channel(const float* input_channel, float* output_channel,
                                     int32_t channel_idx, const TensorShape& input_shape,
                                     const TensorShape& output_shape) {
    // Process a single channel with its depthwise filters
    const int32_t in_height = input_shape.height;
    const int32_t in_width = input_shape.width;
    const int32_t out_height = output_shape.height;
    const int32_t out_width = output_shape.width;
    
    for (int32_t dm = 0; dm < config_.depth_multiplier; ++dm) {
        int32_t filter_idx = channel_idx * config_.depth_multiplier + dm;
        
        for (int32_t oh = 0; oh < out_height; ++oh) {
            for (int32_t ow = 0; ow < out_width; ++ow) {
                float sum = 0.0f;
                
                // Apply bias if present
                if (config_.use_bias && bias_) {
                    sum = bias_[filter_idx];
                }
                
                // Apply depthwise filter
                for (int32_t kh = 0; kh < config_.kernel_size[0]; ++kh) {
                    for (int32_t kw = 0; kw < config_.kernel_size[1]; ++kw) {
                        int32_t ih = oh * config_.strides[0] + kh * config_.dilation[0] - config_.padding[0];
                        int32_t iw = ow * config_.strides[1] + kw * config_.dilation[1] - config_.padding[2];
                        
                        if (ih >= 0 && ih < in_height && iw >= 0 && iw < in_width) {
                            int32_t input_idx = ih * in_width + iw;
                            int32_t weight_idx = (filter_idx * config_.kernel_size[0] + kh) * config_.kernel_size[1] + kw;
                            
                            sum += input_channel[input_idx] * weights_[weight_idx];
                        }
                    }
                }
                
                int32_t output_idx = oh * out_width + ow;
                output_channel[output_idx] = sum;
            }
        }
    }
}