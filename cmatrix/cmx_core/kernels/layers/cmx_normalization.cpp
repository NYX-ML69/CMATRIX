#include "cmx_normalization.hpp"
#include <algorithm>
#include <cmath>

namespace cmx {
namespace kernels {

CmxNormalization::CmxNormalization()
    : norm_type_(NormalizationType::BATCH_NORM)
    , batch_size_(0)
    , height_(0)
    , width_(0)
    , channels_(0)
    , scale_(nullptr)
    , offset_(nullptr)
    , mean_(nullptr)
    , variance_(nullptr)
    , epsilon_(1e-5f)
    , activation_(PostNormActivation::NONE)
    , configured_(false) {
}

CmxNormalization::~CmxNormalization() {
}

bool CmxNormalization::configure(NormalizationType norm_type,
                                uint32_t batch_size, uint32_t height, uint32_t width, uint32_t channels,
                                const float* scale, const float* offset,
                                const float* mean, const float* variance,
                                float epsilon, PostNormActivation activation) {
    // Validate input parameters
    if (batch_size == 0 || height == 0 || width == 0 || channels == 0 ||
        scale == nullptr || offset == nullptr || epsilon <= 0.0f) {
        return false;
    }
    
    // For batch norm, mean and variance are required
    if (norm_type == NormalizationType::BATCH_NORM && (mean == nullptr || variance == nullptr)) {
        return false;
    }
    
    // Store configuration
    norm_type_ = norm_type;
    batch_size_ = batch_size;
    height_ = height;
    width_ = width;
    channels_ = channels;
    scale_ = scale;
    offset_ = offset;
    mean_ = mean;
    variance_ = variance;
    epsilon_ = epsilon;
    activation_ = activation;
    
    configured_ = true;
    return true;
}

bool CmxNormalization::run(const float* input_data, float* output_data) {
    if (!configured_ || !input_data || !output_data) {
        return false;
    }
    
    switch (norm_type_) {
        case NormalizationType::BATCH_NORM:
            execute_batch_norm(input_data, output_data);
            break;
        case NormalizationType::LAYER_NORM:
            execute_layer_norm(input_data, output_data);
            break;
        case NormalizationType::INSTANCE_NORM:
            execute_instance_norm(input_data, output_data);
            break;
        default:
            return false;
    }
    
    return true;
}

bool CmxNormalization::infer_shape(uint32_t& output_batch, uint32_t& output_height,
                                  uint32_t& output_width, uint32_t& output_channels) const {
    if (!configured_) {
        return false;
    }
    
    output_batch = batch_size_;
    output_height = height_;
    output_width = width_;
    output_channels = channels_;
    
    return true;
}

size_t CmxNormalization::get_memory_requirements() const {
    if (!configured_) {
        return 0;
    }
    
    // Memory for temporary statistics in layer/instance norm
    if (norm_type_ == NormalizationType::LAYER_NORM) {
        return 2 * sizeof(float); // mean and variance per sample
    } else if (norm_type_ == NormalizationType::INSTANCE_NORM) {
        return 2 * batch_size_ * channels_ * sizeof(float); // mean and variance per instance
    }
    
    return 0;
}

void CmxNormalization::execute_batch_norm(const float* input_data, float* output_data) {
    for (uint32_t batch = 0; batch < batch_size_; ++batch) {
        for (uint32_t h = 0; h < height_; ++h) {
            for (uint32_t w = 0; w < width_; ++w) {
                for (uint32_t c = 0; c < channels_; ++c) {
                    uint32_t idx = ((batch * height_ + h) * width_ + w) * channels_ + c;
                    
                    // Normalize: (x - mean) / sqrt(variance + epsilon)
                    float normalized = (input_data[idx] - mean_[c]) / std::sqrt(variance_[c] + epsilon_);
                    
                    // Scale and shift: gamma * normalized + beta
                    float result = scale_[c] * normalized + offset_[c];
                    
                    // Apply activation
                    result = apply_activation(result);
                    
                    output_data[idx] = result;
                }
            }
        }
    }
}

void CmxNormalization::execute_layer_norm(const float* input_data, float* output_data) {
    for (uint32_t batch = 0; batch < batch_size_; ++batch) {
        for (uint32_t h = 0; h < height_; ++h) {
            for (uint32_t w = 0; w < width_; ++w) {
                // Calculate mean and variance across channels for this spatial location
                float mean_val, var_val;
                uint32_t base_idx = ((batch * height_ + h) * width_ + w) * channels_;
                calculate_stats(&input_data[base_idx], channels_, mean_val, var_val);
                
                for (uint32_t c = 0; c < channels_; ++c) {
                    uint32_t idx = base_idx + c;
                    
                    // Normalize
                    float normalized = (input_data[idx] - mean_val) / std::sqrt(var_val + epsilon_);
                    
                    // Scale and shift
                    float result = scale_[c] * normalized + offset_[c];
                    
                    // Apply activation
                    result = apply_activation(result);
                    
                    output_data[idx] = result;
                }
            }
        }
    }
}

void CmxNormalization::execute_instance_norm(const float* input_data, float* output_data) {
    for (uint32_t batch = 0; batch < batch_size_; ++batch) {
        for (uint32_t c = 0; c < channels_; ++c) {
            // Calculate mean and variance for this instance (batch, channel)
            float mean_val = 0.0f;
            float var_val = 0.0f;
            uint32_t spatial_size = height_ * width_;
            
            // Calculate mean
            for (uint32_t h = 0; h < height_; ++h) {
                for (uint32_t w = 0; w < width_; ++w) {
                    uint32_t idx = ((batch * height_ + h) * width_ + w) * channels_ + c;
                    mean_val += input_data[idx];
                }
            }
            mean_val /= static_cast<float>(spatial_size);
            
            // Calculate variance
            for (uint32_t h = 0; h < height_; ++h) {
                for (uint32_t w = 0; w < width_; ++w) {
                    uint32_t idx = ((batch * height_ + h) * width_ + w) * channels_ + c;
                    float diff = input_data[idx] - mean_val;
                    var_val += diff * diff;
                }
            }
            var_val /= static_cast<float>(spatial_size);
            
            // Normalize all spatial locations for this instance
            for (uint32_t h = 0; h < height_; ++h) {
                for (uint32_t w = 0; w < width_; ++w) {
                    uint32_t idx = ((batch * height_ + h) * width_ + w) * channels_ + c;
                    
                    // Normalize
                    float normalized = (input_data[idx] - mean_val) / std::sqrt(var_val + epsilon_);
                    
                    // Scale and shift
                    float result = scale_[c] * normalized + offset_[c];
                    
                    // Apply activation
                    result = apply_activation(result);
                    
                    output_data[idx] = result;
                }
            }
        }
    }
}

float CmxNormalization::apply_activation(float value) const {
    switch (activation_) {
        case PostNormActivation::NONE:
            return value;
        case PostNormActivation::RELU:
            return std::max(0.0f, value);
        case PostNormActivation::RELU6:
            return std::max(0.0f, std::min(6.0f, value));
        case PostNormActivation::TANH:
            return std::tanh(value);
        case PostNormActivation::SIGMOID:
            return 1.0f / (1.0f + std::exp(-value));
        default:
            return value;
    }
}

void CmxNormalization::calculate_stats(const float* data, uint32_t size, float& mean, float& variance) const {
    // Calculate mean
    mean = 0.0f;
    for (uint32_t i = 0; i < size; ++i) {
        mean += data[i];
    }
    mean /= static_cast<float>(size);
    
    // Calculate variance
    variance = 0.0f;
    for (uint32_t i = 0; i < size; ++i) {
        float diff = data[i] - mean;
        variance += diff * diff;
    }
    variance /= static_cast<float>(size);
}

} // namespace kernels
} // namespace cmx