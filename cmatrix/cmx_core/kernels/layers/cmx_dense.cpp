#include "cmx_dense.hpp"
#include <cstring>
#include <algorithm>
#include <cmath>

namespace cmx {
namespace kernels {

CmxDense::CmxDense() : weights_(nullptr), bias_(nullptr), is_configured_(false) {}

CmxDense::~CmxDense() {}

bool CmxDense::configure(const Config& config, const float* weights, const float* bias) {
    if (!weights) {
        return false;
    }

    config_ = config;
    weights_ = weights;
    bias_ = bias;
    
    // Validate configuration
    if (config_.input_units <= 0 || config_.output_units <= 0) {
        return false;
    }
    
    if (config_.use_bias && !bias) {
        return false;
    }

    is_configured_ = true;
    return true;
}

bool CmxDense::run(const float* input, const TensorShape& input_shape,
                   float* output, const TensorShape& output_shape) {
    if (!is_configured_ || !input || !output) {
        return false;
    }

    // Validate shapes
    if (input_shape.features != config_.input_units) {
        return false;
    }
    
    if (output_shape.features != config_.output_units) {
        return false;
    }

    // Choose implementation based on matrix size
    if (config_.input_units <= 128 && config_.output_units <= 128) {
        matmul_naive(input, input_shape, output, output_shape);
    } else if (config_.input_units >= 512 && config_.output_units >= 512) {
        matmul_blocked(input, input_shape, output, output_shape);
    } else {
        #ifdef __ARM_NEON
            matmul_simd(input, input_shape, output, output_shape);
        #else
            matmul_naive(input, input_shape, output, output_shape);
        #endif
    }

    return true;
}

bool CmxDense::infer_shape(const TensorShape& input_shape, TensorShape& output_shape) {
    if (!is_configured_) {
        return false;
    }

    output_shape.batch = input_shape.batch;
    output_shape.features = config_.output_units;
    return true;
}

bool CmxDense::get_memory_requirements(const TensorShape& input_shape,
                                      size_t& weight_memory, size_t& workspace_memory) {
    if (!is_configured_) {
        return false;
    }

    // Weight memory
    weight_memory = config_.input_units * config_.output_units * sizeof(float);
    
    if (config_.use_bias) {
        weight_memory += config_.output_units * sizeof(float);
    }

    // Workspace memory for blocked matrix multiplication
    workspace_memory = 0;
    if (config_.input_units >= 512 && config_.output_units >= 512) {
        workspace_memory = 64 * 64 * sizeof(float); // Block size buffer
    }

    return true;
}

void CmxDense::apply_activation(float* data, size_t size) {
    if (!config_.fused_activation) {
        return;
    }

    switch (config_.activation_type) {
        case ActivationType::RELU:
            vector_relu(data, size);
            break;
        case ActivationType::RELU6:
            vector_relu6(data, size);
            break;
        case ActivationType::TANH:
            vector_tanh(data, size);
            break;
        case ActivationType::SIGMOID:
            vector_sigmoid(data, size);
            break;
        case ActivationType::SOFTMAX:
            // Softmax needs special handling for batch dimension
            break;
        default:
            break;
    }
}

void CmxDense::apply_softmax(float* data, size_t batch_size, size_t features) {
    for (size_t b = 0; b < batch_size; ++b) {
        float* batch_data = data + b * features;
        
        // Find max for numerical stability
        float max_val = batch_data[0];
        for (size_t i = 1; i < features; ++i) {
            max_val = std::max(max_val, batch_data[i]);
        }
        
        // Compute exponentials and sum
        float sum = 0.0f;
        for (size_t i = 0; i < features; ++i) {
            batch_data[i] = std::exp(batch_data[i] - max_val);
            sum += batch_data[i];
        }
        
        // Normalize
        for (size_t i = 0; i < features; ++i) {
            batch_data[i] /= sum;
        }
    }
}

void CmxDense::matmul_naive(const float* input, const TensorShape& input_shape,
                           float* output, const TensorShape& output_shape) {
    const int32_t batch_size = input_shape.batch;
    const int32_t input_units = config_.input_units;
    const int32_t output_units = config_.output_units;

    // Initialize output with bias if present
    if (config_.use_bias && bias_) {
        for (int32_t b = 0; b < batch_size; ++b) {
            std::memcpy(output + b * output_units, bias_, output_units * sizeof(float));
        }
    } else {
        std::memset(output, 0, batch_size * output_units * sizeof(float));
    }

    // Matrix multiplication
    for (int32_t b = 0; b < batch_size; ++b) {
        for (int32_t i = 0; i < output_units; ++i) {
            float sum = config_.use_bias && bias_ ? bias_[i] : 0.0f;
            
            for (int32_t j = 0; j < input_units; ++j) {
                if (config_.transpose_weights) {
                    sum += input[b * input_units + j] * weights_[i * input_units + j];
                } else {
                    sum += input[b * input_units + j] * weights_[j * output_units + i];
                }
            }
            
            output[b * output_units + i] = sum;
        }
    }
    
    // Apply activation
    if (config_.fused_activation) {
        if (config_.activation_type == ActivationType::SOFTMAX) {
            apply_softmax(output, batch_size, output_units);
        } else {
            apply_activation(output, batch_size * output_units);
        }
    }
}

void CmxDense::matmul_blocked(const float* input, const TensorShape& input_shape,
                             float* output, const TensorShape& output_shape) {
    const int32_t batch_size = input_shape.batch;
    const int32_t input_units = config_.input_units;
    const int32_t output_units = config_.output_units;
    const int32_t block_size = 64;

    // Initialize output with bias if present
    if (config_.use_bias && bias_) {
        for (int32_t b = 0; b < batch_size; ++b) {
            std::memcpy(output + b * output_units, bias_, output_units * sizeof(float));
        }
    } else {
        std::memset(output, 0, batch_size * output_units * sizeof(float));
    }

    // Blocked matrix multiplication
    for (int32_t b = 0; b < batch_size; ++b) {
        for (int32_t ii = 0; ii < output_units; ii += block_size) {
            for (int32_t jj = 0; jj < input_units; jj += block_size) {
                int32_t i_end = std::min(ii + block_size, output_units);
                int32_t j_end = std::min(jj + block_size, input_units);
                
                for (int32_t i = ii; i < i_end; ++i) {
                    float sum = 0.0f;
                    for (int32_t j = jj; j < j_end; ++j) {
                        if (config_.transpose_weights) {
                            sum += input[b * input_units + j] * weights_[i * input_units + j];
                        } else {
                            sum += input[b * input_units + j] * weights_[j * output_units + i];
                        }
                    }
                    output[b * output_units + i] += sum;
                }
            }
        }
    }
    
    // Apply activation
    if (config_.fused_activation) {
        if (config_.activation_type == ActivationType::SOFTMAX) {
            apply_softmax(output, batch_size, output_units);
        } else {
            apply_activation(output, batch_size * output_units);
        }
    }
}

void CmxDense::matmul_simd(const float* input, const TensorShape& input_shape,
                          float* output, const TensorShape& output_shape) {
    // SIMD optimized implementation would go here
    // For now, fall back to naive implementation
    matmul_naive(input, input_shape, output, output_shape);
}

void CmxDense::vector_add_bias(float* output, const float* bias, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        output[i] += bias[i];
    }
}

void CmxDense::vector_relu(float* data, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        data[i] = std::max(0.0f, data[i]);
    }
}

void CmxDense::vector_relu6(float* data, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        data[i] = std::min(6.0f, std::max(0.0f, data[i]));
    }
}

void CmxDense::vector_tanh(float* data, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        data[i] = std::tanh(data[i]);
    }
}

void CmxDense::vector_sigmoid(float* data, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        data[i] = 1.0f / (1.0f + std::exp(-data[i]));
    }
}

} // namespace kernels
} // namespace cmx