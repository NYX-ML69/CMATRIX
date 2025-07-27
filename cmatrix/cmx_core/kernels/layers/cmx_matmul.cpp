#include "cmx_matmul.hpp"
#include <algorithm>
#include <cmath>

namespace cmx {
namespace kernels {

CmxMatMul::CmxMatMul()
    : batch_size_(0)
    , input_rows_(0)
    , input_cols_(0)
    , output_cols_(0)
    , weights_(nullptr)
    , bias_(nullptr)
    , activation_(ActivationType::NONE)
    , transpose_weights_(false)
    , quantized_(false)
    , input_scale_(1.0f)
    , input_zero_point_(0)
    , weight_scale_(1.0f)
    , weight_zero_point_(0)
    , output_scale_(1.0f)
    , output_zero_point_(0)
    , configured_(false) {
}

CmxMatMul::~CmxMatMul() {
}

bool CmxMatMul::configure(uint32_t batch_size,
                         uint32_t input_rows, uint32_t input_cols, uint32_t output_cols,
                         const float* weights, const float* bias,
                         ActivationType activation, bool transpose_weights) {
    // Validate input parameters
    if (batch_size == 0 || input_rows == 0 || input_cols == 0 || 
        output_cols == 0 || weights == nullptr) {
        return false;
    }
    
    // Store configuration
    batch_size_ = batch_size;
    input_rows_ = input_rows;
    input_cols_ = input_cols;
    output_cols_ = output_cols;
    weights_ = weights;
    bias_ = bias;
    activation_ = activation;
    transpose_weights_ = transpose_weights;
    
    configured_ = true;
    return true;
}

bool CmxMatMul::run(const float* input_data, float* output_data) {
    if (!configured_ || !input_data || !output_data) {
        return false;
    }
    
    if (quantized_) {
        execute_quantized_matmul(input_data, output_data);
    } else {
        execute_float_matmul(input_data, output_data);
    }
    
    return true;
}

bool CmxMatMul::infer_shape(uint32_t& output_batch, uint32_t& output_rows,
                           uint32_t& output_cols) const {
    if (!configured_) {
        return false;
    }
    
    output_batch = batch_size_;
    output_rows = input_rows_;
    output_cols = output_cols_;
    
    return true;
}

size_t CmxMatMul::get_memory_requirements() const {
    if (!configured_) {
        return 0;
    }
    
    // No additional memory required for matrix multiplication
    return 0;
}

void CmxMatMul::set_quantization_params(float input_scale, int32_t input_zero_point,
                                       float weight_scale, int32_t weight_zero_point,
                                       float output_scale, int32_t output_zero_point) {
    input_scale_ = input_scale;
    input_zero_point_ = input_zero_point;
    weight_scale_ = weight_scale;
    weight_zero_point_ = weight_zero_point;
    output_scale_ = output_scale;
    output_zero_point_ = output_zero_point;
    quantized_ = true;
}

void CmxMatMul::execute_float_matmul(const float* input_data, float* output_data) {
    for (uint32_t batch = 0; batch < batch_size_; ++batch) {
        for (uint32_t row = 0; row < input_rows_; ++row) {
            for (uint32_t col = 0; col < output_cols_; ++col) {
                float sum = 0.0f;
                
                // Compute dot product
                for (uint32_t k = 0; k < input_cols_; ++k) {
                    uint32_t input_idx = (batch * input_rows_ + row) * input_cols_ + k;
                    float input_val = input_data[input_idx];
                    float weight_val = get_weight_value(k, col);
                    
                    sum += input_val * weight_val;
                }
                
                // Add bias if present
                sum = add_bias(sum, col);
                
                // Apply activation
                sum = apply_activation(sum);
                
                // Store result
                uint32_t output_idx = (batch * input_rows_ + row) * output_cols_ + col;
                output_data[output_idx] = sum;
            }
        }
    }
}

void CmxMatMul::execute_quantized_matmul(const float* input_data, float* output_data) {
    for (uint32_t batch = 0; batch < batch_size_; ++batch) {
        for (uint32_t row = 0; row < input_rows_; ++row) {
            for (uint32_t col = 0; col < output_cols_; ++col) {
                int32_t sum = 0;
                
                // Compute dot product in quantized space
                for (uint32_t k = 0; k < input_cols_; ++k) {
                    uint32_t input_idx = (batch * input_rows_ + row) * input_cols_ + k;
                    
                    // Quantize input
                    int32_t input_quant = static_cast<int32_t>(input_data[input_idx] / input_scale_) + input_zero_point_;
                    
                    // Quantize weight
                    float weight_val = get_weight_value(k, col);
                    int32_t weight_quant = static_cast<int32_t>(weight_val / weight_scale_) + weight_zero_point_;
                    
                    sum += (input_quant - input_zero_point_) * (weight_quant - weight_zero_point_);
                }
                
                // Dequantize result
                float result = static_cast<float>(sum) * input_scale_ * weight_scale_;
                
                // Add bias if present
                result = add_bias(result, col);
                
                // Apply activation
                result = apply_activation(result);
                
                // Store result
                uint32_t output_idx = (batch * input_rows_ + row) * output_cols_ + col;
                output_data[output_idx] = result;
            }
        }
    }
}

float CmxMatMul::apply_activation(float value) const {
    switch (activation_) {
        case ActivationType::NONE:
            return value;
        case ActivationType::RELU:
            return std::max(0.0f, value);
        case ActivationType::RELU6:
            return std::max(0.0f, std::min(6.0f, value));
        case ActivationType::TANH:
            return std::tanh(value);
        case ActivationType::SIGMOID:
            return 1.0f / (1.0f + std::exp(-value));
        default:
            return value;
    }
}

float CmxMatMul::get_weight_value(uint32_t row, uint32_t col) const {
    if (transpose_weights_) {
        // Weights are stored as (output_cols x input_cols)
        return weights_[col * input_cols_ + row];
    } else {
        // Weights are stored as (input_cols x output_cols)
        return weights_[row * output_cols_ + col];
    }
}

float CmxMatMul::add_bias(float value, uint32_t col) const {
    if (bias_ != nullptr) {
        return value + bias_[col];
    }
    return value;
}

} // namespace kernels
} // namespace cmx