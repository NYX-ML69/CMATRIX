#include "cmx_quantization.hpp"
#include <cmath>
#include <algorithm>

namespace cmx {
namespace utils {

QuantizationParams calculate_quantization_params(
    float min_val, 
    float max_val, 
    QuantizationScheme scheme,
    int8_t qmin,
    int8_t qmax
) {
    QuantizationParams params;
    params.qmin = qmin;
    params.qmax = qmax;
    
    // Handle edge cases
    if (min_val == max_val) {
        params.scale = 1.0f;
        params.zero_point = 0;
        return params;
    }
    
    // Ensure min_val < max_val
    if (min_val > max_val) {
        float temp = min_val;
        min_val = max_val;
        max_val = temp;
    }
    
    if (scheme == QuantizationScheme::SYMMETRIC) {
        // Symmetric quantization: zero_point = 0
        params.zero_point = 0;
        float max_range = std::max(std::abs(min_val), std::abs(max_val));
        params.scale = (2.0f * max_range) / (qmax - qmin);
    } else {
        // Asymmetric quantization
        params.scale = (max_val - min_val) / (qmax - qmin);
        
        // Calculate zero_point
        float zero_point_from_min = qmin - min_val / params.scale;
        float zero_point_from_max = qmax - max_val / params.scale;
        float zero_point_clamped = std::max(
            static_cast<float>(qmin),
            std::min(static_cast<float>(qmax), zero_point_from_min)
        );
        params.zero_point = static_cast<int32_t>(fast_round(zero_point_clamped));
    }
    
    return params;
}

int8_t quantize_value(float value, const QuantizationParams& params) {
    int32_t quantized = fast_round(value / params.scale) + params.zero_point;
    return clamp_quantized(quantized, params.qmin, params.qmax);
}

float dequantize_value(int8_t qvalue, const QuantizationParams& params) {
    return params.scale * (static_cast<int32_t>(qvalue) - params.zero_point);
}

void quantize_tensor(
    const float* input, 
    int8_t* output, 
    int32_t size, 
    const QuantizationParams& params
) {
    for (int32_t i = 0; i < size; ++i) {
        output[i] = quantize_value(input[i], params);
    }
}

void dequantize_tensor(
    const int8_t* input, 
    float* output, 
    int32_t size, 
    const QuantizationParams& params
) {
    for (int32_t i = 0; i < size; ++i) {
        output[i] = dequantize_value(input[i], params);
    }
}

void requantize_tensor(
    const int8_t* input,
    int8_t* output,
    int32_t size,
    const QuantizationParams& input_params,
    const QuantizationParams& output_params
) {
    // Calculate combined scale and zero point transformation
    float scale_ratio = input_params.scale / output_params.scale;
    int32_t zero_point_diff = input_params.zero_point - output_params.zero_point;
    
    for (int32_t i = 0; i < size; ++i) {
        // Efficient requantization without intermediate float conversion
        int32_t input_val = static_cast<int32_t>(input[i]);
        int32_t transformed = fast_round(scale_ratio * (input_val - input_params.zero_point)) + output_params.zero_point;
        output[i] = clamp_quantized(transformed, output_params.qmin, output_params.qmax);
    }
}

void find_tensor_range(
    const float* input, 
    int32_t size, 
    float& min_val, 
    float& max_val
) {
    if (size <= 0) {
        min_val = max_val = 0.0f;
        return;
    }
    
    min_val = max_val = input[0];
    
    for (int32_t i = 1; i < size; ++i) {
        float val = input[i];
        if (val < min_val) {
            min_val = val;
        }
        if (val > max_val) {
            max_val = val;
        }
    }
}

} // namespace utils
} // namespace cmx