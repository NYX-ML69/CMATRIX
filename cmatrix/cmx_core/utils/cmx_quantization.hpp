#pragma once

#include <cstdint>
#include <cmath>

namespace cmx {
namespace utils {

/**
 * @brief Quantization parameters for int8 and fixed-point operations
 */
struct QuantizationParams {
    float scale;         ///< Scaling factor for quantization
    int32_t zero_point;  ///< Zero point offset for asymmetric quantization
    int8_t qmin;         ///< Minimum quantized value (typically -128)
    int8_t qmax;         ///< Maximum quantized value (typically 127)
};

/**
 * @brief Quantization scheme types
 */
enum class QuantizationScheme {
    SYMMETRIC,    ///< Symmetric quantization (zero_point = 0)
    ASYMMETRIC    ///< Asymmetric quantization (zero_point != 0)
};

/**
 * @brief Calculate quantization parameters from float tensor statistics
 * @param min_val Minimum value in the tensor
 * @param max_val Maximum value in the tensor
 * @param scheme Quantization scheme to use
 * @param qmin Minimum quantized value
 * @param qmax Maximum quantized value
 * @return Quantization parameters
 */
QuantizationParams calculate_quantization_params(
    float min_val, 
    float max_val, 
    QuantizationScheme scheme = QuantizationScheme::ASYMMETRIC,
    int8_t qmin = -128,
    int8_t qmax = 127
);

/**
 * @brief Quantize a single float value to int8
 * @param value Float value to quantize
 * @param params Quantization parameters
 * @return Quantized int8 value
 */
int8_t quantize_value(float value, const QuantizationParams& params);

/**
 * @brief Dequantize a single int8 value to float
 * @param qvalue Quantized int8 value
 * @param params Quantization parameters
 * @return Dequantized float value
 */
float dequantize_value(int8_t qvalue, const QuantizationParams& params);

/**
 * @brief Quantize a tensor from float to int8
 * @param input Input float tensor
 * @param output Output int8 tensor (pre-allocated)
 * @param size Number of elements in the tensor
 * @param params Quantization parameters
 */
void quantize_tensor(
    const float* input, 
    int8_t* output, 
    int32_t size, 
    const QuantizationParams& params
);

/**
 * @brief Dequantize a tensor from int8 to float
 * @param input Input int8 tensor
 * @param output Output float tensor (pre-allocated)
 * @param size Number of elements in the tensor
 * @param params Quantization parameters
 */
void dequantize_tensor(
    const int8_t* input, 
    float* output, 
    int32_t size, 
    const QuantizationParams& params
);

/**
 * @brief Requantize from one quantization scheme to another
 * @param input Input quantized tensor
 * @param output Output quantized tensor (pre-allocated)
 * @param size Number of elements
 * @param input_params Input quantization parameters
 * @param output_params Output quantization parameters
 */
void requantize_tensor(
    const int8_t* input,
    int8_t* output,
    int32_t size,
    const QuantizationParams& input_params,
    const QuantizationParams& output_params
);

/**
 * @brief Find min/max values in a float tensor for quantization parameter calculation
 * @param input Input float tensor
 * @param size Number of elements
 * @param min_val Output minimum value
 * @param max_val Output maximum value
 */
void find_tensor_range(
    const float* input, 
    int32_t size, 
    float& min_val, 
    float& max_val
);

/**
 * @brief Clamp a value to quantization range
 * @param value Value to clamp
 * @param qmin Minimum quantized value
 * @param qmax Maximum quantized value
 * @return Clamped value
 */
inline int8_t clamp_quantized(int32_t value, int8_t qmin, int8_t qmax) {
    if (value < qmin) return qmin;
    if (value > qmax) return qmax;
    return static_cast<int8_t>(value);
}

/**
 * @brief Fast rounding for quantization (avoids floating point round)
 * @param value Value to round
 * @return Rounded integer value
 */
inline int32_t fast_round(float value) {
    return static_cast<int32_t>(value + (value >= 0.0f ? 0.5f : -0.5f));
}

} // namespace utils
} // namespace cmx