#pragma once

/**
 * @file cmx_math_utils.hpp
 * @brief Mathematical utility functions for embedded ML inference
 * 
 * Provides fast approximations and mathematical primitives optimized for
 * embedded systems. Includes activation function support, clamping, and
 * other common mathematical operations.
 */

namespace cmx {
namespace utils {

/**
 * @brief Clamp value between min and max
 * 
 * @param value Input value
 * @param min_val Minimum value
 * @param max_val Maximum value
 * @return Clamped value
 */
float clamp(float value, float min_val, float max_val);

/**
 * @brief Find minimum of two values
 * 
 * @param a First value
 * @param b Second value
 * @return Minimum value
 */
float min(float a, float b);

/**
 * @brief Find maximum of two values
 * 
 * @param a First value
 * @param b Second value
 * @return Maximum value
 */
float max(float a, float b);

/**
 * @brief Absolute value
 * 
 * @param value Input value
 * @return Absolute value
 */
float abs(float value);

/**
 * @brief Fast approximation of exponential function
 * 
 * Uses polynomial approximation for speed on embedded systems
 * 
 * @param x Input value
 * @return Approximated exp(x)
 */
float fast_exp(float x);

/**
 * @brief Fast approximation of natural logarithm
 * 
 * @param x Input value (must be positive)
 * @return Approximated ln(x)
 */
float fast_log(float x);

/**
 * @brief Fast approximation of hyperbolic tangent
 * 
 * @param x Input value
 * @return Approximated tanh(x)
 */
float fast_tanh(float x);

/**
 * @brief Fast approximation of sigmoid function
 * 
 * @param x Input value
 * @return Approximated sigmoid(x)
 */
float fast_sigmoid(float x);

/**
 * @brief Fast approximation of square root
 * 
 * @param x Input value (must be non-negative)
 * @return Approximated sqrt(x)
 */
float fast_sqrt(float x);

/**
 * @brief Fast approximation of reciprocal square root
 * 
 * @param x Input value (must be positive)
 * @return Approximated 1/sqrt(x)
 */
float fast_rsqrt(float x);

/**
 * @brief Fast approximation of power function
 * 
 * @param base Base value
 * @param exp Exponent
 * @return Approximated base^exp
 */
float fast_pow(float base, float exp);

/**
 * @brief ReLU activation function
 * 
 * @param x Input value
 * @return max(0, x)
 */
float relu(float x);

/**
 * @brief Leaky ReLU activation function
 * 
 * @param x Input value
 * @param alpha Slope for negative values (default: 0.01)
 * @return Leaky ReLU output
 */
float leaky_relu(float x, float alpha = 0.01f);

/**
 * @brief ELU activation function
 * 
 * @param x Input value
 * @param alpha Alpha parameter (default: 1.0)
 * @return ELU output
 */
float elu(float x, float alpha = 1.0f);

/**
 * @brief GELU activation function (approximated)
 * 
 * @param x Input value
 * @return Approximated GELU output
 */
float gelu(float x);

/**
 * @brief Swish activation function
 * 
 * @param x Input value
 * @return Swish output (x * sigmoid(x))
 */
float swish(float x);

/**
 * @brief Softplus activation function
 * 
 * @param x Input value
 * @return Softplus output (log(1 + exp(x)))
 */
float softplus(float x);

/**
 * @brief Apply softmax to an array of values
 * 
 * @param input Input array
 * @param output Output array
 * @param size Array size
 */
void softmax(const float* input, float* output, int size);

/**
 * @brief Apply layer normalization
 * 
 * @param input Input array
 * @param output Output array
 * @param size Array size
 * @param epsilon Small value for numerical stability (default: 1e-5)
 */
void layer_norm(const float* input, float* output, int size, float epsilon = 1e-5f);

/**
 * @brief Calculate mean of array
 * 
 * @param input Input array
 * @param size Array size
 * @return Mean value
 */
float mean(const float* input, int size);

/**
 * @brief Calculate variance of array
 * 
 * @param input Input array
 * @param size Array size
 * @param mean_val Pre-calculated mean (optional)
 * @return Variance value
 */
float variance(const float* input, int size, float mean_val = 0.0f);

/**
 * @brief Calculate standard deviation of array
 * 
 * @param input Input array
 * @param size Array size
 * @param mean_val Pre-calculated mean (optional)
 * @return Standard deviation
 */
float std_dev(const float* input, int size, float mean_val = 0.0f);

/**
 * @brief Element-wise addition
 * 
 * @param a First array
 * @param b Second array
 * @param output Output array
 * @param size Array size
 */
void add(const float* a, const float* b, float* output, int size);

/**
 * @brief Element-wise subtraction
 * 
 * @param a First array
 * @param b Second array
 * @param output Output array
 * @param size Array size
 */
void subtract(const float* a, const float* b, float* output, int size);

/**
 * @brief Element-wise multiplication
 * 
 * @param a First array
 * @param b Second array
 * @param output Output array
 * @param size Array size
 */
void multiply(const float* a, const float* b, float* output, int size);

/**
 * @brief Element-wise division
 * 
 * @param a First array
 * @param b Second array
 * @param output Output array
 * @param size Array size
 */
void divide(const float* a, const float* b, float* output, int size);

/**
 * @brief Scale array by constant
 * 
 * @param input Input array
 * @param output Output array
 * @param size Array size
 * @param scale Scale factor
 */
void scale(const float* input, float* output, int size, float scale);

/**
 * @brief Add constant to array
 * 
 * @param input Input array
 * @param output Output array
 * @param size Array size
 * @param bias Bias value to add
 */
void add_bias(const float* input, float* output, int size, float bias);

} // namespace utils
} // namespace cmx