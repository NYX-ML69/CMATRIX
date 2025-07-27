/**
 * @file cmx_gelu.hpp
 * @brief Gaussian Error Linear Unit (GELU) activation function for embedded ML runtime
 * 
 * GELU activation function: f(x) = x * Φ(x) where Φ is the cumulative distribution
 * function of standard normal distribution. Commonly approximated as:
 * f(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
 * Used extensively in transformer models.
 */

#pragma once

namespace cmx::kernels::activations {

/**
 * @brief Apply GELU activation function (approximation)
 * @param input Input tensor data
 * @param output Output tensor data
 * @param size Number of elements to process
 */
void gelu(const float* input, float* output, int size);

} // namespace cmx::kernels::activations