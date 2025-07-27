/**
 * @file cmx_elu.hpp
 * @brief Exponential Linear Unit (ELU) activation function for embedded ML runtime
 * 
 * ELU activation function: f(x) = x if x > 0, else alpha * (exp(x) - 1)
 * Produces negative outputs for negative inputs, helping with mean activation
 * closer to zero. Common alpha value is 1.0.
 */

#pragma once

namespace cmx::kernels::activations {

/**
 * @brief Apply ELU activation function
 * @param input Input tensor data
 * @param output Output tensor data
 * @param size Number of elements to process
 * @param alpha Alpha parameter (default: 1.0)
 */
void elu(const float* input, float* output, int size, float alpha = 1.0f);

} // namespace cmx::kernels::activations