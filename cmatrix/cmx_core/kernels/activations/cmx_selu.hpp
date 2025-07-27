/**
 * @file cmx_selu.hpp
 * @brief Scaled Exponential Linear Unit (SELU) activation function for embedded ML runtime
 * 
 * SELU activation function: f(x) = scale * (x if x > 0, else alpha * (exp(x) - 1))
 * Self-normalizing activation with specific scale and alpha values that preserve
 * mean and variance. Standard values: alpha ≈ 1.6733, scale ≈ 1.0507.
 */

#pragma once

namespace cmx::kernels::activations {

/**
 * @brief Apply SELU activation function
 * @param input Input tensor data
 * @param output Output tensor data
 * @param size Number of elements to process
 * @param alpha Alpha parameter (default: 1.6732632423543772848170429916717)
 * @param scale Scale parameter (default: 1.0507009873554804934193349852946)
 */
void selu(const float* input, float* output, int size, 
          float alpha = 1.6732632423543772848170429916717f, 
          float scale = 1.0507009873554804934193349852946f);

} // namespace cmx::kernels::activations