/**
 * @file cmx_swish.hpp
 * @brief Swish activation function for embedded ML runtime
 * 
 * Swish activation function: f(x) = x * sigmoid(x) = x / (1 + exp(-x))
 * Smooth, non-monotonic activation that can outperform ReLU in some cases.
 * Also known as SiLU (Sigmoid Linear Unit).
 */

#pragma once

namespace cmx::kernels::activations {

/**
 * @brief Apply Swish activation function
 * @param input Input tensor data
 * @param output Output tensor data
 * @param size Number of elements to process
 */
void swish(const float* input, float* output, int size);

} // namespace cmx::kernels::activations