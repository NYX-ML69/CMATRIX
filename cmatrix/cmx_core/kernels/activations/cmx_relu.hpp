/**
 * @file cmx_relu.hpp
 * @brief Rectified Linear Unit (ReLU) activation function for embedded ML runtime
 * 
 * ReLU activation function: f(x) = max(0, x)
 * Simple and computationally efficient, widely used in deep neural networks.
 * Helps mitigate vanishing gradient problem.
 */

#pragma once

namespace cmx::kernels::activations {

/**
 * @brief Apply ReLU activation function
 * @param input Input tensor data
 * @param output Output tensor data
 * @param size Number of elements to process
 */
void relu(const float* input, float* output, int size);

} // namespace cmx::kernels::activations