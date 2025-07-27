/**
 * @file cmx_sigmoid.hpp
 * @brief Sigmoid activation function for embedded ML runtime
 * 
 * Sigmoid activation function: f(x) = 1 / (1 + exp(-x))
 * Maps any real value to (0, 1), commonly used in binary classification
 * and as a gating function in neural networks.
 */

#pragma once

namespace cmx::kernels::activations {

/**
 * @brief Apply sigmoid activation function
 * @param input Input tensor data
 * @param output Output tensor data
 * @param size Number of elements to process
 */
void sigmoid(const float* input, float* output, int size);

} // namespace cmx::kernels::activations