/**
 * @file cmx_softmax.hpp
 * @brief Softmax activation function for embedded ML runtime
 * 
 * Softmax activation function: f(x_i) = exp(x_i) / Σ(exp(x_j))
 * Converts a vector of real numbers to a probability distribution.
 * Uses numerically stable implementation with max subtraction to prevent overflow.
 */

#pragma once

namespace cmx::kernels::activations {

/**
 * @brief Apply Softmax activation function
 * @param input Input tensor data
 * @param output Output tensor data
 * @param size Number of elements to process
 */
void softmax(const float* input, float* output, int size);

} // namespace cmx::kernels::activations