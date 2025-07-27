/**
 * @file cmx_leaky_relu.hpp
 * @brief Leaky ReLU activation function for embedded ML runtime
 * 
 * Leaky ReLU activation function: f(x) = x if x > 0, else alpha * x
 * Allows small negative values to pass through, preventing "dying ReLU" problem.
 * Common alpha values are 0.01 or 0.1.
 */

#pragma once

namespace cmx::kernels::activations {

/**
 * @brief Apply Leaky ReLU activation function
 * @param input Input tensor data
 * @param output Output tensor data
 * @param size Number of elements to process
 * @param alpha Slope for negative values (default: 0.01)
 */
void leaky_relu(const float* input, float* output, int size, float alpha = 0.01f);

} // namespace cmx::kernels::activations