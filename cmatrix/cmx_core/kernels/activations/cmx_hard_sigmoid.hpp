/**
 * @file cmx_hard_sigmoid.hpp
 * @brief Hard Sigmoid activation function for embedded ML runtime
 * 
 * Hard Sigmoid activation function: f(x) = max(0, min(1, 0.2 * x + 0.5))
 * Piecewise linear approximation of sigmoid that's computationally efficient.
 * No exponential operations required, making it ideal for embedded systems.
 */

#pragma once

namespace cmx::kernels::activations {

/**
 * @brief Apply Hard Sigmoid activation function
 * @param input Input tensor data
 * @param output Output tensor data
 * @param size Number of elements to process
 */
void hard_sigmoid(const float* input, float* output, int size);

} // namespace cmx::kernels::activations