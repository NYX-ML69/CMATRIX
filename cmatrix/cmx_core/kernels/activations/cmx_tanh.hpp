/**
 * @file cmx_tanh.hpp
 * @brief Hyperbolic tangent activation function for embedded ML runtime
 * 
 * Tanh activation function: f(x) = tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
 * Maps any real value to (-1, 1), often preferred over sigmoid for hidden layers
 * due to zero-centered output.
 */

#pragma once

namespace cmx::kernels::activations {

/**
 * @brief Apply hyperbolic tangent activation function
 * @param input Input tensor data
 * @param output Output tensor data
 * @param size Number of elements to process
 */
void tanh(const float* input, float* output, int size);

} // namespace cmx::kernels::activations