/**
 * @file cmx_softplus.hpp
 * @brief Softplus activation function for embedded ML runtime
 * 
 * Softplus activation function: f(x) = log(1 + exp(x))
 * Smooth approximation to ReLU that's always positive and differentiable.
 * Approaches ReLU for large positive x and approaches 0 for large negative x.
 */

#pragma once

namespace cmx::kernels::activations {

/**
 * @brief Apply Softplus activation function
 * @param input Input tensor data
 * @param output Output tensor data
 * @param size Number of elements to process
 */
void softplus(const float* input, float* output, int size);

} // namespace cmx::kernels::activations