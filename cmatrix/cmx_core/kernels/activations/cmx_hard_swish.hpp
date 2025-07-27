/**
 * @file cmx_hard_swish.hpp
 * @brief Hard Swish activation function for embedded ML runtime
 * 
 * Hard Swish activation function: f(x) = x * hard_sigmoid(x)
 * Efficient approximation of Swish using Hard Sigmoid, eliminating exponential
 * operations. Popular in mobile-optimized neural networks like MobileNetV3.
 */

#pragma once

namespace cmx::kernels::activations {

/**
 * @brief Apply Hard Swish activation function
 * @param input Input tensor data
 * @param output Output tensor data
 * @param size Number of elements to process
 */
void hard_swish(const float* input, float* output, int size);

} // namespace cmx::kernels::activations