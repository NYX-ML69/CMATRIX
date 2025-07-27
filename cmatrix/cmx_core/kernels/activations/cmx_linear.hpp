V/**
 * @file cmx_linear.hpp
 * @brief Linear (Identity) activation function for embedded ML runtime
 * 
 * Linear activation simply passes input through unchanged: f(x) = x
 * This is commonly used as a default activation or in output layers
 * for regression tasks.
 */

#pragma once

namespace cmx::kernels::activations {

/**
 * @brief Apply linear (identity) activation function
 * @param input Input tensor data
 * @param output Output tensor data (can be same as input)
 * @param size Number of elements to process
 */
void linear(const float* input, float* output, int size);

} // namespace cmx::kernels::activations