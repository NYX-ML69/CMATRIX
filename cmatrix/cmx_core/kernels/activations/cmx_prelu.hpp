/**
 * @file cmx_prelu.hpp
 * @brief Parametric ReLU activation function for embedded ML runtime
 * 
 * PReLU activation function: f(x) = x if x > 0, else alpha[i] * x
 * Similar to Leaky ReLU but with learnable alpha parameters per channel.
 * For embedded use, alpha values are typically pre-computed and fixed.
 */

#pragma once

namespace cmx::kernels::activations {

/**
 * @brief Apply Parametric ReLU activation function
 * @param input Input tensor data
 * @param output Output tensor data
 * @param size Number of elements to process
 * @param alpha Array of alpha values (one per channel/element)
 */
void prelu(const float* input, float* output, int size, const float* alpha);

/**
 * @brief Apply Parametric ReLU with single alpha value
 * @param input Input tensor data
 * @param output Output tensor data
 * @param size Number of elements to process
 * @param alpha Single alpha value for all elements
 */
void prelu(const float* input, float* output, int size, float alpha);

} // namespace cmx::kernels::activations