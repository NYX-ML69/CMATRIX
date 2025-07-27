/**
 * @file cmx_maxout.hpp
 * @brief Maxout activation function for embedded ML runtime
 * 
 * Maxout activation function: f(x) = max(x_1, x_2, ..., x_k)
 * Takes the maximum over groups of k units. Commonly used with k=2.
 * Can approximate any convex function and is useful in certain architectures.
 */

#pragma once

namespace cmx::kernels::activations {

/**
 * @brief Apply Maxout activation function
 * @param input Input tensor data
 * @param output Output tensor data
 * @param input_size Total number of input elements
 * @param group_size Number of units per group (k)
 */
void maxout(const float* input, float* output, int input_size, int group_size);

/**
 * @brief Apply Maxout activation function with group_size = 2
 * @param input Input tensor data
 * @param output Output tensor data
 * @param input_size Total number of input elements (must be even)
 */
void maxout_pairs(const float* input, float* output, int input_size);

} // namespace cmx::kernels::activations