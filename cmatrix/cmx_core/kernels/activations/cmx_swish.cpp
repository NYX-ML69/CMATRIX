/**
 * @file cmx_swish.cpp
 * @brief Implementation of Swish activation function
 */

#include "cmx_swish.hpp"
#include <cmath>

namespace cmx::kernels::activations {

void swish(const float* input, float* output, int size) {
    // Swish activation: f(x) = x * sigmoid(x) = x / (1 + exp(-x))
    for (int i = 0; i < size; ++i) {
        float x = input[i];
        // Handle overflow cases
        if (x > 88.0f) {
            output[i] = x; // sigmoid(x) ≈ 1 for large x
        } else if (x < -88.0f) {
            output[i] = 0.0f; // sigmoid(x) ≈ 0 for very negative x
        } else {
            float sigmoid_x = 1.0f / (1.0f + expf(-x));
            output[i] = x * sigmoid_x;
        }
    }
}

} // namespace cmx::kernels::activations