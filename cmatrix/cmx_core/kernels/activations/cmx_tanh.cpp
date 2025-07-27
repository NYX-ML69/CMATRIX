/**
 * @file cmx_tanh.cpp
 * @brief Implementation of hyperbolic tangent activation function
 */

#include "cmx_tanh.hpp"
#include <cmath>

namespace cmx::kernels::activations {

void tanh(const float* input, float* output, int size) {
    // Tanh activation: f(x) = tanh(x)
    for (int i = 0; i < size; ++i) {
        float x = input[i];
        // Clamp to prevent overflow
        if (x > 88.0f) {
            output[i] = 1.0f;
        } else if (x < -88.0f) {
            output[i] = -1.0f;
        } else {
            output[i] = tanhf(x);
        }
    }
}

} // namespace cmx::kernels::activations