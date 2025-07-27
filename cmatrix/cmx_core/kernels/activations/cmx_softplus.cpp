/**
 * @file cmx_softplus.cpp
 * @brief Implementation of Softplus activation function
 */

#include "cmx_softplus.hpp"
#include <cmath>

namespace cmx::kernels::activations {

void softplus(const float* input, float* output, int size) {
    // Softplus activation: f(x) = log(1 + exp(x))
    for (int i = 0; i < size; ++i) {
        float x = input[i];
        // For numerical stability and efficiency
        if (x > 88.0f) {
            // For large x, log(1 + exp(x)) ≈ x
            output[i] = x;
        } else if (x < -88.0f) {
            // For very negative x, log(1 + exp(x)) ≈ 0
            output[i] = 0.0f;
        } else {
            output[i] = logf(1.0f + expf(x));
        }
    }
}

} // namespace cmx::kernels::activations