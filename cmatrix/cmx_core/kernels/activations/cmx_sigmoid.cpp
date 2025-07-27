/**
 * @file cmx_sigmoid.cpp
 * @brief Implementation of sigmoid activation function
 */

#include "cmx_sigmoid.hpp"
#include <cmath>

namespace cmx::kernels::activations {

void sigmoid(const float* input, float* output, int size) {
    // Sigmoid activation: f(x) = 1 / (1 + exp(-x))
    for (int i = 0; i < size; ++i) {
        // Clamp input to prevent overflow in exp()
        float x = input[i];
        if (x > 88.0f) {
            output[i] = 1.0f;
        } else if (x < -88.0f) {
            output[i] = 0.0f;
        } else {
            output[i] = 1.0f / (1.0f + expf(-x));
        }
    }
}

} // namespace cmx::kernels::activations