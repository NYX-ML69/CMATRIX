/**
 * @file cmx_relu.cpp
 * @brief Implementation of ReLU activation function
 */

#include "cmx_relu.hpp"

namespace cmx::kernels::activations {

void relu(const float* input, float* output, int size) {
    // ReLU activation: f(x) = max(0, x)
    for (int i = 0; i < size; ++i) {
        float x = input[i];
        output[i] = (x > 0.0f) ? x : 0.0f;
    }
}

} // namespace cmx::kernels::activations