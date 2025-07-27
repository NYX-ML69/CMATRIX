/**
 * @file cmx_leaky_relu.cpp
 * @brief Implementation of Leaky ReLU activation function
 */

#include "cmx_leaky_relu.hpp"

namespace cmx::kernels::activations {

void leaky_relu(const float* input, float* output, int size, float alpha) {
    // Leaky ReLU activation: f(x) = x if x > 0, else alpha * x
    for (int i = 0; i < size; ++i) {
        float x = input[i];
        output[i] = (x > 0.0f) ? x : alpha * x;
    }
}

} // namespace cmx::kernels::activations