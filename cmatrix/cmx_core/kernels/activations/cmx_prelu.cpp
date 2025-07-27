/**
 * @file cmx_prelu.cpp
 * @brief Implementation of Parametric ReLU activation function
 */

#include "cmx_prelu.hpp"

namespace cmx::kernels::activations {

void prelu(const float* input, float* output, int size, const float* alpha) {
    // PReLU activation: f(x) = x if x > 0, else alpha[i] * x
    for (int i = 0; i < size; ++i) {
        float x = input[i];
        output[i] = (x > 0.0f) ? x : alpha[i] * x;
    }
}

void prelu(const float* input, float* output, int size, float alpha) {
    // PReLU activation with single alpha: f(x) = x if x > 0, else alpha * x
    for (int i = 0; i < size; ++i) {
        float x = input[i];
        output[i] = (x > 0.0f) ? x : alpha * x;
    }
}

} // namespace cmx::kernels::activations