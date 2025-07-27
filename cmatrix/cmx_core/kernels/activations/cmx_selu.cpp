/**
 * @file cmx_selu.cpp
 * @brief Implementation of SELU activation function
 */

#include "cmx_selu.hpp"
#include <cmath>

namespace cmx::kernels::activations {

void selu(const float* input, float* output, int size, float alpha, float scale) {
    // SELU activation: f(x) = scale * (x if x > 0, else alpha * (exp(x) - 1))
    for (int i = 0; i < size; ++i) {
        float x = input[i];
        if (x > 0.0f) {
            output[i] = scale * x;
        } else {
            // Clamp input to prevent overflow
            if (x < -88.0f) {
                output[i] = scale * (-alpha);
            } else {
                output[i] = scale * alpha * (expf(x) - 1.0f);
            }
        }
    }
}

} // namespace cmx::kernels::activations