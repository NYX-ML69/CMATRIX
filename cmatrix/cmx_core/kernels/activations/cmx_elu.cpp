/**
 * @file cmx_elu.cpp
 * @brief Implementation of ELU activation function
 */

#include "cmx_elu.hpp"
#include <cmath>

namespace cmx::kernels::activations {

void elu(const float* input, float* output, int size, float alpha) {
    // ELU activation: f(x) = x if x > 0, else alpha * (exp(x) - 1)
    for (int i = 0; i < size; ++i) {
        float x = input[i];
        if (x > 0.0f) {
            output[i] = x;
        } else {
            // Clamp input to prevent overflow
            if (x < -88.0f) {
                output[i] = -alpha;
            } else {
                output[i] = alpha * (expf(x) - 1.0f);
            }
        }
    }
}

} // namespace cmx::kernels::activations