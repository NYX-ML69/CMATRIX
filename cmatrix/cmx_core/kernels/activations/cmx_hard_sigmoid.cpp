/**
 * @file cmx_hard_sigmoid.cpp
 * @brief Implementation of Hard Sigmoid activation function
 */

#include "cmx_hard_sigmoid.hpp"

namespace cmx::kernels::activations {

void hard_sigmoid(const float* input, float* output, int size) {
    // Hard Sigmoid activation: f(x) = max(0, min(1, 0.2 * x + 0.5))
    for (int i = 0; i < size; ++i) {
        float x = input[i];
        float linear = 0.2f * x + 0.5f;
        
        // Clamp to [0, 1] range
        if (linear <= 0.0f) {
            output[i] = 0.0f;
        } else if (linear >= 1.0f) {
            output[i] = 1.0f;
        } else {
            output[i] = linear;
        }
    }
}

} // namespace cmx::kernels::activations