/**
 * @file cmx_hard_swish.cpp
 * @brief Implementation of Hard Swish activation function
 */

#include "cmx_hard_swish.hpp"

namespace cmx::kernels::activations {

void hard_swish(const float* input, float* output, int size) {
    // Hard Swish activation: f(x) = x * hard_sigmoid(x)
    // where hard_sigmoid(x) = max(0, min(1, 0.2 * x + 0.5))
    for (int i = 0; i < size; ++i) {
        float x = input[i];
        float linear = 0.2f * x + 0.5f;
        
        // Compute hard_sigmoid(x)
        float hard_sigmoid_x;
        if (linear <= 0.0f) {
            hard_sigmoid_x = 0.0f;
        } else if (linear >= 1.0f) {
            hard_sigmoid_x = 1.0f;
        } else {
            hard_sigmoid_x = linear;
        }
        
        // Apply Hard Swish: x * hard_sigmoid(x)
        output[i] = x * hard_sigmoid_x;
    }
}

} // namespace cmx::kernels::activations