/**
 * @file cmx_softmax.cpp
 * @brief Implementation of Softmax activation function
 */

#include "cmx_softmax.hpp"
#include <cmath>

namespace cmx::kernels::activations {

void softmax(const float* input, float* output, int size) {
    // Softmax with numerical stability: f(x_i) = exp(x_i - max) / Σ(exp(x_j - max))
    
    // Find maximum value for numerical stability
    float max_val = input[0];
    for (int i = 1; i < size; ++i) {
        if (input[i] > max_val) {
            max_val = input[i];
        }
    }
    
    // Compute exp(x_i - max) and sum
    float sum = 0.0f;
    for (int i = 0; i < size; ++i) {
        float exp_val = expf(input[i] - max_val);
        output[i] = exp_val;
        sum += exp_val;
    }
    
    // Normalize by sum
    for (int i = 0; i < size; ++i) {
        output[i] /= sum;
    }
}

} // namespace cmx::kernels::activations