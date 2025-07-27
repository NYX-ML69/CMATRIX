/**
 * @file cmx_gelu.cpp
 * @brief Implementation of GELU activation function
 */

#include "cmx_gelu.hpp"
#include <cmath>

namespace cmx::kernels::activations {

void gelu(const float* input, float* output, int size) {
    // GELU approximation: f(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
    const float sqrt_2_over_pi = 0.7978845608f; // √(2/π)
    const float coeff = 0.044715f;
    
    for (int i = 0; i < size; ++i) {
        float x = input[i];
        float x_cubed = x * x * x;
        float inner = sqrt_2_over_pi * (x + coeff * x_cubed);
        
        // Clamp to prevent overflow in tanh
        if (inner > 88.0f) {
            output[i] = x; // tanh(inner) ≈ 1
        } else if (inner < -88.0f) {
            output[i] = 0.0f; // tanh(inner) ≈ -1
        } else {
            output[i] = 0.5f * x * (1.0f + tanhf(inner));
        }
    }
}

} // namespace cmx::kernels::activations