/**
 * @file cmx_linear.cpp
 * @brief Implementation of linear (identity) activation function
 */

#include "cmx_linear.hpp"

namespace cmx::kernels::activations {

void linear(const float* input, float* output, int size) {
    // Linear activation: f(x) = x
    // Simply copy input to output
    for (int i = 0; i < size; ++i) {
        output[i] = input[i];
    }
}

} // namespace cmx::kernels::activations