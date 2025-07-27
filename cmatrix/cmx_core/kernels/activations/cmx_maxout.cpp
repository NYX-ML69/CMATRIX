/**
 * @file cmx_maxout.cpp
 * @brief Implementation of Maxout activation function
 */

#include "cmx_maxout.hpp"

namespace cmx::kernels::activations {

void maxout(const float* input, float* output, int input_size, int group_size) {
    // Maxout activation: f(x) = max(x_1, x_2, ..., x_k) for each group
    int num_groups = input_size / group_size;
    
    for (int group = 0; group < num_groups; ++group) {
        int start_idx = group * group_size;
        float max_val = input[start_idx];
        
        // Find maximum in current group
        for (int i = 1; i < group_size; ++i) {
            float val = input[start_idx + i];
            if (val > max_val) {
                max_val = val;
            }
        }
        
        output[group] = max_val;
    }
}

void maxout_pairs(const float* input, float* output, int input_size) {
    // Maxout with group_size = 2: f(x) = max(x_1, x_2) for each pair
    int num_pairs = input_size / 2;
    
    for (int i = 0; i < num_pairs; ++i) {
        int idx1 = i * 2;
        int idx2 = i * 2 + 1;
        
        float val1 = input[idx1];
        float val2 = input[idx2];
        
        output[i] = (val1 > val2) ? val1 : val2;
    }
}

} // namespace cmx::kernels::activations