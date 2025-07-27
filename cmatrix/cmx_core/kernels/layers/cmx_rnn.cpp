#include "cmx_rnn.hpp"
#include <cmath>
#include <algorithm>

namespace cmx {
namespace kernels {

CmxRNN::CmxRNN() 
    : configured_(false)
    , state_buffer_(nullptr)
    , temp_output_(nullptr) {
    
    state_.hidden_state = nullptr;
    state_.initialized = false;
}

CmxRNN::~CmxRNN() {
    // Note: We don't free state_buffer_ as it's managed externally
}

bool CmxRNN::configure(const Config& config, void* state_buffer, size_t state_buffer_size) {
    if (!state_buffer || state_buffer_size < get_state_buffer_size()) {
        return false;
    }

    config_ = config;
    state_buffer_ = state_buffer;
    
    if (!initialize_state_buffer()) {
        return false;
    }

    reset_state();
    configured_ = true;
    return true;
}

bool CmxRNN::run(const float* input, uint32_t sequence_length, float* output, float* final_state) {
    if (!configured_ || !input || !output) {
        return false;
    }

    const uint32_t batch_size = config_.batch_size;
    const uint32_t input_size = config_.input_size;
    const uint32_t hidden_size = config_.hidden_size;

    // Process each timestep
    for (uint32_t t = 0; t < sequence_length; ++t) {
        for (uint32_t b = 0; b < batch_size; ++b) {
            const float* current_input = input + (t * batch_size + b) * input_size;
            const float* prev_hidden = state_.hidden_state + b * hidden_size;
            float* current_hidden = state_.hidden_state + b * hidden_size;

            // Compute RNN step
            compute_step(current_input, prev_hidden, current_hidden);

            // Copy to output if returning sequences
            if (config_.return_sequences) {
                float* output_ptr = output + (t * batch_size + b) * hidden_size;
                std::memcpy(output_ptr, current_hidden, hidden_size * sizeof(float));
            }
        }
    }

    // Copy final hidden state to output if not returning sequences
    if (!config_.return_sequences) {
        std::memcpy(output, state_.hidden_state, batch_size * hidden_size * sizeof(float));
    }

    // Copy final state if requested
    if (config_.return_state && final_state) {
        std::memcpy(final_state, state_.hidden_state, batch_size * hidden_size * sizeof(float));
    }

    return true;
}

bool CmxRNN::infer_shape(const uint32_t* input_shape, uint32_t* output_shape) {
    if (!input_shape || !output_shape) {
        return false;
    }

    // Input shape: [batch_size, sequence_length, input_size]
    // Output shape: [batch_size, sequence_length or 1, hidden_size]
    
    output_shape[0] = input_shape[0];  // batch_size
    output_shape[1] = config_.return_sequences ? input_shape[1] : 1;  // sequence_length
    output_shape[2] = config_.hidden_size;  // hidden_size

    return true;
}

void CmxRNN::reset_state() {
    if (!configured_) {
        return;
    }

    const uint32_t batch_size = config_.batch_size;
    const uint32_t hidden_size = config_.hidden_size;
    const uint32_t state_size = batch_size * hidden_size;

    // Zero initialize state
    std::memset(state_.hidden_state, 0, state_size * sizeof(float));
    
    state_.initialized = true;
}

size_t CmxRNN::get_state_buffer_size() const {
    const uint32_t batch_size = config_.batch_size;
    const uint32_t hidden_size = config_.hidden_size;
    
    // Calculate required buffer size
    size_t size = 0;
    
    // Hidden state
    size += batch_size * hidden_size * sizeof(float);
    
    // Temporary output buffer
    size += hidden_size * sizeof(float);
    
    // Alignment padding
    size += 16; // Conservative alignment padding
    
    return size;
}

const float* CmxRNN::get_hidden_state() const {
    return configured_ ? state_.hidden_state : nullptr;
}

bool CmxRNN::set_initial_state(const float* initial_state) {
    if (!configured_ || !initial_state) {
        return false;
    }

    const uint32_t batch_size = config_.batch_size;
    const uint32_t hidden_size = config_.hidden_size;
    const uint32_t state_size = batch_size * hidden_size;

    std::memcpy(state_.hidden_state, initial_state, state_size * sizeof(float));
    state_.initialized = true;
    
    return true;
}

void CmxRNN::set_activation(ActivationType activation) {
    config_.activation = activation;
}

void CmxRNN::compute_step(const float* input, const float* prev_hidden, float* output) {
    const uint32_t input_size = config_.input_size;
    const uint32_t hidden_size = config_.hidden_size;

    // Initialize output with bias
    if (config_.bias) {
        std::memcpy(temp_output_, config_.bias, hidden_size * sizeof(float));
    } else {
        std::memset(temp_output_, 0, hidden_size * sizeof(float));
    }

    // Input contribution: W_ih * x_t
    if (config_.input_weights) {
        for (uint32_t i = 0; i < hidden_size; ++i) {
            float sum = 0.0f;
            for (uint32_t j = 0; j < input_size; ++j) {
                sum += config_.input_weights[i * input_size + j] * input[j];
            }
            temp_output_[i] += sum;
        }
    }

    // Hidden contribution: W_hh * h_{t-1}
    if (config_.hidden_weights) {
        for (uint32_t i = 0; i < hidden_size; ++i) {
            float sum = 0.0f;
            for (uint32_t j = 0; j < hidden_size; ++j) {
                sum += config_.hidden_weights[i * hidden_size + j] * prev_hidden[j];
            }
            temp_output_[i] += sum;
        }
    }

    // Apply activation function
    apply_activation(temp_output_, hidden_size);

    // Copy to output
    std::memcpy(output, temp_output_, hidden_size * sizeof(float));
}

void CmxRNN::apply_activation(float* data, uint32_t size) {
    switch (config_.activation) {
        case ActivationType::TANH:
            tanh_inplace(data, size);
            break;
        case ActivationType::RELU:
            relu_inplace(data, size);
            break;
        case ActivationType::NONE:
            // No activation applied
            break;
    }
}

void CmxRNN::relu_inplace(float* data, uint32_t size) {
    for (uint32_t i = 0; i < size; ++i) {
        data[i] = std::max(0.0f, data[i]);
    }
}

void CmxRNN::tanh_inplace(float* data, uint32_t size) {
    for (uint32_t i = 0; i < size; ++i) {
        data[i] = std::tanh(data[i]);
    }
}

void CmxRNN::matvec_multiply(const float* matrix, const float* vector, 
                            const float* bias, float* result, 
                            uint32_t rows, uint32_t cols) {
    for (uint32_t i = 0; i < rows; ++i) {
        float sum = bias ? bias[i] : 0.0f;
        for (uint32_t j = 0; j < cols; ++j) {
            sum += matrix[i * cols + j] * vector[j];
        }
        result[i] = sum;
    }
}

void CmxRNN::vector_add(const float* a, const float* b, float* result, uint32_t size) {
    for (uint32_t i = 0; i < size; ++i) {
        result[i] = a[i] + b[i];
    }
}

bool CmxRNN::initialize_state_buffer() {
    if (!state_buffer_) {
        return false;
    }

    const uint32_t batch_size = config_.batch_size;
    const uint32_t hidden_size = config_.hidden_size;
    
    uint8_t* buffer = static_cast<uint8_t*>(state_buffer_);
    size_t offset = 0;

    // Align to 4-byte boundaries
    auto align_offset = [](size_t offset, size_t alignment) {
        return (offset + alignment - 1) & ~(alignment - 1);
    };

    // Allocate hidden state
    offset = align_offset(offset, sizeof(float));
    state_.hidden_state = reinterpret_cast<float*>(buffer + offset);
    offset += batch_size * hidden_size * sizeof(float);

    // Allocate temporary output buffer
    offset = align_offset(offset, sizeof(float));
    temp_output_ = reinterpret_cast<float*>(buffer + offset);
    offset += hidden_size * sizeof(float);

    return true;
}

} // namespace kernels
} // namespace cmx