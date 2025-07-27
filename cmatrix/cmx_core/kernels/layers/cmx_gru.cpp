#include "cmx_gru.hpp"
#include <cmath>
#include <algorithm>

namespace cmx {
namespace kernels {

CmxGRU::CmxGRU() 
    : configured_(false)
    , state_buffer_(nullptr)
    , temp_update_gate_(nullptr)
    , temp_reset_gate_(nullptr)
    , temp_new_gate_(nullptr)
    , temp_reset_hidden_(nullptr) {
    
    state_.hidden_state = nullptr;
    state_.initialized = false;
}

CmxGRU::~CmxGRU() {
    // Note: We don't free state_buffer_ as it's managed externally
}

bool CmxGRU::configure(const Config& config, void* state_buffer, size_t state_buffer_size) {
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

bool CmxGRU::run(const float* input, uint32_t sequence_length, float* output, float* final_state) {
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
            
            // Compute update and reset gates
            compute_gate(current_input, prev_hidden, config_.update_gate, temp_update_gate_);
            compute_gate(current_input, prev_hidden, config_.reset_gate, temp_reset_gate_);

            // Apply sigmoid activations to gates
            sigmoid_inplace(temp_update_gate_, hidden_size);
            sigmoid_inplace(temp_reset_gate_, hidden_size);

            // Compute new state candidate with reset gate
            compute_new_state(current_input, prev_hidden, temp_reset_gate_, 
                            config_.new_gate, temp_new_gate_);

            // Apply tanh activation to new state candidate
            tanh_inplace(temp_new_gate_, hidden_size);

            // Update hidden state: h_t = (1 - z_t) * h_{t-1} + z_t * n_t
            float* current_hidden = state_.hidden_state + b * hidden_size;
            linear_interpolate(prev_hidden, temp_new_gate_, temp_update_gate_, 
                             current_hidden, hidden_size);

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

bool CmxGRU::infer_shape(const uint32_t* input_shape, uint32_t* output_shape) {
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

void CmxGRU::reset_state() {
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

size_t CmxGRU::get_state_buffer_size() const {
    const uint32_t batch_size = config_.batch_size;
    const uint32_t hidden_size = config_.hidden_size;
    
    // Calculate required buffer size
    size_t size = 0;
    
    // Hidden state
    size += batch_size * hidden_size * sizeof(float);
    
    // Temporary gate buffers
    size += 4 * hidden_size * sizeof(float);  // update, reset, new, reset_hidden
    
    // Alignment padding
    size += 32; // Conservative alignment padding
    
    return size;
}

const float* CmxGRU::get_hidden_state() const {
    return configured_ ? state_.hidden_state : nullptr;
}

bool CmxGRU::set_initial_state(const float* initial_state) {
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

void CmxGRU::compute_gate(const float* input, const float* hidden, 
                         const GateConfig& gate_config, float* output) {
    const uint32_t input_size = config_.input_size;
    const uint32_t hidden_size = config_.hidden_size;

    // Initialize output with bias
    if (gate_config.bias) {
        std::memcpy(output, gate_config.bias, hidden_size * sizeof(float));
    } else {
        std::memset(output, 0, hidden_size * sizeof(float));
    }

    // Input contribution: W_i * x
    if (gate_config.input_weights) {
        for (uint32_t i = 0; i < hidden_size; ++i) {
            float sum = 0.0f;
            for (uint32_t j = 0; j < input_size; ++j) {
                sum += gate_config.input_weights[i * input_size + j] * input[j];
            }
            output[i] += sum;
        }
    }

    // Hidden contribution: W_h * h
    if (gate_config.hidden_weights) {
        for (uint32_t i = 0; i < hidden_size; ++i) {
            float sum = 0.0f;
            for (uint32_t j = 0; j < hidden_size; ++j) {
                sum += gate_config.hidden_weights[i * hidden_size + j] * hidden[j];
            }
            output[i] += sum;
        }
    }
}

void CmxGRU::compute_new_state(const float* input, const float* hidden, 
                              const float* reset_gate, const GateConfig& gate_config, 
                              float* output) {
    const uint32_t input_size = config_.input_size;
    const uint32_t hidden_size = config_.hidden_size;

    // Apply reset gate to hidden state
    elementwise_multiply(reset_gate, hidden, temp_reset_hidden_, hidden_size);

    // Initialize output with bias
    if (gate_config.bias) {
        std::memcpy(output, gate_config.bias, hidden_size * sizeof(float));
    } else {
        std::memset(output, 0, hidden_size * sizeof(float));
    }

    // Input contribution: W_i * x
    if (gate_config.input_weights) {
        for (uint32_t i = 0; i < hidden_size; ++i) {
            float sum = 0.0f;
            for (uint32_t j = 0; j < input_size; ++j) {
                sum += gate_config.input_weights[i * input_size + j] * input[j];
            }
            output[i] += sum;
        }
    }

    // Reset-gated hidden contribution: W_h * (r_t * h_{t-1})
    if (gate_config.hidden_weights) {
        for (uint32_t i = 0; i < hidden_size; ++i) {
            float sum = 0.0f;
            for (uint32_t j = 0; j < hidden_size; ++j) {
                sum += gate_config.hidden_weights[i * hidden_size + j] * temp_reset_hidden_[j];
            }
            output[i] += sum;
        }
    }
}

void CmxGRU::sigmoid_inplace(float* data, uint32_t size) {
    for (uint32_t i = 0; i < size; ++i) {
        data[i] = 1.0f / (1.0f + std::exp(-data[i]));
    }
}

void CmxGRU::tanh_inplace(float* data, uint32_t size) {
    for (uint32_t i = 0; i < size; ++i) {
        data[i] = std::tanh(data[i]);
    }
}

void CmxGRU::elementwise_multiply(const float* a, const float* b, float* result, uint32_t size) {
    for (uint32_t i = 0; i < size; ++i) {
        result[i] = a[i] * b[i];
    }
}

void CmxGRU::linear_interpolate(const float* a, const float* b, const float* alpha, 
                               float* result, uint32_t size) {
    for (uint32_t i = 0; i < size; ++i) {
        result[i] = (1.0f - alpha[i]) * a[i] + alpha[i] * b[i];
    }
}

bool CmxGRU::initialize_state_buffer() {
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

    // Allocate temporary gate buffers
    offset = align_offset(offset, sizeof(float));
    temp_update_gate_ = reinterpret_cast<float*>(buffer + offset);
    offset += hidden_size * sizeof(float);

    offset = align_offset(offset, sizeof(float));
    temp_reset_gate_ = reinterpret_cast<float*>(buffer + offset);
    offset += hidden_size * sizeof(float);

    offset = align_offset(offset, sizeof(float));
    temp_new_gate_ = reinterpret_cast<float*>(buffer + offset);
    offset += hidden_size * sizeof(float);

    offset = align_offset(offset, sizeof(float));
    temp_reset_hidden_ = reinterpret_cast<float*>(buffer + offset);
    offset += hidden_size * sizeof(float);

    return true;
}

} // namespace kernels
} // namespace cmx