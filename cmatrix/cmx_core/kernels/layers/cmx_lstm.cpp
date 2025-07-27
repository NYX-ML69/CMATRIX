#include "cmx_lstm.hpp"
#include <cmath>
#include <algorithm>

namespace cmx {
namespace kernels {

CmxLSTM::CmxLSTM() 
    : configured_(false)
    , state_buffer_(nullptr)
    , temp_input_gate_(nullptr)
    , temp_forget_gate_(nullptr)
    , temp_output_gate_(nullptr)
    , temp_cell_gate_(nullptr) {
    
    state_.hidden_state = nullptr;
    state_.cell_state = nullptr;
    state_.initialized = false;
}

CmxLSTM::~CmxLSTM() {
    // Note: We don't free state_buffer_ as it's managed externally
}

bool CmxLSTM::configure(const Config& config, void* state_buffer, size_t state_buffer_size) {
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

bool CmxLSTM::run(const float* input, uint32_t sequence_length, float* output) {
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
            
            // Compute gates
            compute_gate(current_input, prev_hidden, config_.input_gate, temp_input_gate_);
            compute_gate(current_input, prev_hidden, config_.forget_gate, temp_forget_gate_);
            compute_gate(current_input, prev_hidden, config_.output_gate, temp_output_gate_);
            compute_gate(current_input, prev_hidden, config_.cell_gate, temp_cell_gate_);

            // Apply activations
            sigmoid_inplace(temp_input_gate_, hidden_size);
            sigmoid_inplace(temp_forget_gate_, hidden_size);
            sigmoid_inplace(temp_output_gate_, hidden_size);
            tanh_inplace(temp_cell_gate_, hidden_size);

            // Update cell state: C_t = f_t * C_{t-1} + i_t * g_t
            float* current_cell = state_.cell_state + b * hidden_size;
            for (uint32_t i = 0; i < hidden_size; ++i) {
                current_cell[i] = temp_forget_gate_[i] * current_cell[i] + 
                                 temp_input_gate_[i] * temp_cell_gate_[i];
            }

            // Update hidden state: h_t = o_t * tanh(C_t)
            float* current_hidden = state_.hidden_state + b * hidden_size;
            for (uint32_t i = 0; i < hidden_size; ++i) {
                current_hidden[i] = temp_output_gate_[i] * std::tanh(current_cell[i]);
            }

            // Copy to output if needed
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

    return true;
}

bool CmxLSTM::infer_shape(const uint32_t* input_shape, uint32_t* output_shape) {
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

void CmxLSTM::reset_state() {
    if (!configured_) {
        return;
    }

    const uint32_t batch_size = config_.batch_size;
    const uint32_t hidden_size = config_.hidden_size;
    const uint32_t state_size = batch_size * hidden_size;

    // Zero initialize states
    std::memset(state_.hidden_state, 0, state_size * sizeof(float));
    std::memset(state_.cell_state, 0, state_size * sizeof(float));
    
    state_.initialized = true;
}

size_t CmxLSTM::get_state_buffer_size() const {
    const uint32_t batch_size = config_.batch_size;
    const uint32_t hidden_size = config_.hidden_size;
    
    // Calculate required buffer size
    size_t size = 0;
    
    // Hidden and cell states
    size += 2 * batch_size * hidden_size * sizeof(float);
    
    // Temporary gate buffers
    size += 4 * hidden_size * sizeof(float);
    
    // Alignment padding
    size += 32; // Conservative alignment padding
    
    return size;
}

const float* CmxLSTM::get_hidden_state() const {
    return configured_ ? state_.hidden_state : nullptr;
}

const float* CmxLSTM::get_cell_state() const {
    return configured_ ? state_.cell_state : nullptr;
}

void CmxLSTM::compute_gate(const float* input, const float* hidden, 
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

void CmxLSTM::sigmoid_inplace(float* data, uint32_t size) {
    for (uint32_t i = 0; i < size; ++i) {
        data[i] = 1.0f / (1.0f + std::exp(-data[i]));
    }
}

void CmxLSTM::tanh_inplace(float* data, uint32_t size) {
    for (uint32_t i = 0; i < size; ++i) {
        data[i] = std::tanh(data[i]);
    }
}

void CmxLSTM::elementwise_multiply(const float* a, const float* b, float* result, uint32_t size) {
    for (uint32_t i = 0; i < size; ++i) {
        result[i] = a[i] * b[i];
    }
}

void CmxLSTM::elementwise_add(const float* a, const float* b, float* result, uint32_t size) {
    for (uint32_t i = 0; i < size; ++i) {
        result[i] = a[i] + b[i];
    }
}

bool CmxLSTM::initialize_state_buffer() {
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

    // Allocate cell state
    offset = align_offset(offset, sizeof(float));
    state_.cell_state = reinterpret_cast<float*>(buffer + offset);
    offset += batch_size * hidden_size * sizeof(float);

    // Allocate temporary gate buffers
    offset = align_offset(offset, sizeof(float));
    temp_input_gate_ = reinterpret_cast<float*>(buffer + offset);
    offset += hidden_size * sizeof(float);

    offset = align_offset(offset, sizeof(float));
    temp_forget_gate_ = reinterpret_cast<float*>(buffer + offset);
    offset += hidden_size * sizeof(float);

    offset = align_offset(offset, sizeof(float));
    temp_output_gate_ = reinterpret_cast<float*>(buffer + offset);
    offset += hidden_size * sizeof(float);

    offset = align_offset(offset, sizeof(float));
    temp_cell_gate_ = reinterpret_cast<float*>(buffer + offset);
    offset += hidden_size * sizeof(float);

    return true;
}

} // namespace kernels
} // namespace cmx